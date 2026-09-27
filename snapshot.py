"""
Record a live fetch once, replay it forever.

Every script here re-queries The Graph on each run. That is slow, burns the
free tier's quota, and -- worse for anything calling itself research -- means
no two runs are performed on the same data, so a result can never be exactly
reproduced or handed to someone else to check.

This records every subgraph response to a single gzipped JSON file, keyed by
(subgraph, query, variables), and replays it byte-for-byte afterwards. A
recorded snapshot is a reproducible artifact: commit it next to a result and
anyone can re-derive that result offline, with no API key.

    # record once (needs GRAPH_API_KEY)
    GRAPH_SNAPSHOT=snapshots/2026-09-26.json.gz GRAPH_SNAPSHOT_MODE=record \\
        python3 bad_debt_sweep.py

    # replay forever (no key, no network, identical numbers)
    GRAPH_SNAPSHOT=snapshots/2026-09-26.json.gz python3 bad_debt_sweep.py

`replay` is the default whenever GRAPH_SNAPSHOT points at a file that exists,
so the safe mode is the one you get by accident.

A replayed run is pinned to the moment of recording. That is the point, but
it also means a stale snapshot silently answers questions about a market that
has moved on -- so the age is printed on load, and loudly past a week.
"""

import gzip
import hashlib
import json
import os
import time
from typing import Dict, Optional

MODE_RECORD = "record"
MODE_REPLAY = "replay"
STALE_AFTER_SECONDS = 7 * 24 * 3600


def _key(subgraph_id: str, query: str, variables: Optional[dict]) -> str:
    blob = json.dumps({
        "subgraph": subgraph_id,
        "query": " ".join(query.split()),          # whitespace-insensitive
        "variables": variables or {},
    }, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


class Snapshot:
    """A recorded set of subgraph responses."""

    def __init__(self, path: str, mode: str):
        self.path = path
        self.mode = mode
        self.responses: Dict[str, dict] = {}
        self.meta: Dict[str, object] = {}
        self.hits = 0
        self.misses = 0
        self._live = None
        self._saved = False

        if mode == MODE_REPLAY:
            self.load()

    # -- persistence -------------------------------------------------------
    def load(self) -> None:
        opener = gzip.open if self.path.endswith(".gz") else open
        with opener(self.path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.responses = payload["responses"]
        self.meta = payload.get("meta", {})
        recorded_at = float(self.meta.get("recorded_at", 0))
        age = time.time() - recorded_at if recorded_at else None
        stamp = self.meta.get("recorded_at_iso", "unknown time")
        print(f"  [snapshot] replaying {len(self.responses)} responses recorded "
              f"{stamp}")
        if age and age > STALE_AFTER_SECONDS:
            print(f"  [snapshot] WARNING: this snapshot is {age/86400:.1f} days "
                  f"old. Every price, position and pool below is from then, not "
                  f"now -- re-record before quoting anything as current.")

    def save(self) -> None:
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self.meta.update({
            "recorded_at": time.time(),
            "recorded_at_iso": time.strftime("%Y-%m-%d %H:%M:%SZ", time.gmtime()),
            "response_count": len(self.responses),
        })
        opener = gzip.open if self.path.endswith(".gz") else open
        with opener(self.path, "wt", encoding="utf-8") as fh:
            json.dump({"meta": self.meta, "responses": self.responses}, fh)
        size_mb = os.path.getsize(self.path) / 1e6
        self._saved = True
        print(f"  [snapshot] recorded {len(self.responses)} responses to "
              f"{self.path} ({size_mb:.1f} MB)")

    # -- the transport ----------------------------------------------------
    def __call__(self, subgraph_id: str, query: str,
                 variables: Optional[dict] = None) -> dict:
        key = _key(subgraph_id, query, variables)

        if self.mode == MODE_REPLAY:
            try:
                self.hits += 1
                return self.responses[key]
            except KeyError:
                self.misses += 1
                raise RuntimeError(
                    "This query is not in the snapshot, so replaying it would "
                    "have to invent data:\n"
                    f"  variables: {variables}\n"
                    "The script changed its queries since the snapshot was "
                    "recorded. Re-record with GRAPH_SNAPSHOT_MODE=record."
                ) from None

        response = self._live(subgraph_id, query, variables)
        self.responses[key] = response
        return response


_active: Optional[Snapshot] = None


def active() -> Optional[Snapshot]:
    return _active


def install(live_transport):
    """Wrap `live_transport` with whatever GRAPH_SNAPSHOT asks for.

    Returns the transport to actually use. Called once by live_data at import.
    """
    global _active
    path = os.environ.get("GRAPH_SNAPSHOT")
    if not path:
        return live_transport

    mode = os.environ.get("GRAPH_SNAPSHOT_MODE")
    if not mode:
        # default to the safe direction: replay if we have one, else record
        mode = MODE_REPLAY if os.path.exists(path) else MODE_RECORD
    if mode not in (MODE_RECORD, MODE_REPLAY):
        raise ValueError(f"GRAPH_SNAPSHOT_MODE must be record or replay, got {mode!r}")

    if mode == MODE_REPLAY and not os.path.exists(path):
        raise FileNotFoundError(
            f"GRAPH_SNAPSHOT={path} does not exist. Record it first with "
            f"GRAPH_SNAPSHOT_MODE=record and a valid GRAPH_API_KEY.")

    snap = Snapshot(path, mode)
    snap._live = live_transport
    _active = snap

    if mode == MODE_RECORD:
        import atexit
        # only if the run did not already save explicitly, so a script that
        # calls save() itself does not write the file twice
        atexit.register(lambda: None if snap._saved else snap.save())
        print(f"  [snapshot] recording to {path}")

    return snap
