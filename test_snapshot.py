"""
Round-trip tests for the record/replay snapshot layer -- no network, no key.

The property that matters: a replayed run must be byte-identical to the run
that recorded it, and must REFUSE to answer a query it has never seen rather
than inventing one. A cache that silently returns something plausible for an
unrecorded query is worse than no cache, because the result still looks like
a result.
"""

import os
import tempfile

import snapshot


def _transport_factory(log):
    counter = {"n": 0}

    def transport(subgraph_id, query, variables=None):
        counter["n"] += 1
        log.append((subgraph_id, " ".join(query.split()), dict(variables or {})))
        return {"calls": counter["n"], "vars": variables or {},
                "sub": subgraph_id}
    return transport, counter


def test_record_then_replay_is_identical_and_offline():
    calls = []
    live, counter = _transport_factory(calls)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "snap.json.gz")

        os.environ["GRAPH_SNAPSHOT"] = path
        os.environ["GRAPH_SNAPSHOT_MODE"] = snapshot.MODE_RECORD
        recorder = snapshot.install(live)
        recorded = [recorder("aave", "query A($x: Int!) { a }", {"x": i})
                    for i in range(4)]
        recorder.save()
        assert counter["n"] == 4

        os.environ["GRAPH_SNAPSHOT_MODE"] = snapshot.MODE_REPLAY
        def exploding(*a, **k):
            raise AssertionError("replay must not touch the network")
        player = snapshot.install(exploding)
        replayed = [player("aave", "query A($x: Int!) { a }", {"x": i})
                    for i in range(4)]

        assert replayed == recorded, "replay diverged from the recording"
        assert counter["n"] == 4, "replay made a live call"
        print(f"  4 responses recorded, replayed identically, 0 network calls")


def test_replay_refuses_an_unrecorded_query():
    calls = []
    live, _ = _transport_factory(calls)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "snap.json.gz")
        os.environ["GRAPH_SNAPSHOT"] = path
        os.environ["GRAPH_SNAPSHOT_MODE"] = snapshot.MODE_RECORD
        rec = snapshot.install(live)
        rec("aave", "{ a }", {"x": 1})
        rec.save()

        os.environ["GRAPH_SNAPSHOT_MODE"] = snapshot.MODE_REPLAY
        player = snapshot.install(live)
        try:
            player("aave", "{ b }", {"x": 99})
        except RuntimeError as exc:
            assert "not in the snapshot" in str(exc)
            print("  unrecorded query refused rather than invented")
            return
        raise AssertionError("replay answered a query it never recorded")


def test_whitespace_in_a_query_does_not_break_the_key():
    calls = []
    live, counter = _transport_factory(calls)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "snap.json.gz")
        os.environ["GRAPH_SNAPSHOT"] = path
        os.environ["GRAPH_SNAPSHOT_MODE"] = snapshot.MODE_RECORD
        rec = snapshot.install(live)
        rec("aave", "query {\n  a\n}", {"x": 1})
        rec.save()

        os.environ["GRAPH_SNAPSHOT_MODE"] = snapshot.MODE_REPLAY
        player = snapshot.install(live)
        got = player("aave", "query {       a    }", {"x": 1})
        assert got["calls"] == 1 and counter["n"] == 1
        print("  reindenting a query still hits the same cache entry")


def test_defaults_to_replay_when_the_file_exists():
    calls = []
    live, _ = _transport_factory(calls)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "snap.json.gz")
        os.environ["GRAPH_SNAPSHOT"] = path
        os.environ.pop("GRAPH_SNAPSHOT_MODE", None)

        first = snapshot.install(live)          # no file yet -> record
        assert first.mode == snapshot.MODE_RECORD
        first("aave", "{ a }", {})
        first.save()

        second = snapshot.install(live)         # file exists -> replay
        assert second.mode == snapshot.MODE_REPLAY
        print("  safe default: records when absent, replays when present")


def test_no_snapshot_env_is_a_passthrough():
    os.environ.pop("GRAPH_SNAPSHOT", None)
    os.environ.pop("GRAPH_SNAPSHOT_MODE", None)
    sentinel = object()
    assert snapshot.install(sentinel) is sentinel
    print("  unset GRAPH_SNAPSHOT leaves the live transport untouched")


if __name__ == "__main__":
    saved = {k: os.environ.get(k) for k in ("GRAPH_SNAPSHOT", "GRAPH_SNAPSHOT_MODE")}
    try:
        tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
        for fn in tests:
            print(f"{fn.__name__}:")
            fn()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    print(f"\nAll {len(tests)} checks passed.")
