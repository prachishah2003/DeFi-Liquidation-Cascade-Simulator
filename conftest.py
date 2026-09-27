"""
Makes the test files importable by pytest from the repository root.

Every test file here also runs standalone (`python3 test_routing.py`) and
prints a readable trace -- that is the primary way to run them, because the
printed numbers are usually the point. pytest is for CI, where only pass or
fail matters.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
