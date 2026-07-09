"""Run the swarmint test suite.

Tests are plain scripts (each `tests/test_*.py` has a __main__ runner), so no
pytest is required. This discovers and runs them in-process, reporting pass/fail.
Slow/opt-in tests (4096-dim faces) honor SWARMINT_SLOW_TESTS=1.

    python run_tests.py            # fast suite
    SWARMINT_SLOW_TESTS=1 python run_tests.py   # include slow real-data tests
"""

import importlib.util
import sys
import traceback
from pathlib import Path

TESTS = Path(__file__).parent / "tests"


def run_module(path: Path) -> tuple[bool, str]:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception:
        return False, traceback.format_exc()
    fns = [getattr(mod, n) for n in dir(mod) if n.startswith("test_") and callable(getattr(mod, n))]
    import asyncio
    for fn in fns:
        try:
            if asyncio.iscoroutinefunction(fn):
                asyncio.run(fn())
            else:
                fn()
        except Exception:
            return False, f"{fn.__name__}:\n{traceback.format_exc()}"
    return True, f"{len(fns)} tests"


def main() -> int:
    files = sorted(TESTS.glob("test_*.py"))
    if not files:
        print("no tests found")
        return 1
    failed = []
    for f in files:
        ok, detail = run_module(f)
        print(f"[{'PASS' if ok else 'FAIL'}] {f.name}  ({detail if ok else 'see below'})")
        if not ok:
            failed.append((f.name, detail))
    print("-" * 60)
    if failed:
        for name, detail in failed:
            print(f"\n=== {name} ===\n{detail}")
        print(f"\n{len(failed)}/{len(files)} test files FAILED")
        return 1
    print(f"all {len(files)} test files passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
