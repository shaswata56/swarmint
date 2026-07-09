# Contributing to swarmint

Thanks for your interest! swarmint is an early research prototype, so there's a
lot of high-impact work available and the codebase is small enough to learn
quickly.

## License

By contributing, you agree that your contributions are licensed under the
project's **AGPL-3.0-or-later** license. This is a deliberate choice: swarmint
must stay open — anyone who runs it (including as a network service) or ships a
modified version must publish their source under the same terms. If you're
integrating swarmint commercially and AGPL is a blocker, open an issue to
discuss.

## Development setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # or bin/activate on unix
pip install -e ".[dev]"
python run_tests.py                                # fast suite (seconds)
SWARMINT_SLOW_TESTS=1 python run_tests.py          # + slow real-data tests
```

Python 3.10+. All pinned deps ship as prebuilt wheels (no compiler needed).

## Design invariants (please don't break these)

These are load-bearing; PRs that violate them will be asked to change:

1. **`SwarmNode` stays synchronous and representation-agnostic.** All async lives
   in the transport (`UdpBus`) and `NodeRunner`; all feature-space concerns live
   in the embedding layer. The deterministic simulator must keep reproducing its
   exact numbers (`run_tests.py` + `test_core.py`).
2. **The learning core is frozen behavior.** Merge / trust / quarantine /
   corroboration / aggregation are byte-for-byte stable; changes need a strong
   justification and updated results. Add new behavior behind opt-in flags with
   defaults unchanged.
3. **Shared-space invariant.** Any embedding/projection must be identical across
   all nodes (fit once at genesis, distributed frozen). Per-node fitting breaks
   prototype comparability — there's a test that proves it.
4. **No silent truncation.** If something caps coverage (top-N, budget,
   sampling), log what was dropped.

## Good first areas

- **Break the accuracy ceiling on few-shot data** (faces) — currently limited by
  the private-data-vs-shared-embedding tension (see the experimental record).
- **Adaptive Byzantine defense for overlapping classes** — the trust margin
  narrows when class distances overlap.
- **Real multi-NAT testing** — hole-punching is implemented and unit-tested on
  loopback, but never exercised against real NATs.
- **A learned metric** that stays shareable (the frozen-genesis constraint).
- **Second/third real datasets** to map where the approach holds and breaks.

## Conventions

- Match the surrounding style; keep comments explaining *why*, not *what*.
- Every new mechanism gets a test in `tests/test_*.py` (plain-script style with a
  `__main__` runner and, for adversarial features, a test that tries to defeat it).
- Report negative results honestly — this project treats them as first-class.
