"""SARMS — check a repository's architecture from a Git URL.

NORMAL WORKFLOW
  uv run sarms auto <git-url> --tests "python -m pytest -q"
  uv run sarms auto <git-url> --tests "python -m pytest -q" --apply
  uv run sarms auto <git-url> --tests "python -m pytest -q" --apply --push

RESET ONE REPOSITORY
  uv run sarms clean <git-url> --yes

The default is a dry run. ``--apply`` enables verified fixes on the ``sarms/fixes``
branch; ``--push`` also pushes that branch to the repository's origin.

ADVANCED COMMANDS
  run, mine, certify-ledger, status, compile, check, replay, ledger, draft, learn,
  adopt, certify, verify-rules, dst

Results are stored automatically under data/<repository>/rms.
"""
from __future__ import annotations

import sys


def main() -> None:
    argv = sys.argv[1:]
    cmds = {"compile": "sarms.rules.compile", "check": "sarms.differ.diff", "replay": "sarms.differ.replay",
            "ledger": "sarms.ledger.build", "draft": "sarms.synthesize.draft",
            "learn": "sarms.rules.evidence", "adopt": "sarms.synthesize.learn",
            "certify": ("sarms.rules.certify", ["certify"]), "verify-rules": ("sarms.rules.certify", ["verify"]),
            "mine": ("sarms.product", ["mine"]), "certify-ledger": ("sarms.product", ["certify-ledger"]),
            "run": ("sarms.product", ["run"]), "auto": ("sarms.product", ["run"]),
            "clean": ("sarms.product", ["clean"]),
            "status": ("sarms.product", ["status"]), "dst": ("sarms.product", ["dst"])}
    if not argv or argv[0] in ("-h", "--help") or argv[0] not in cmds:
        print(__doc__)
        sys.exit(0 if argv and argv[0] in ("-h", "--help") else 2)
    import importlib
    target = cmds[argv[0]]
    modname, pre = (target, []) if isinstance(target, str) else target
    mod = importlib.import_module(modname)
    sys.argv = [f"sarms {argv[0]}", *pre, *argv[1:]]
    mod.main()
