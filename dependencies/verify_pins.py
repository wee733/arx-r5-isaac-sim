# Copyright 2026 wee733
# SPDX-License-Identifier: Apache-2.0
"""Compare the dependency lock with the repository's submodule gitlinks.

The gitlink in the parent tree is what ``git clone`` will check out; the lock
is what the launcher and the Docker image label assert at runtime. They are
two records of one decision, so this script treats any disagreement as drift.

Reads the parent index/tree only, so it works before the submodules are
checked out (CI) and without network access.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCK_PATH = PROJECT_ROOT / 'dependencies' / 'autodata.lock.json'
GITLINK_MODE = '160000'


def _git(*argv: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ['git', '-C', str(cwd), *argv], text=True, capture_output=True, check=False,
    )


def load_lock(path: Path = LOCK_PATH) -> dict:
    """Read the reviewed lock; a malformed lock is a hard failure."""
    return json.loads(path.read_text())


def owner_relative(path: str, owner: str) -> str:
    """Strip the owning checkout prefix so git resolves the path in that repo."""
    return path if owner == '.' else path[len(owner) + 1:]


def checkouts(lock: dict) -> tuple[tuple[str, str, str, str], ...]:
    """Pair each checkout with its path, expected commit and owning checkout.

    A gitlink lives in the tree of the repository that contains it: Arena's is
    recorded by AutoData, Lab's by Arena. The owner is the first path segment
    chain that precedes the locked submodule path.
    """
    source = lock['source']
    entries = [('autodata', source['path'], source['commit'], '.')]
    owner = source['path']
    for name, submodule in lock['submodules'].items():
        entries.append((name, f"{source['path']}/{submodule['path']}",
                        submodule['commit'], owner))
        owner = f"{source['path']}/{submodule['path']}"
    return tuple(entries)


def _stage_gitlink(path: str, root: Path) -> str | None:
    """Read a gitlink from the index; used before the submodule is committed."""
    result = _git('ls-files', '--stage', '--', path, cwd=root)
    for line in result.stdout.splitlines():
        meta, _, name = line.partition('\t')
        mode, sha, stage = meta.split()
        if mode == GITLINK_MODE and stage == '0' and name == path:
            return sha
    return None


def _tree_gitlink(tree: str, path: str, root: Path) -> str | None:
    result = _git('ls-tree', tree, '--', path, cwd=root)
    if result.returncode != 0 or not result.stdout.strip():
        return None
    for line in result.stdout.splitlines():
        meta, _, name = line.partition('\t')
        mode, kind, sha = meta.split()
        if kind == 'commit' and mode == GITLINK_MODE and name == path:
            return sha
    return None


def gitlink(path: str, root: Path = PROJECT_ROOT, owner: Path | None = None) -> str | None:
    """Return the mode-160000 object recorded for *path*.

    The index answers first so an uncommitted submodule addition is visible;
    HEAD is the fallback for a clean clone. Nested checkouts are looked up in
    their owning repository's tree (AutoData owns Arena, Arena owns Lab), never
    in this repository's tree.
    """
    owner = root if owner is None else owner
    return _stage_gitlink(path, owner) or _tree_gitlink('HEAD', path, owner)


def checkout_head(path: str, root: Path = PROJECT_ROOT) -> str | None:
    """Return the checked-out HEAD, or None when the submodule is absent."""
    checkout = root / path
    # A missing submodule directory would otherwise let git resolve the parent HEAD.
    if not (checkout / '.git').exists():
        return None
    result = _git('rev-parse', 'HEAD', cwd=checkout)
    return result.stdout.strip() if result.returncode == 0 else None


def checkout_dirty(path: str, root: Path = PROJECT_ROOT) -> bool | None:
    """Report local edits in a checked-out vendor tree; None when absent."""
    checkout = root / path
    if not (checkout / '.git').exists():
        return None
    result = _git('status', '--porcelain', '--untracked-files=normal', cwd=checkout)
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def verify(
    lock: dict, root: Path = PROJECT_ROOT, require_checkout: bool = False,
    strict_clean: bool = False,
) -> dict:
    """Return the per-checkout evidence and the list of blocking problems."""
    problems: list[str] = []
    warnings: list[str] = []
    pins: dict[str, dict] = {}
    absent_owners: list[str] = []
    for name, path, expected, owner in checkouts(lock):
        owner_root = root / owner
        # Without the owning checkout there is no tree to read a nested gitlink
        # from; that is an uninitialised dependency, not drift.
        recorded = None if owner in absent_owners else gitlink(owner_relative(path, owner), owner_root)
        actual = checkout_head(path, root)
        if actual is None:
            absent_owners.append(path)
        dirty = checkout_dirty(path, root)
        pins[name] = {
            'path': path,
            'owner': owner,
            'expected': expected,
            'gitlink': recorded,
            'checkout': actual,
            'dirty': dirty,
        }
        if owner in absent_owners and owner != path:
            continue
        if recorded is None:
            problems.append(f'{name}: no gitlink recorded for {path}')
        elif recorded != expected:
            problems.append(f'{name}: gitlink {recorded} != lock {expected} ({path})')
        if actual is None:
            if require_checkout:
                problems.append(f'{name}: checkout missing at {path}')
            continue
        if actual != expected:
            problems.append(f'{name}: checkout HEAD {actual} != lock {expected} ({path})')
        if dirty:
            message = f'{name}: vendor checkout has local changes ({path})'
            (problems if strict_clean else warnings).append(message)
    return {
        'pins': pins,
        'problems': problems,
        'warnings': warnings,
        'source_ready': not problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--require-checkout', action='store_true',
                        help='fail when the submodules are not initialised')
    parser.add_argument('--strict-clean', action='store_true',
                        help='treat local changes in a vendor checkout as a failure')
    parser.add_argument('--json', action='store_true', help='emit the report as JSON')
    args = parser.parse_args(argv)

    report = verify(
        load_lock(args.project_root / 'dependencies/autodata.lock.json'),
        args.project_root, args.require_checkout, args.strict_clean,
    )
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        for name, pin in report['pins'].items():
            state = 'absent' if pin['checkout'] is None else pin['checkout']
            print(f"{name:8} expected {pin['expected']}  gitlink {pin['gitlink']}  checkout {state}")
        for warning in report['warnings']:
            print(f'warning: {warning}', file=sys.stderr)
        for problem in report['problems']:
            print(f'error: {problem}', file=sys.stderr)
        print('pins verified' if report['source_ready'] else 'pin drift detected')
    return 0 if report['source_ready'] else 1


if __name__ == '__main__':
    sys.exit(main())
