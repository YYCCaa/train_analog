"""pytest plugin: keep one round-robin shard of the collected tests (`--shard i/K`), used by tests/run_all.py.

    collected items, in collection order          --shard 1/3  (0-based i, K shards)
    index:  0   1   2   3   4   5   6      ──►    keep  index % 3 == 1  ->  1, 4
                                                  rest  -> pytest_deselected  (pytest prints "5 deselected")

    shard 0/3: 0 3 6   shard 1/3: 1 4   shard 2/3: 2 5    -> every test runs in exactly one shard

Load it with `-p run_all_shard`; its folder (tests/) must be importable, so tests/run_all.py prepends that folder to
its children's PYTHONPATH (the module name is unique on purpose: `--root` may point at a project whose own `tests`
package lacks this file). Without `--shard` the plugin does nothing; a malformed value (e.g. `3/2`, `a/b`, `1/0`) is
a usage error (pytest exit code 4).
"""

from __future__ import annotations

import argparse
import re

import pytest

# SHARD_RE: re.Pattern, accepted --shard syntax "<i>/<K>" (two non-negative decimal integers)
SHARD_RE = re.compile(r"(\d+)/(\d+)")


def parse_shard(text: str) -> tuple[int, int]:
    """Parse a --shard value `text` (str, "i/K") into (i, K) (tuple[int, int]) with K >= 1 and 0 <= i < K.
    Raises argparse.ArgumentTypeError otherwise (pytest turns it into a usage error, exit code 4)."""
    m = SHARD_RE.fullmatch(text)
    if not m:
        raise argparse.ArgumentTypeError(f"expected i/K (0-based shard i of K), got {text!r}")
    i, k = int(m.group(1)), int(m.group(2))  # int, int: shard index and shard count
    if k < 1 or i >= k:
        raise argparse.ArgumentTypeError(f"need K >= 1 and 0 <= i < K, got {text!r}")
    return i, k


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register `--shard i/K` (parsed to tuple[int, int], default None = run everything)."""
    parser.addoption("--shard", action="store", type=parse_shard, default=None, metavar="i/K",
                     help="run_all_shard: keep only the collected tests whose index % K == i (0-based i)")


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Keep items[index] with index % K == i (after -k/-m filtering, so shards stay balanced); report the others
    through pytest_deselected. `items`: list[pytest.Item], modified in place."""
    shard = config.getoption("shard")  # tuple[int, int] | None, (i, K) from --shard
    if shard is None:
        return
    i, k = shard
    keep = [item for idx, item in enumerate(items) if idx % k == i]  # list[pytest.Item], this shard's tests
    drop = [item for idx, item in enumerate(items) if idx % k != i]  # list[pytest.Item], the other shards' tests
    if drop:
        config.hook.pytest_deselected(items=drop)
    items[:] = keep
