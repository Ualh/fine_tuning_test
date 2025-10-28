"""Helpers to sanitise runtime probe output for batch parsing."""

from __future__ import annotations

import argparse
import re
import sys
from typing import Iterable, List

_ENV_LINE = re.compile(r"^[A-Z][A-Z0-9_]*=.*$")


def filter_env_lines(lines: Iterable[str]) -> List[str]:
    """Return only the KEY=VALUE lines from an iterable of text lines."""

    filtered: List[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if _ENV_LINE.match(line):
            filtered.append(line)
    return filtered


def parse_env(lines: Iterable[str]) -> dict[str, str]:
    """Convert probe output into a dictionary of environment variables."""

    env: dict[str, str] = {}
    for item in filter_env_lines(lines):
        key, value = item.split("=", 1)
        env[key] = value
    return env


def _read_lines_from_source(source: str | None) -> List[str]:
    if source:
        with open(source, "r", encoding="utf-8") as handle:
            return handle.read().splitlines()
    return sys.stdin.read().splitlines()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("source", nargs="?")
    parser.add_argument("-h", "--help", action="help")
    args = parser.parse_args(argv)

    lines = _read_lines_from_source(args.source)
    filtered = filter_env_lines(lines)

    if not filtered:
        sys.stderr.write("runtime_probe: no KEY=VALUE lines detected\n")
        return 1

    sys.stdout.write("\n".join(filtered) + "\n")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
