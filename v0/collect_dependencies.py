#!/usr/bin/env python3
"""Collect top-level third-party imports used by this project."""

from __future__ import annotations

import ast
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_MODULES = {"utils"}
IMPORT_TO_PACKAGE = {
    "camel": "camel-ai",
    "oasis": "camel-oasis",
    "pandas": "pandas",
}


def top_level_imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            imports.add(node.module.split(".", 1)[0])
    return imports


def main() -> int:
    stdlib = set(getattr(sys, "stdlib_module_names", ()))
    third_party: set[str] = set()

    for path in sorted(PROJECT_ROOT.glob("*.py")):
        if path.name == Path(__file__).name:
            continue
        third_party.update(
            name
            for name in top_level_imports(path)
            if name not in stdlib and name not in LOCAL_MODULES
        )

    packages = sorted(IMPORT_TO_PACKAGE.get(name, name) for name in third_party)
    print("\n".join(packages))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
