#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Block dependency changes unless the commit records an OSRB approval.

Runs at the ``commit-msg`` stage. If any staged file is a dependency
declaration (manifest, lockfile, Dockerfile, ...), the commit message must
carry an ``OSRB-Approved:`` trailer naming the approval (NVBug id, OSRB
ticket, or the link where the TAO Infra team approved the dependency).
The trailer lands in git history, so reviewers can see which approval covers
which dependency change. Commits without dependency changes are unaffected.
"""

import re
import subprocess
import sys

TRAILER = "OSRB-Approved"
CONTACT = "Please reach out to TAO Infra team for dependency change"

# Mirrors the former ``files:`` pattern from .pre-commit-config.yaml. It lives
# here because a commit-msg hook receives the message file, not the staged
# paths, so the hook selects the dependency files itself.
DEPENDENCY_FILE = re.compile(
    r"(?i)(^|/)("
    r"requirements[^/]*\.(txt|in)|constraints[^/]*\.txt|pyproject\.toml|setup\.py"
    r"|setup\.cfg|Pipfile|Pipfile\.lock|poetry\.lock|uv\.lock|environment[^/]*\.ya?ml"
    r"|conda[^/]*\.ya?ml|package\.json|package-lock\.json|npm-shrinkwrap\.json"
    r"|yarn\.lock|pnpm-lock\.yaml|bun\.lockb|go\.mod|go\.sum|Cargo\.toml|Cargo\.lock"
    r"|Gemfile|Gemfile\.lock|pom\.xml|build\.gradle(\.kts)?|settings\.gradle(\.kts)?"
    r"|gradle\.lockfile|composer\.json|composer\.lock|Package\.swift|Package\.resolved"
    r"|Podfile|Podfile\.lock|Cartfile|conanfile\.(txt|py)|vcpkg\.json|packages\.config"
    r"|paket\.dependencies|Chart\.ya?ml|Chart\.lock|Dockerfile|Dockerfile\.[^/]+"
    r"|[^/]+\.Dockerfile|[^/]+\.csproj|\.gitmodules"
    r")$"
)
TRAILER_LINE = re.compile(rf"^{TRAILER}:\s*(?P<ref>\S.*?)\s*$", re.IGNORECASE)


def staged_dependency_files():
    """Return the staged paths that declare dependencies."""
    out = subprocess.run(
        ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
        check=True, capture_output=True, text=True,
    ).stdout
    return sorted(p for p in out.splitlines() if p and DEPENDENCY_FILE.search(p))


def approval_reference(message_path):
    """Return the OSRB-Approved trailer value, or None when absent/empty."""
    with open(message_path, encoding="utf-8", errors="replace") as handle:
        for line in handle:
            if line.startswith("#"):
                continue  # git's commented commit template lines
            match = TRAILER_LINE.match(line.rstrip("\n"))
            if match:
                return match.group("ref")
    return None


def main(argv):
    """Fail when dependency files are staged without an OSRB-Approved trailer."""
    if not argv:
        print("usage: check_dependency_change.py <commit-message-file>")
        return 2
    paths = staged_dependency_files()
    if not paths:
        return 0
    reference = approval_reference(argv[0])
    if reference:
        print(f"Dependency change attested ({TRAILER}: {reference}):")
        for path in paths:
            print(f"  {path}")
        return 0
    print("Dependency change blocked. This commit modifies:")
    for path in paths:
        print(f"  {path}")
    print()
    print(CONTACT)
    print()
    print("Once the dependency is OSRB approved, record the approval in the commit")
    print("message and the commit will pass, e.g.:")
    print()
    print(f'  git commit --trailer "{TRAILER}: NVBug 1234567"')
    print(f'  git commit --trailer "{TRAILER}: <link to the TAO Infra approval>"')
    print()
    print("The trailer stays in git history so reviewers can see which approval")
    print("covers this change. Do not add it for dependencies that are not approved.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
