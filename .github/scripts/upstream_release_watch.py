#!/usr/bin/env python3
"""Open one issue per upstream patch release that is newer than the version this fork is on.

Usage
-----
    python3 upstream_release_watch.py pending --upstream OWNER/REPO --repo OWNER/REPO
    python3 upstream_release_watch.py prompt --upstream OWNER/REPO --version 1.108.7
    python3 upstream_release_watch.py report --upstream OWNER/REPO --repo OWNER/REPO \
        --version 1.108.7 --summary="$SUMMARY" --model-name NAME

`pending` prints a JSON list of the upstream `vX.Y.Z` tags that share X.Y with [project].version in
pyproject.toml, are newer than it, and have no issue in --repo yet. `prompt` prints the commit messages
between the fork's version and --version, trimmed to fit a small local model. `report` opens the issue for
--version: a keyword scan of the upstream commit messages for security fixes, the model's --summary of
them, and the commit list. It exits 1 after opening the issue when the summary is empty, so a broken model
never hides a release.
"""

from __future__ import annotations

import argparse
import itertools
import json
import pathlib
import re
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from typing import Final

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

Version = tuple[int, int, int]
Run = Callable[[Sequence[str]], str]

RELEASE_VERSION: Final = re.compile(r"v?([0-9]+)\.([0-9]+)\.([0-9]+)")
SECURITY_HINT: Final = re.compile(
    r"\bsecurity\b|vulnerab|\bCVE-[0-9]{4}-[0-9]+|\bGHSA(-[0-9a-z]{4}){3}|\bexploit|\bSSRF\b|\bXSS\b|\bRCE\b"
    r"|path traversal|\b(auth\w*|permission) bypass|privilege escalation|\b(sql|command|code) injection",
    re.IGNORECASE,
)
MENTION: Final = re.compile(r"@(?=\w)")
ISSUE_TITLE_SUBJECT: Final = "Upstream LiteLLM"
PROMPT_BUDGET_CHARS: Final = 36_000
BODY_EXCERPT_CHARS: Final = 400
LISTED_COMMITS: Final = 200


@dataclass(frozen=True, slots=True)
class ReportTarget:
    upstream: str
    repo: str
    model_name: str
    dry_run: bool


@dataclass(frozen=True, slots=True)
class Commit:
    sha: str
    message: str

    @property
    def subject(self) -> str:
        return self.message.split("\n", 1)[0].strip()

    @property
    def body(self) -> str:
        return " ".join(" ".join(self.message.split("\n", 1)[1:]).split())


def parse_version(text: str) -> Version | None:
    match: Final = RELEASE_VERSION.fullmatch(text)
    return None if match is None else (int(match[1]), int(match[2]), int(match[3]))


def format_version(version: Version) -> str:
    return ".".join(str(part) for part in version)


def issue_title(version: Version) -> str:
    return f"{ISSUE_TITLE_SUBJECT} v{format_version(version)} is available"


def newer_patch_releases(current: Version, ls_remote_output: str) -> tuple[Version, ...]:
    tags: Final = (line.rsplit("refs/tags/", 1)[-1] for line in ls_remote_output.splitlines())
    versions: Final = (parse_version(tag) for tag in tags)
    return tuple(sorted({v for v in versions if v is not None and v[:2] == current[:2] and v > current}))


def parse_commits(json_lines: str) -> tuple[Commit, ...]:
    rows: Final = (json.loads(line) for line in json_lines.splitlines() if line.strip())
    return tuple(Commit(sha=str(row["sha"]), message=str(row["message"])) for row in rows)


def security_hint(commit: Commit) -> str | None:
    return next((line.strip() for line in commit.message.splitlines() if SECURITY_HINT.search(line)), None)


def without_mentions(text: str) -> str:
    return MENTION.sub("@​", text)


def prompt_entries(commits: Iterable[Commit]) -> tuple[str, ...]:
    entries: Final = tuple(
        f"- {commit.subject}" + (f"\n  {commit.body[:BODY_EXCERPT_CHARS]}" if commit.body else "") for commit in commits
    )
    running_sizes: Final = itertools.accumulate(len(entry) + 1 for entry in entries)
    return tuple(entry for entry, size in zip(entries, running_sizes) if size <= PROMPT_BUDGET_CHARS)


def model_prompt(base: Version, new: Version, commits: Sequence[Commit]) -> str:
    entries: Final = prompt_entries(commits)
    return (
        f"Upstream moved from v{format_version(base)} to v{format_version(new)}. "
        f"These are {len(entries)} of its {len(commits)} commit messages.\n\n" + "\n".join(entries)
    )


def commit_line(commit: Commit) -> str:
    return f"- `{commit.sha[:10]}` {without_mentions(commit.subject)}"


def security_section(commits: Sequence[Commit]) -> str:
    hints: Final = ((commit, security_hint(commit)) for commit in commits)
    flagged: Final = tuple(
        commit_line(commit) + ("" if hint == commit.subject else f"\n  - {without_mentions(hint)}")
        for commit, hint in hints
        if hint is not None
    )
    if not flagged:
        return "No commit message in this range matched the keyword scan"
    return "\n".join(flagged)


def render_issue(
    upstream: str, base: Version, new: Version, commits: Sequence[Commit], summary: str | None, model_name: str
) -> str:
    base_tag: Final = f"v{format_version(base)}"
    new_tag: Final = f"v{format_version(new)}"
    listed: Final = commits[:LISTED_COMMITS]
    unlisted_note: Final = (
        f"\n- and {len(commits) - len(listed)} more, see the comparison link above"
        if len(commits) > len(listed)
        else ""
    )
    model_section: Final = (
        f"Written by {model_name} running in llama.cpp, from the commit messages only. "
        f"It can be wrong or incomplete\n\n{without_mentions(summary)}"
        if summary is not None
        else "The local model did not return a summary for this run"
    )
    return (
        f"Upstream released {new_tag} and this fork is on {base_tag}. "
        f"[Compare {base_tag} with {new_tag}](https://github.com/{upstream}/compare/{base_tag}...{new_tag}) "
        f"to see the {len(commits)} commits in between\n\n"
        "## Commits that mention security\n\n"
        "Found by matching the commit messages against words like security, vulnerability, CVE and GHSA. "
        "It is plain text matching, so read the commits before relying on it\n\n"
        f"{security_section(commits)}\n\n"
        "## Summary from the local model\n\n"
        f"{model_section}\n\n"
        "## All commits\n\n"
        f"<details>\n<summary>{len(commits)} commits</summary>\n\n"
        + "\n".join(commit_line(commit) for commit in listed)
        + f"{unlisted_note}\n\n</details>\n"
    )


def read_fork_version(pyproject: pathlib.Path) -> Version | None:
    with pyproject.open("rb") as f:
        return parse_version(str(tomllib.load(f)["project"]["version"]))


def existing_issue_titles(run: Run, repo: str) -> frozenset[str]:
    listing: Final = run(
        ("gh", "issue", "list", "--repo", repo, "--state", "all", "--limit", "1000")
        + ("--search", f'"{ISSUE_TITLE_SUBJECT}" in:title', "--json", "title")
    )
    return frozenset(str(issue["title"]) for issue in json.loads(listing))


def pending(run: Run, current: Version, upstream: str, repo: str, ignore_existing_issues: bool) -> tuple[Version, ...]:
    release_line: Final = f"v{current[0]}.{current[1]}.*"
    newer: Final = newer_patch_releases(
        current, run(("git", "ls-remote", "--tags", f"https://github.com/{upstream}", release_line))
    )
    if ignore_existing_issues or not newer:
        return newer
    taken: Final = existing_issue_titles(run, repo)
    return tuple(version for version in newer if issue_title(version) not in taken)


def fetch_commits(run: Run, upstream: str, base: Version, new: Version) -> tuple[Commit, ...]:
    compare: Final = f"repos/{upstream}/compare/v{format_version(base)}...v{format_version(new)}?per_page=100"
    return parse_commits(
        run(("gh", "api", "--paginate", compare, "--jq", ".commits[] | {sha: .sha, message: .commit.message}"))
    )


def report(run: Run, current: Version, new: Version, summary: str | None, target: ReportTarget) -> None:
    commits: Final = fetch_commits(run, target.upstream, current, new)
    body: Final = render_issue(target.upstream, current, new, commits, summary, target.model_name)
    if target.dry_run:
        print(f"# {issue_title(new)}\n\n{body}")  # noqa: T201  # the dry run's output is the issue it would open
    else:
        run(("gh", "issue", "create", "--repo", target.repo, "--title", issue_title(new), "--body", body))


def run_command(command: Sequence[str]) -> str:
    return subprocess.run(command, check=True, capture_output=True, text=True).stdout


def release_version(text: str) -> Version:
    version: Final = parse_version(text)
    if version is None:
        raise argparse.ArgumentTypeError(f"{text!r} is not an X.Y.Z release version")
    return version


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser: Final = argparse.ArgumentParser(description=__doc__)
    commands: Final = parser.add_subparsers(dest="command", required=True)
    pending_parser: Final = commands.add_parser("pending")
    prompt_parser: Final = commands.add_parser("prompt")
    report_parser: Final = commands.add_parser("report")
    for command in (pending_parser, prompt_parser, report_parser):
        command.add_argument("--pyproject", type=pathlib.Path, default=pathlib.Path("pyproject.toml"))
        command.add_argument("--upstream", required=True)
    for command in (pending_parser, report_parser):
        command.add_argument("--repo", required=True)
    for command in (prompt_parser, report_parser):
        command.add_argument("--version", type=release_version, required=True)
    pending_parser.add_argument("--ignore-existing-issues", action="store_true")
    report_parser.add_argument("--summary", required=True)
    report_parser.add_argument("--model-name", required=True)
    report_parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str], run: Run = run_command) -> int:
    args: Final = parse_args(argv)
    current: Final = read_fork_version(args.pyproject)
    if current is None:
        print(  # noqa: T201  # the ::error:: line to stderr is the workflow's failure signal
            f"::error::{args.pyproject} version is not an X.Y.Z release version", file=sys.stderr
        )
        return 1
    if args.command == "pending":
        versions: Final = pending(run, current, str(args.upstream), str(args.repo), bool(args.ignore_existing_issues))
        print(json.dumps([format_version(version) for version in versions]))  # noqa: T201  # stdout is the step output
        return 0
    new: Final[Version] = args.version
    if args.command == "prompt":
        commits: Final = fetch_commits(run, str(args.upstream), current, new)
        print(model_prompt(current, new, commits))  # noqa: T201  # stdout is the step output
        return 0
    summary: Final = str(args.summary).strip() or None
    target: Final = ReportTarget(
        upstream=str(args.upstream), repo=str(args.repo), model_name=str(args.model_name), dry_run=bool(args.dry_run)
    )
    report(run, current, new, summary, target)
    return 0 if summary is not None else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
