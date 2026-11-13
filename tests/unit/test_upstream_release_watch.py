"""Tests for .github/scripts/upstream_release_watch.py."""

import importlib.util
import json
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import pytest

_REPO_ROOT: Final = Path(__file__).resolve().parents[2]
_MODULE_PATH: Final = _REPO_ROOT / ".github" / "scripts" / "upstream_release_watch.py"
_spec: Final = importlib.util.spec_from_file_location("upstream_release_watch", _MODULE_PATH)
watch: Final = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = watch
_spec.loader.exec_module(watch)

_LS_REMOTE: Final = "\n".join(
    f"{index:040x}\trefs/tags/{tag}"
    for index, tag in enumerate(
        (
            "v1.108.5",
            "v1.108.6",
            "v1.108.7",
            "v1.108.7^{}",
            "v1.108.8-rc.1",
            "v1.108.9.dev2",
            "v1.108.10",
            "v1.108.11-stable",
            "v1.109.0",
        )
    )
)
_SECURITY_MERGE: Final = (
    "Merge pull request #7 from upstream/backport\n\nsecurity(proxy): backport #5 to the release line"
)
_COMMITS: Final = (
    ("a" * 40, "fix(router): retry on 429"),
    ("b" * 40, _SECURITY_MERGE),
    ("c" * 40, "fix(auth): close the GHSA-abcd-1234-wxyz key leak\n\nReported by @reporter"),
    ("d" * 40, "docs: explain prompt injection guardrails"),
)


@dataclass
class FakeGitHub:
    issue_titles: tuple[str, ...] = ()
    commits: tuple[tuple[str, str], ...] = _COMMITS
    calls: list[tuple[str, ...]] = field(default_factory=list)  # mutable-ok: records the commands the script ran

    def __call__(self, command: Sequence[str]) -> str:
        self.calls.append(tuple(command))
        match tuple(command[:3]):
            case ("git", "ls-remote", "--tags"):
                return _LS_REMOTE
            case ("gh", "issue", "list"):
                return json.dumps([{"title": title} for title in self._search(command[command.index("--search") + 1])])
            case ("gh", "api", "--paginate"):
                return "\n".join(json.dumps({"sha": sha, "message": message}) for sha, message in self.commits)
            case ("gh", "issue", "create"):
                return "https://github.example/issues/1\n"
        raise AssertionError(f"unexpected command: {command}")

    def _search(self, query: str) -> tuple[str, ...]:
        # GitHub issue search matches a quoted phrase on whole words only: `"Upstream LiteLLM v" in:title`
        # did not find "Upstream LiteLLM v1.108.7 is available" (observed on github.com, 2026-11-12)
        quoted_phrase: Final = query.split('"')[1]
        whole_words: Final = re.compile(rf"\b{re.escape(quoted_phrase)}\b")
        return tuple(title for title in self.issue_titles if whole_words.search(title))

    def created_issues(self) -> tuple[tuple[str, str], ...]:
        created: Final = (call for call in self.calls if call[:3] == ("gh", "issue", "create"))
        return tuple((call[call.index("--title") + 1], call[call.index("--body") + 1]) for call in created)


def _pyproject(tmp_path: Path, version: str = "1.108.6") -> str:
    pyproject: Final = tmp_path / "pyproject.toml"
    pyproject.write_text(f'[project]\nname = "litellm"\nversion = "{version}"\n', encoding="utf-8")
    return str(pyproject)


def _pending(tmp_path: Path, github: FakeGitHub, *extra: str) -> tuple[str, ...]:
    return ("pending", "--pyproject", _pyproject(tmp_path), "--upstream", "up/stream", "--repo", "my/fork", *extra)


def _prompt(tmp_path: Path) -> tuple[str, ...]:
    return ("prompt", "--pyproject", _pyproject(tmp_path), "--upstream", "up/stream", "--version", "1.108.7")


def _report(tmp_path: Path, *extra: str, summary: str = "ok") -> tuple[str, ...]:
    return (
        "report",
        *("--pyproject", _pyproject(tmp_path), "--upstream", "up/stream", "--repo", "my/fork"),
        *("--version", "1.108.7", "--model-name", "Tiny Model", f"--summary={summary}"),
        *extra,
    )


def test_pending_lists_only_newer_stable_releases_on_the_forks_release_line(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    github: Final = FakeGitHub()

    code: Final = watch.main(_pending(tmp_path, github), run=github)

    assert code == 0
    assert json.loads(capsys.readouterr().out) == ["1.108.7", "1.108.10"]


def test_pending_skips_a_release_that_already_has_an_issue_whatever_its_state(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    github: Final = FakeGitHub(issue_titles=(watch.issue_title((1, 108, 7)), "Upstream LiteLLM v1.108.1 looks odd"))

    watch.main(_pending(tmp_path, github), run=github)

    assert json.loads(capsys.readouterr().out) == ["1.108.10"]


def test_pending_can_ignore_existing_issues(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    github: Final = FakeGitHub(issue_titles=(watch.issue_title((1, 108, 7)),))

    watch.main(_pending(tmp_path, github, "--ignore-existing-issues"), run=github)

    assert json.loads(capsys.readouterr().out) == ["1.108.7", "1.108.10"]


def test_pending_is_empty_when_the_fork_is_on_the_newest_patch(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    github: Final = FakeGitHub()
    argv: Final = ("pending", "--pyproject", _pyproject(tmp_path, "1.108.10"), "--upstream", "u/p", "--repo", "m/f")

    watch.main(argv, run=github)

    assert json.loads(capsys.readouterr().out) == []


@pytest.mark.parametrize("version", ["1.108", "1.108.6rc1", "1.108.6.post1"])
def test_a_fork_version_that_is_not_a_release_fails_without_touching_github(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], version: str
) -> None:
    github: Final = FakeGitHub()
    argv: Final = ("pending", "--pyproject", _pyproject(tmp_path, version), "--upstream", "u/p", "--repo", "m/f")

    code: Final = watch.main(argv, run=github)

    captured: Final = capsys.readouterr()
    assert (code, captured.out, github.calls) == (1, "", [])
    assert captured.err.startswith("::error::")


def test_prompt_prints_every_commit_message_and_the_version_range(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    github: Final = FakeGitHub()

    code: Final = watch.main(_prompt(tmp_path), run=github)

    prompt: Final = capsys.readouterr().out
    assert code == 0
    assert "v1.108.6" in prompt and "v1.108.7" in prompt
    assert all(message.split("\n", 1)[0] in prompt for _, message in _COMMITS)
    assert "backport #5 to the release line" in prompt
    assert github.created_issues() == ()


def test_report_opens_an_issue_with_the_security_commits_and_the_model_summary(tmp_path: Path) -> None:
    github: Final = FakeGitHub()
    summary_text: Final = "### What changed\n- retries\n\n### Security\n- key leak closed"

    code: Final = watch.main(_report(tmp_path, summary=summary_text), run=github)

    ((title, body),) = github.created_issues()
    security, _, rest = body.partition("## Summary from the local model")
    summary, _, all_commits = rest.partition("## All commits")
    assert code == 0
    assert title == watch.issue_title((1, 108, 7))
    assert "https://github.com/up/stream/compare/v1.108.6...v1.108.7" in body
    assert "security(proxy): backport #5 to the release line" in security
    assert "GHSA-abcd-1234-wxyz" in security
    assert "retry on 429" not in security
    assert "prompt injection guardrails" not in security
    assert "key leak closed" in summary
    assert "Tiny Model" in summary
    assert all(message.split("\n", 1)[0] in all_commits for _, message in _COMMITS)


@pytest.mark.parametrize("summary", ["", "  \n "])
def test_report_still_opens_the_issue_and_exits_1_when_the_model_gave_no_summary(tmp_path: Path, summary: str) -> None:
    github: Final = FakeGitHub()

    code: Final = watch.main(_report(tmp_path, summary=summary), run=github)

    ((_, body),) = github.created_issues()
    assert code == 1
    assert "did not return a summary" in body
    assert "GHSA-abcd-1234-wxyz" in body


def test_report_never_writes_a_live_mention_from_a_commit_or_from_the_model(tmp_path: Path) -> None:
    github: Final = FakeGitHub(commits=(("e" * 40, "fix: thanks @someone for the security report"),))

    watch.main(_report(tmp_path, summary="- cc @maintainers please review"), run=github)

    ((_, body),) = github.created_issues()
    assert "@someone" not in body and "@maintainers" not in body
    assert "someone for the security report" in body and "maintainers please review" in body


def test_report_dry_run_prints_the_issue_and_creates_nothing(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    github: Final = FakeGitHub()

    code: Final = watch.main(_report(tmp_path, "--dry-run", summary="all good"), run=github)

    out: Final = capsys.readouterr().out
    assert code == 0
    assert github.created_issues() == ()
    assert watch.issue_title((1, 108, 7)) in out and "all good" in out


def test_a_release_too_large_for_the_model_is_trimmed_in_the_prompt_but_not_lost_from_the_issue(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    commits: Final = tuple((f"{index:040x}", f"fix: change {index}\n\n" + "x" * 1000) for index in range(400))
    github: Final = FakeGitHub(commits=commits)

    watch.main(_prompt(tmp_path), run=github)
    watch.main(_report(tmp_path), run=github)

    prompt: Final = capsys.readouterr().out
    ((_, body),) = github.created_issues()
    assert len(prompt) < watch.PROMPT_BUDGET_CHARS + 500
    assert "fix: change 0" in prompt
    assert "400 commits" in body
    assert f"and {400 - watch.LISTED_COMMITS} more" in body
    assert len(body) < 65536
