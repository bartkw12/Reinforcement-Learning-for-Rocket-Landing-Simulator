from pathlib import Path

import pytest

from lunarlander_rl.docs_sync import default_documents, render, stale, sync

REPO_ROOT = Path(__file__).resolve().parents[2]


def make_repo(tmp_path: Path) -> Path:
    (tmp_path / "results").mkdir()
    (tmp_path / "results" / "t.md").write_text("| a |\n| --- |\n| 1 |\n")
    (tmp_path / "docs").mkdir()
    return tmp_path


def test_render_replaces_the_block_and_keeps_the_rest(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    text = "intro\n<!-- include: results/t.md -->\nold\n<!-- end include -->\noutro\n"
    assert render(text, root) == (
        "intro\n<!-- include: results/t.md -->\n"
        "| a |\n| --- |\n| 1 |\n"
        "<!-- end include -->\noutro\n"
    )
    # Rendering is idempotent.
    assert render(render(text, root), root) == render(text, root)


def test_sync_and_check(tmp_path: Path) -> None:
    root = make_repo(tmp_path)
    doc = root / "docs" / "a.md"
    doc.write_text("<!-- include: results/t.md -->\n<!-- end include -->\n")
    (root / "README.md").write_text("no includes\n")
    documents = default_documents(root)
    assert stale(documents, root) == [doc]
    assert sync(documents, root) == [doc]
    assert stale(documents, root) == []
    # A changed result makes the document stale again.
    (root / "results" / "t.md").write_text("| a |\n| --- |\n| 2 |\n")
    assert stale(documents, root) == [doc]


def test_missing_include_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=r"nope\.md"):
        render("<!-- include: nope.md -->\n<!-- end include -->", tmp_path)


def test_repository_documents_are_in_sync() -> None:
    """Every table quoted in the README and docs/ matches results/tables/."""
    assert stale(default_documents(REPO_ROOT), REPO_ROOT) == []
