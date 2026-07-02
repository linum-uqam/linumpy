"""Tests for Phase 4 PR chain inventory validator (.planning/scripts/validate_pr_chain.py)."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VALIDATOR = REPO_ROOT / ".planning" / "scripts" / "validate_pr_chain.py"
GENERATOR = REPO_ROOT / ".planning" / "scripts" / "generate_pr_chain_inventory.py"


def _load_validator_module():
    spec = importlib.util.spec_from_file_location("validate_pr_chain", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_generator_module():
    spec = importlib.util.spec_from_file_location("generate_pr_chain_inventory", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_pr_chain_inventory"] = module
    spec.loader.exec_module(module)
    return module


def _header_order_body(order: tuple[int, ...]) -> str:
    order_str = " → ".join(f"#{n}" for n in order)
    return f"> **Stacked PR 1/24 — review order:** {order_str}\n\n---\n"


def _consistent_open_prs() -> list[dict[str, str | int]]:
    gen = _load_generator_module()
    canonical_order = (115, 97, *gen.CANONICAL_OPEN_PR_ORDER)
    return [
        {
            "number": pr_number,
            "body": _header_order_body(canonical_order),
            "headRefName": f"pr-test-{pr_number}",
            "title": f"PR {pr_number}",
        }
        for pr_number in gen.CANONICAL_OPEN_PR_ORDER
    ]


def _write_valid_manifest(path: Path, pr_number: int = 98) -> None:
    path.write_text(
        f"# PR #{pr_number} — test manifest\n\n"
        "## Intended scope\n\nTest scope.\n\n"
        "## Assigned commits\n\n<!-- stub -->\n\n"
        "## File ownership\n\n<!-- stub -->\n\n"
        "## Recorded base\n\nmain\n\n"
        "## Intended base\n\nmain\n\n"
        "## Drift flags\n\n"
        "- staleness (MEDIUM): 0 commits on dev ahead of branch tip\n"
        "- branch_commit_count: 0 commits since recorded base\n\n"
        "## Phase 5 verification checklist\n\n- [ ] stub\n",
        encoding="utf-8",
    )


def _write_valid_index(path: Path, *, include_open: bool = True, include_foundation: bool = False) -> None:
    mod = _load_validator_module()
    lines = [mod.CANONICAL_HEADER]
    if include_foundation:
        lines.append("115,pr-a-build-tooling,main,0,merged,0,")
        lines.append("97,pr-c-utility-preprocessing,pr-a-build-tooling,1,merged,0,")
    if include_open:
        for i, pr in enumerate(mod.CANONICAL_OPEN_PR_ORDER):
            lines.append(f"{pr},pr-test-{pr},base-{pr},{i + 2},open,0,")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@pytest.fixture
def fixture_repo(tmp_path: Path):
    """Minimal repo layout with valid rollup CSV and one manifest."""
    planning = tmp_path / ".planning"
    chain_dir = planning / "pr-chain"
    chain_dir.mkdir(parents=True)
    index_path = planning / "pr-chain-index.csv"
    manifest_path = chain_dir / "98-motor-stacking.md"
    _write_valid_manifest(manifest_path)
    _write_valid_index(index_path, include_open=False)
    return tmp_path


def test_structure_mode(fixture_repo: Path) -> None:
    """structure passes when rollup header and required manifest sections exist."""
    mod = _load_validator_module()
    reasons = mod.check_structure(fixture_repo)
    assert reasons == []


def test_structure_mode_fails_on_missing_section(fixture_repo: Path) -> None:
    """structure fails when a required manifest section is absent."""
    mod = _load_validator_module()
    manifest = fixture_repo / ".planning" / "pr-chain" / "98-motor-stacking.md"
    manifest.write_text("# PR #98\n\n## Intended scope\n\nOnly one section.\n", encoding="utf-8")
    reasons = mod.check_structure(fixture_repo)
    assert any("missing H2 section" in reason for reason in reasons)


def test_structure_mode_fails_on_bad_header(fixture_repo: Path) -> None:
    """structure fails when the rollup CSV header does not match the canonical schema."""
    mod = _load_validator_module()
    index_path = fixture_repo / ".planning" / "pr-chain-index.csv"
    index_path.write_text("wrong,header\n", encoding="utf-8")
    reasons = mod.check_structure(fixture_repo)
    assert any("header mismatch" in reason for reason in reasons)


def test_inventory_mode(tmp_path: Path) -> None:
    """inventory passes when all open PR manifests and foundation CSV rows exist."""
    mod = _load_validator_module()
    planning = tmp_path / ".planning"
    chain_dir = planning / "pr-chain"
    chain_dir.mkdir(parents=True)
    for pr in mod.CANONICAL_OPEN_PR_ORDER:
        _write_valid_manifest(chain_dir / f"{pr}-test-slug.md", pr_number=pr)
    _write_valid_index(planning / "pr-chain-index.csv", include_open=True, include_foundation=True)

    reasons = mod.check_inventory(tmp_path)
    assert reasons == []


def test_inventory_mode_fails_without_foundation_rows(tmp_path: Path) -> None:
    """inventory fails when merged foundation PRs are missing from the CSV."""
    mod = _load_validator_module()
    planning = tmp_path / ".planning"
    chain_dir = planning / "pr-chain"
    chain_dir.mkdir(parents=True)
    for pr in mod.CANONICAL_OPEN_PR_ORDER:
        _write_valid_manifest(chain_dir / f"{pr}-test-slug.md", pr_number=pr)
    _write_valid_index(planning / "pr-chain-index.csv", include_open=True, include_foundation=False)

    reasons = mod.check_inventory(tmp_path)
    assert any("foundation PR #115" in reason or "foundation PR #97" in reason for reason in reasons)


def test_validator_structure_red_before_inventory() -> None:
    """structure exits non-zero when the live inventory has not been generated yet."""
    assert VALIDATOR.is_file(), f"missing validator script: {VALIDATOR}"
    index_path = REPO_ROOT / ".planning" / "pr-chain-index.csv"
    if index_path.is_file():
        pytest.skip("inventory already exists — RED precondition not applicable")
    result = subprocess.run(
        [sys.executable, str(VALIDATOR), "structure"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode != 0


def _write_commit_map(path: Path, rows: list[dict[str, str]]) -> None:
    mod = _load_validator_module()
    lines = [mod.COMMIT_MAP_HEADER]
    lines.extend(
        f"{row['sha']},{row['pr_number']},{row.get('is_merge', 'false')},"
        f"{row.get('ambiguous', 'false')},{row.get('candidates', '')},"
        f"{row.get('assignment_reason', 'test')}"
        for row in rows
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_map_completeness(tmp_path: Path) -> None:
    """map mode detects gaps, duplicates, and orphan pr_numbers (hermetic dev set)."""
    mod = _load_validator_module()
    planning = tmp_path / ".planning"
    chain_dir = planning / "pr-chain"
    chain_dir.mkdir(parents=True)

    dev_shas = {
        "a" * 40,
        "b" * 40,
        "c" * 40,
    }
    _write_valid_manifest(chain_dir / "98-motor-stacking.md", pr_number=98)
    _write_valid_manifest(chain_dir / "99-gpu-acceleration.md", pr_number=99)

    rows = [
        {
            "sha": "a" * 40,
            "pr_number": "98",
            "ambiguous": "false",
            "candidates": "",
        },
        {
            "sha": "b" * 40,
            "pr_number": "99",
            "ambiguous": "true",
            "candidates": "98;99",
        },
        {
            "sha": "c" * 40,
            "pr_number": "98",
            "ambiguous": "false",
            "candidates": "",
        },
    ]
    _write_commit_map(chain_dir / "commit-map.csv", rows)

    assert mod.check_map(tmp_path, dev_shas=dev_shas) == []

    # Gap: remove one row
    gap_rows = rows[:2]
    _write_commit_map(chain_dir / "commit-map.csv", gap_rows)
    gap_reasons = mod.check_map(tmp_path, dev_shas=dev_shas)
    assert any("dev-but-not-map" in r for r in gap_reasons)

    # Duplicate SHA
    dup_rows = list(rows)
    dup_rows.append(dict(rows[0]))
    _write_commit_map(chain_dir / "commit-map.csv", dup_rows)
    dup_reasons = mod.check_map(tmp_path, dev_shas=dev_shas)
    assert any("duplicated SHAs" in r for r in dup_reasons)

    # Unknown pr_number without manifest

    orphan_rows = list(rows)
    orphan_rows[2] = dict(orphan_rows[2], pr_number="126")
    _write_commit_map(chain_dir / "commit-map.csv", orphan_rows)
    orphan_reasons = mod.check_map(tmp_path, dev_shas=dev_shas)
    assert any("no manifest file" in r for r in orphan_reasons)


def _write_stack_index(path: Path, *, break_link_at: int | None = None) -> dict[int, dict[str, str]]:
    mod = _load_validator_module()
    lines = [mod.CANONICAL_HEADER]
    lines.append("115,pr-a-build-tooling,main,0,merged,0,")
    lines.append("97,pr-c-utility-preprocessing,pr-a-build-tooling,1,merged,0,")

    index_by_pr: dict[int, dict[str, str]] = {}
    predecessor = "main"
    for i, pr in enumerate(mod.CANONICAL_OPEN_PR_ORDER):
        branch = f"pr-test-{pr}"
        intended = "wrong-base" if break_link_at == pr else predecessor
        row = {
            "pr_number": str(pr),
            "branch": branch,
            "intended_base": intended,
            "merge_position": str(i + 2),
            "status": "open",
            "commit_count": "0",
            "drift_flags": "staleness=0",
        }
        index_by_pr[pr] = row
        lines.append(f"{pr},{branch},{intended},{i + 2},open,0,staleness=0")
        predecessor = branch

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return index_by_pr


def test_stack_linearity(tmp_path: Path) -> None:
    """stack mode validates linear intended-base chain and D-12 end-state union."""
    mod = _load_validator_module()
    planning = tmp_path / ".planning"
    chain_dir = planning / "pr-chain"
    chain_dir.mkdir(parents=True)

    dev_shas = {
        "a" * 40,
        "b" * 40,
        "c" * 40,
    }
    dev_tip = "c" * 40

    for pr in mod.CANONICAL_OPEN_PR_ORDER:
        _write_valid_manifest(chain_dir / f"{pr}-test-slug.md", pr_number=pr)

    index_by_pr = _write_stack_index(planning / "pr-chain-index.csv")
    _write_commit_map(
        chain_dir / "commit-map.csv",
        [
            {"sha": "a" * 40, "pr_number": "98", "ambiguous": "false", "candidates": ""},
            {"sha": "b" * 40, "pr_number": "99", "ambiguous": "false", "candidates": ""},
            {"sha": dev_tip, "pr_number": "133", "ambiguous": "false", "candidates": ""},
        ],
    )

    assert mod.check_stack(tmp_path, dev_shas=dev_shas, index_by_pr=index_by_pr) == []

    broken_index = _write_stack_index(planning / "pr-chain-index.csv", break_link_at=99)
    broken_reasons = mod.check_stack(tmp_path, dev_shas=dev_shas, index_by_pr=broken_index)
    assert any("stack link broken" in reason for reason in broken_reasons)

    _write_stack_index(planning / "pr-chain-index.csv")
    gap_rows = [
        {"sha": "a" * 40, "pr_number": "98", "ambiguous": "false", "candidates": ""},
        {"sha": "b" * 40, "pr_number": "99", "ambiguous": "false", "candidates": ""},
    ]
    _write_commit_map(chain_dir / "commit-map.csv", gap_rows)
    end_state_reasons = mod.check_stack(tmp_path, dev_shas=dev_shas)
    assert any("dev-but-not-map" in reason or "dev tip" in reason for reason in end_state_reasons)


def _write_audit_section(path: Path, *, out_of_scope: int = 0, files: list[str] | None = None) -> None:
    text = path.read_text(encoding="utf-8")
    file_lines = ""
    if files:
        file_lines = "- out_of_scope_files:\n" + "".join(f"  - `{f}`\n" for f in files)
    audit = (
        "## Simulated Diff Audit\n\n"
        f"- method: contiguous\n"
        f"- commits: 1\n"
        f"- diff_files: {out_of_scope}\n"
        f"- out_of_scope (HIGH): {out_of_scope}\n"
        f"- missing: 0\n"
        f"{file_lines}\n"
    )
    if "## Simulated Diff Audit" in text:
        import re

        text = re.sub(
            r"## Simulated Diff Audit\n\n.*?(?=\n## Phase 5 verification checklist)",
            audit.rstrip() + "\n\n",
            text,
            count=1,
            flags=re.DOTALL,
        )
    else:
        text = text.replace(
            "## Phase 5 verification checklist",
            audit + "## Phase 5 verification checklist",
        )
    path.write_text(text, encoding="utf-8")


def test_audit_mode(tmp_path: Path) -> None:
    """audit mode requires report, manifest sections, and catalogued out_of_scope in index."""
    mod = _load_validator_module()
    planning = tmp_path / ".planning"
    chain_dir = planning / "pr-chain"
    chain_dir.mkdir(parents=True)

    for pr in mod.CANONICAL_OPEN_PR_ORDER:
        manifest = chain_dir / f"{pr}-test-slug.md"
        _write_valid_manifest(manifest, pr_number=pr)
        _write_audit_section(manifest, out_of_scope=0)

    index_path = planning / "pr-chain-index.csv"
    _write_stack_index(index_path)
    index_path.write_text(
        index_path.read_text(encoding="utf-8").replace(",staleness=0", ",staleness=0;out_of_scope=0"),
        encoding="utf-8",
    )
    (chain_dir / "audit-report.md").write_text("# audit rollup\n", encoding="utf-8")

    assert mod.check_audit(tmp_path) == []

    manifest_98 = chain_dir / "98-test-slug.md"
    text = manifest_98.read_text(encoding="utf-8")
    text = text.replace("## Simulated Diff Audit", "## Missing Audit")
    manifest_98.write_text(text, encoding="utf-8")
    missing_section = mod.check_audit(tmp_path)
    assert any("Simulated Diff Audit" in reason for reason in missing_section)

    _write_audit_section(manifest_98, out_of_scope=2, files=["extra.py", "other.py"])
    uncatalogued = mod.check_audit(tmp_path)
    assert any("not catalogued" in reason for reason in uncatalogued)


def test_all_mode_gate(tmp_path: Path) -> None:
    """all mode chains inventory, map, stack, and audit on a complete fixture set."""
    mod = _load_validator_module()
    planning = tmp_path / ".planning"
    chain_dir = planning / "pr-chain"
    chain_dir.mkdir(parents=True)

    dev_shas = {f"{i:040x}" for i in range(1, 4)}
    dev_tip = f"{3:040x}"

    for pr in mod.CANONICAL_OPEN_PR_ORDER:
        manifest = chain_dir / f"{pr}-test-slug.md"
        _write_valid_manifest(manifest, pr_number=pr)
        _write_audit_section(manifest, out_of_scope=0)

    _write_stack_index(planning / "pr-chain-index.csv")
    index_text = (planning / "pr-chain-index.csv").read_text(encoding="utf-8")
    (planning / "pr-chain-index.csv").write_text(
        index_text.replace(",staleness=0", ",staleness=0;out_of_scope=0"),
        encoding="utf-8",
    )

    _write_commit_map(
        chain_dir / "commit-map.csv",
        [
            {"sha": f"{1:040x}", "pr_number": "98", "ambiguous": "false", "candidates": ""},
            {"sha": f"{2:040x}", "pr_number": "99", "ambiguous": "false", "candidates": ""},
            {"sha": dev_tip, "pr_number": "133", "ambiguous": "false", "candidates": ""},
        ],
    )
    (chain_dir / "audit-report.md").write_text("# audit rollup\n", encoding="utf-8")

    assert mod.check_inventory(tmp_path) == []
    assert mod.check_map(tmp_path, dev_shas=dev_shas) == []
    assert mod.check_stack(tmp_path, dev_shas=dev_shas) == []
    assert mod.check_audit(tmp_path) == []


def test_header_consensus_accepts_consistent_orders() -> None:
    """validate_header_consensus returns canonical order when all PR bodies agree."""
    gen = _load_generator_module()
    canonical_order = gen.validate_header_consensus(_consistent_open_prs())
    expected = [115, 97, *gen.CANONICAL_OPEN_PR_ORDER]
    assert canonical_order == expected


def test_header_consensus_blocks_on_order_mismatch() -> None:
    """validate_header_consensus aborts BLOCKING when one PR header order differs."""
    gen = _load_generator_module()
    base_order = [115, 97, *gen.CANONICAL_OPEN_PR_ORDER]
    base_order[5], base_order[6] = base_order[6], base_order[5]
    bad_order = tuple(base_order)
    open_prs = _consistent_open_prs()
    open_prs[1] = {
        "number": 99,
        "body": _header_order_body(bad_order),
        "headRefName": "pr-test-99",
        "title": "PR 99",
    }
    with pytest.raises(SystemExit, match="BLOCKING: header sequence inconsistency"):
        gen.validate_header_consensus(open_prs)


def test_header_consensus_blocks_on_wrong_open_pr_count() -> None:
    """validate_header_consensus aborts when open PR count does not match canonical set."""
    gen = _load_generator_module()
    open_prs = [pr for pr in _consistent_open_prs() if pr["number"] not in gen.PASS_THROUGH_PRS]
    open_prs.append(dict(open_prs[0]))
    with pytest.raises(SystemExit, match="BLOCKING: expected 20 open PRs, got 21"):
        gen.validate_header_consensus(open_prs)


def test_header_consensus_accepts_pass_through_closed_slots() -> None:
    """validate_header_consensus allows 20 open PRs when 4 pass-through slots are CLOSED on GitHub."""
    gen = _load_generator_module()
    open_prs = [pr for pr in _consistent_open_prs() if pr["number"] not in gen.PASS_THROUGH_PRS]
    assert len(open_prs) == len(gen.CANONICAL_OPEN_PR_ORDER) - len(gen.PASS_THROUGH_PRS)
    canonical_order = gen.validate_header_consensus(open_prs)
    expected = [115, 97, *gen.CANONICAL_OPEN_PR_ORDER]
    assert canonical_order == expected


def test_header_consensus_blocks_when_non_pass_through_missing() -> None:
    """validate_header_consensus still requires every non-pass-through PR to be open."""
    gen = _load_generator_module()
    open_prs = [pr for pr in _consistent_open_prs() if pr["number"] not in gen.PASS_THROUGH_PRS and pr["number"] != 98]
    with pytest.raises(SystemExit, match="BLOCKING: missing open PRs"):
        gen.validate_header_consensus(open_prs)
