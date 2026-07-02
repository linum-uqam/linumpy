"""Smoke tests for the scripts/subject/update_files.sh CLI contract.

These tests invoke the script via subprocess and never contact a real host:
a local fake nextflow.config fixture plus a dry-run mode exercise the CLI
contract without SSH/rsync.
"""

import os
import subprocess
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "subject" / "update_files.sh"


def _run(args: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(SCRIPT_PATH), *args],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )


def test_rejects_bad_subject_id() -> None:
    ret = _run(["not-a-subject-id"])
    output = ret.stdout + ret.stderr
    assert ret.returncode != 0
    assert "sub-" in output.lower()


def test_help_or_usage() -> None:
    ret = _run(["--help"])
    assert ret.returncode == 0
    assert "usage" in (ret.stdout + ret.stderr).lower()

    ret_no_args = _run([])
    assert ret_no_args.returncode != 0
    assert "usage" in (ret_no_args.stdout + ret_no_args.stderr).lower()


def test_dry_run_lists_outputs(tmp_path: Path) -> None:
    fake_config = tmp_path / "nextflow.config"
    fake_config.write_text(
        "params {\n"
        "    correct_bias_field = true\n"
        "    align_to_ras_enabled = false\n"
        "    common_space_preview = true\n"
        "    rehoming_diagnostics = false\n"
        "}\n"
    )
    env = dict(os.environ)
    env["UPDATE_FILES_CONFIG"] = str(fake_config)

    ret = _run(["sub-99", "--dry-run"], env=env)
    output = ret.stdout + ret.stderr

    assert ret.returncode == 0
    assert "common_space_previews" in output
    assert "correct_bias_field" in output
    assert "stack" in output
    assert "align_to_ras" not in output.replace("align_to_ras_enabled", "")
