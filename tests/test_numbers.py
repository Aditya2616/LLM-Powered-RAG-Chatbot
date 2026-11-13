import subprocess
import sys

from tests.conftest import ROOT


def test_readme_and_ui_data_match_the_results_files():
    r = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_numbers.py"), "--check"],
        capture_output=True,
        text=True,
        cwd=ROOT,
    )
    assert r.returncode == 0, r.stdout + r.stderr
