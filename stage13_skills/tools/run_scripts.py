import subprocess
import sys
from pathlib import Path
from typing import List


def run_python_script(script_path: str, script_options: List[str] = []) -> dict:
    """
    run python script

    Args:
        script_path: path to the python script, e.g. example.py
        script_options: list of arguments to pass to the script
    """

    script_path = Path(script_path).resolve()

    cmd = [sys.executable, str(script_path)] + script_options

    result = subprocess.run(cmd, capture_output=True, text=True)

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
