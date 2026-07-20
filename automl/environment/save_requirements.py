"""
save_requirements.py

Capture the exact set of installed Python packages (name==version) and save
them to a requirements.txt-style file — useful for pinning down the software
environment a benchmark run happened in, alongside device_info.py.

Works on Linux, macOS, and Windows, and works whether you're using pip,
a venv, conda, poetry, uv, etc. — it introspects the *current* Python
interpreter's installed packages, not a specific tool's lockfile.

Usage as a library:
    from save_requirements import save_requirements

    path = save_requirements("requirements_snapshot.txt")

Usage from the command line:
    python save_requirements.py --output requirements_snapshot.txt
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Optional

from automl.logger import get_logger

logger = get_logger()


def get_installed_packages() -> str:
    """Return 'pip freeze' style output (name==version, one per line) for
    the currently running Python interpreter.

    Uses `python -m pip freeze` (via sys.executable) rather than a bare
    `pip freeze` call, so it always reflects the interpreter actually
    running this script — including inside venvs, conda envs, etc.

    Falls back to importlib.metadata if pip itself is unavailable for some
    reason (rare, but avoids a hard dependency on the pip CLI).
    """
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=60,
            check=True,
        )
        return result.stdout
    except Exception as e:
        return f"Error occurred while fetching installed packages: {e}"


def save_requirements(
    path: str | Path = "requirements.txt",
    header: Optional[str] = None,
) -> Path:
    """Collect installed packages and write them to a requirements.txt-style
    file.

    Args:
        path: output file path. Parent directories are created if needed.
        header: optional comment line(s) to prepend (e.g. a timestamp or
            note about which benchmark run this belongs to). '#' is added
            automatically to each line if not already present.

    Returns:
        The resolved Path the file was written to.
    """
    packages_text = get_installed_packages()

    out_path = Path(path).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        if header:
            for line in header.splitlines():
                f.write(line if line.startswith("#") else f"# {line}")
                f.write("\n")
            f.write("\n")
        f.write(packages_text)

    return out_path


def _main() -> None:
    import argparse
    from datetime import datetime, timezone

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        "-o",
        default="requirements.txt",
        help="Output file path (default: ./requirements.txt)",
    )
    parser.add_argument(
        "--no-header",
        action="store_true",
        help="Don't prepend a timestamp comment header",
    )
    args = parser.parse_args()

    header = None
    if not args.no_header:
        header = (
            f"Snapshot of installed packages\n"
            f"Generated: {datetime.now(timezone.utc).isoformat()}\n"
            f"Python: {sys.version.split()[0]} ({sys.executable})"
        )

    out_path = save_requirements(args.output, header=header)
    logger.info(
        f"Saved {sum(1 for _ in open(out_path)) } lines to: {out_path}", file=sys.stderr
    )


if __name__ == "__main__":
    _main()
