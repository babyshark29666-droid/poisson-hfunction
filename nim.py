"""NIM simulation entry point.

All interval calculations are implemented in the neighboring ``code.py``.
This script fixes the method to NIM and forwards the remaining CLI options.
"""

import importlib.util
import sys
from pathlib import Path


METHOD = "NIM"
CORE_PATH = Path(__file__).resolve().with_name("code.py")


def _load_core():
    if not CORE_PATH.is_file():
        raise FileNotFoundError(f"Shared simulation engine not found: {CORE_PATH}")

    module_name = "gpu_hf_shared_core_nim"
    spec = importlib.util.spec_from_file_location(module_name, CORE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load shared simulation engine: {CORE_PATH}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main(argv=None):
    """Run the NIM scenario with the shared simulation options."""
    args = list(sys.argv[1:] if argv is None else argv)
    # argparse also accepts unique prefixes such as --meth for --methods.
    if any(arg.startswith("--me") for arg in args):
        raise SystemExit(
            f"{Path(__file__).name} is fixed to {METHOD}; do not pass --methods."
        )

    core = _load_core()
    return core.main(["--methods", METHOD, *args])


if __name__ == "__main__":
    raise SystemExit(main())
