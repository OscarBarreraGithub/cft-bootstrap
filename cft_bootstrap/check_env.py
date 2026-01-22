#!/usr/bin/env python3
"""
Environment checker for CFT Bootstrap project.

Run this to verify all dependencies are properly installed:
    python check_env.py

Or for verbose output:
    python check_env.py -v
"""

import sys
import shutil
import subprocess
from pathlib import Path

# ANSI colors
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
RESET = "\033[0m"
BOLD = "\033[1m"


def check_mark(ok: bool) -> str:
    return f"{GREEN}✓{RESET}" if ok else f"{RED}✗{RESET}"


def warn_mark() -> str:
    return f"{YELLOW}⚠{RESET}"


def check_python_package(name: str, min_version: str | None = None) -> tuple[bool, str]:
    """Check if a Python package is installed and optionally meets minimum version."""
    try:
        mod = __import__(name)
        version = getattr(mod, "__version__", "unknown")
        if min_version and version != "unknown":
            from packaging import version as pkg_version
            if pkg_version.parse(version) < pkg_version.parse(min_version):
                return False, f"{version} (need >= {min_version})"
        return True, version
    except ImportError:
        return False, "not installed"


def check_command(cmd: str) -> tuple[bool, str]:
    """Check if a command is available in PATH."""
    path = shutil.which(cmd)
    if path:
        return True, path
    return False, "not found"


def check_sdpb() -> tuple[bool, str, str]:
    """
    Check SDPB availability via multiple methods.
    Returns: (available, method, details)
    """
    # Method 1: Direct binary in PATH
    path = shutil.which("sdpb")
    if path:
        try:
            result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=5)
            version = result.stdout.strip() or result.stderr.strip() or "unknown version"
            return True, "binary", f"{path} ({version})"
        except Exception:
            return True, "binary", path

    # Method 2: Docker image
    try:
        result = subprocess.run(
            ["docker", "images", "--format", "{{.Repository}}:{{.Tag}}"],
            capture_output=True, text=True, timeout=10
        )
        for line in result.stdout.splitlines():
            if "sdpb" in line.lower():
                return True, "docker", f"image: {line}"
    except Exception:
        pass

    # Method 3: Homebrew (macOS)
    if sys.platform == "darwin":
        try:
            result = subprocess.run(["brew", "list"], capture_output=True, text=True, timeout=10)
            if "sdpb" in result.stdout.lower():
                return True, "homebrew", "installed via Homebrew"
        except Exception:
            pass

    return False, "none", "not installed"


def check_wolfram() -> tuple[bool, str]:
    """Check for Wolfram kernel availability."""
    # Check PATH first
    path = shutil.which("WolframKernel") or shutil.which("wolframscript")
    if path:
        return True, path

    # Check common installation locations
    common_paths = [
        # macOS
        "/Applications/Wolfram.app/Contents/MacOS/WolframKernel",
        "/Applications/Mathematica.app/Contents/MacOS/WolframKernel",
        Path.home() / "Applications/Wolfram.app/Contents/MacOS/WolframKernel",
        # Linux
        "/usr/local/Wolfram/Mathematica/*/Executables/WolframKernel",
        "/opt/Wolfram/WolframEngine/*/Executables/WolframKernel",
        # Windows (via WSL or native)
        Path("C:/Program Files/Wolfram Research/Mathematica/*/WolframKernel.exe"),
    ]

    for p in common_paths:
        p = Path(p)
        if "*" in str(p):
            # Glob pattern
            matches = list(p.parent.glob(p.name))
            if matches:
                return True, str(matches[0])
        elif p.exists():
            return True, str(p)

    return False, "not found"


def main():
    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    print(f"\n{BOLD}CFT Bootstrap Environment Check{RESET}")
    print("=" * 40)

    all_ok = True
    warnings = []

    # Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_ok = sys.version_info >= (3, 10)
    print(f"\n{BOLD}Python{RESET}")
    print(f"  {check_mark(py_ok)} Python {py_version} {'(need >= 3.10)' if not py_ok else ''}")
    if verbose:
        print(f"      {sys.executable}")
    all_ok &= py_ok

    # Required Python packages
    print(f"\n{BOLD}Required Python Packages{RESET}")
    required = [
        ("numpy", "1.20"),
        ("scipy", "1.7"),
        ("matplotlib", "3.4"),
        ("cvxpy", "1.2"),
    ]
    for pkg, min_ver in required:
        ok, version = check_python_package(pkg, min_ver)
        print(f"  {check_mark(ok)} {pkg}: {version}")
        all_ok &= ok

    # Optional Python packages
    print(f"\n{BOLD}Optional Python Packages{RESET}")
    optional_py = [
        ("mpi4py", None, "MPI parallelism"),
        ("mpmath", None, "arbitrary precision arithmetic"),
    ]
    for pkg, min_ver, desc in optional_py:
        ok, version = check_python_package(pkg, min_ver)
        mark = check_mark(ok) if ok else warn_mark()
        status = version if ok else f"not installed ({desc})"
        print(f"  {mark} {pkg}: {status}")
        if not ok:
            warnings.append(f"{pkg} not installed - needed for {desc}")

    # SDPB
    print(f"\n{BOLD}SDPB (Semidefinite Program Bootstrap){RESET}")
    sdpb_ok, method, details = check_sdpb()
    if sdpb_ok:
        print(f"  {check_mark(True)} SDPB available via {method}")
        if verbose:
            print(f"      {details}")
    else:
        print(f"  {warn_mark()} SDPB not installed")
        print(f"      Install options:")
        print(f"        Docker: docker pull davidsd/sdpb:master")
        print(f"        HPC:    Singularity (see SDPB docs)")
        print(f"        Source: https://github.com/davidsd/sdpb")
        warnings.append("SDPB not installed - required for 60+ constraint runs")

    # Wolfram
    print(f"\n{BOLD}Wolfram Language{RESET}")
    wolfram_ok, wolfram_path = check_wolfram()
    if wolfram_ok:
        print(f"  {check_mark(True)} WolframKernel found")
        if verbose:
            print(f"      {wolfram_path}")
    else:
        print(f"  {warn_mark()} WolframKernel not found")
        print(f"      Required for Wolfram MCP server")
        warnings.append("WolframKernel not found - needed for symbolic computation")

    # Summary
    print(f"\n{'=' * 40}")
    if all_ok and not warnings:
        print(f"{GREEN}{BOLD}All required dependencies OK!{RESET}")
    elif all_ok:
        print(f"{YELLOW}{BOLD}Required dependencies OK, but some optional tools missing:{RESET}")
        for w in warnings:
            print(f"  {warn_mark()} {w}")
    else:
        print(f"{RED}{BOLD}Some required dependencies missing!{RESET}")
        print("Run: pip install -r requirements.txt")

    # SDPB-specific guidance
    if not sdpb_ok:
        print(f"\n{BOLD}To install SDPB (recommended for best results):{RESET}")
        print("  Docker (easiest): docker pull davidsd/sdpb:master")
        print("  Singularity (HPC): see https://github.com/davidsd/sdpb/blob/master/docs/Singularity.md")
        print("  From source: see https://github.com/davidsd/sdpb/blob/master/Install.md")

    print()
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
