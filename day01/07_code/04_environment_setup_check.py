"""
Day 01 — Environment Setup Check
==================================
Verify your environment is correctly configured for the program.
Run this before every day to catch issues early.

Run: python 04_environment_setup_check.py
"""

import sys
import os
import subprocess
import importlib
from typing import NamedTuple


class CheckResult(NamedTuple):
    name: str
    status: str  # "✓", "✗", "⚠"
    message: str


def check_python_version() -> CheckResult:
    """Python 3.11+ required for type hint features used in this program."""
    version = sys.version_info
    if version >= (3, 11):
        return CheckResult(
            "Python Version",
            "✓",
            f"Python {version.major}.{version.minor}.{version.micro}"
        )
    elif version >= (3, 9):
        return CheckResult(
            "Python Version",
            "⚠",
            f"Python {version.major}.{version.minor} — 3.11+ recommended"
        )
    else:
        return CheckResult(
            "Python Version",
            "✗",
            f"Python {version.major}.{version.minor} — upgrade to 3.11+"
        )


def check_pip() -> CheckResult:
    """pip must be available for package installation."""
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            version = result.stdout.split()[1]
            return CheckResult("pip", "✓", f"pip {version}")
        return CheckResult("pip", "✗", "pip not found")
    except Exception as e:
        return CheckResult("pip", "✗", str(e))


def check_git() -> CheckResult:
    """Git for version control."""
    try:
        result = subprocess.run(
            ["git", "--version"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            return CheckResult("Git", "✓", result.stdout.strip())
        return CheckResult("Git", "✗", "Git not found — install from git-scm.com")
    except FileNotFoundError:
        return CheckResult("Git", "✗", "Git not found — install from git-scm.com")


def check_virtual_env() -> CheckResult:
    """Verify running inside a virtual environment."""
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        venv_path = sys.prefix
        return CheckResult("Virtual Environment", "✓", f"Active: {venv_path}")
    return CheckResult(
        "Virtual Environment",
        "⚠",
        "No venv active — run: python -m venv venv && venv\\Scripts\\activate"
    )


def check_env_file() -> CheckResult:
    """Check for .env file."""
    if os.path.exists(".env"):
        return CheckResult(".env file", "✓", ".env file found")
    elif os.path.exists(".env.example"):
        return CheckResult(
            ".env file",
            "⚠",
            "No .env file — copy .env.example to .env and fill in keys"
        )
    return CheckResult(
        ".env file",
        "⚠",
        "No .env file — create one with your API keys"
    )


def check_api_keys() -> list[CheckResult]:
    """Check for LLM API keys."""
    # Try to load .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    results = []

    # Check each provider
    providers = [
        ("OPENAI_API_KEY", "OpenAI API Key"),
        ("ANTHROPIC_API_KEY", "Anthropic API Key"),
        ("GROQ_API_KEY", "Groq API Key (free)"),
    ]

    any_key_found = False
    for env_var, name in providers:
        value = os.getenv(env_var)
        if value and len(value) > 10:
            results.append(CheckResult(name, "✓", f"Set ({value[:8]}...)"))
            any_key_found = True
        else:
            results.append(CheckResult(name, "⚠", "Not set"))

    if not any_key_found:
        results.append(CheckResult(
            "LLM Access",
            "⚠",
            "No API key found. Get free key at console.groq.com OR install Ollama"
        ))

    return results


def check_ollama() -> CheckResult:
    """Check if Ollama is running (local LLM option)."""
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as resp:
            import json
            data = json.loads(resp.read())
            models = [m["name"] for m in data.get("models", [])]
            if models:
                return CheckResult(
                    "Ollama",
                    "✓",
                    f"Running. Models: {', '.join(models[:3])}"
                )
            return CheckResult(
                "Ollama",
                "⚠",
                "Running but no models pulled. Run: ollama pull llama3.2"
            )
    except Exception:
        return CheckResult(
            "Ollama",
            "⚠",
            "Not running (optional). Install from ollama.com for free local LLMs"
        )


def check_package(package_name: str, import_name: Optional[str] = None) -> CheckResult:
    """Check if a Python package is installed."""
    import_name = import_name or package_name
    try:
        module = importlib.import_module(import_name)
        version = getattr(module, "__version__", "installed")
        return CheckResult(package_name, "✓", f"v{version}")
    except ImportError:
        return CheckResult(
            package_name, "⚠",
            f"Not installed. Run: pip install {package_name}"
        )


from typing import Optional


def run_all_checks() -> None:
    """Run all environment checks and display results."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║        AI ENGINEER ENVIRONMENT CHECK                     ║")
    print("║        30-Day AI Engineer Program                        ║")
    print("╚══════════════════════════════════════════════════════════╝\n")

    sections: dict[str, list[CheckResult]] = {}

    # System checks
    system_checks = [
        check_python_version(),
        check_pip(),
        check_git(),
        check_virtual_env(),
    ]
    sections["System"] = system_checks

    # Environment checks
    env_checks = [check_env_file()] + check_api_keys() + [check_ollama()]
    sections["API Keys & Environment"] = env_checks

    # Day 1 packages
    day1_packages = [
        check_package("python-dotenv", "dotenv"),
        check_package("requests"),
        check_package("rich"),
    ]
    sections["Day 1 Packages"] = day1_packages

    # Later packages (informational — not needed until later days)
    later_packages = [
        check_package("numpy"),
        check_package("pandas"),
        check_package("scikit-learn", "sklearn"),
        check_package("torch"),
        check_package("transformers"),
        check_package("fastapi"),
        check_package("openai"),
        check_package("anthropic"),
        check_package("chromadb"),
        check_package("faiss-cpu", "faiss"),
    ]
    sections["Later Program Packages (preview)"] = later_packages

    # Display results
    total_ok = 0
    total_warn = 0
    total_fail = 0

    for section, checks in sections.items():
        print(f"  ── {section} ──")
        for check in checks:
            icon = check.status
            print(f"    {icon}  {check.name:<35} {check.message}")
            if check.status == "✓":
                total_ok += 1
            elif check.status == "⚠":
                total_warn += 1
            else:
                total_fail += 1
        print()

    # Summary
    total = total_ok + total_warn + total_fail
    print("─" * 60)
    print(f"  Results: {total_ok} OK  |  {total_warn} Warnings  |  {total_fail} Failed")
    print()

    if total_fail > 0:
        print("  ✗ Fix failed items before proceeding.")
    elif total_warn > 0:
        print("  ⚠ Warnings found. Address API keys to use LLM features.")
        print("  → Minimum: Get a free Groq API key at console.groq.com")
    else:
        print("  ✓ Environment fully configured. Ready to build AI systems!")

    print()
    print("  Next step: Run 01_ai_concepts_demo.py")
    print("  Then:      Run 03_ai_engineer_profile.py")


if __name__ == "__main__":
    run_all_checks()
