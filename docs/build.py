#!/usr/bin/env python3
"""Build and combine SciBmad documentation."""

import os
import subprocess
import shutil
import sys
from pathlib import Path

# Every Julia subprocess below is launched with `--startup-file=no`. A personal
# ~/.julia/config/startup.jl (e.g. `using Revise`) is not part of the docs
# environment, so loading it would make the build fail on a developer machine for
# reasons unrelated to the docs. The IJulia kernel spec gets the same flag so the
# runnable pages execute in the same clean environment.

# Get directories
docs_dir = Path(__file__).parent
project_root = docs_dir.parent

# Instantiate the docs Julia environment
# `Pkg.develop` is what makes the runnable pages and the API docs build against this
# checkout rather than the registered release -- without it a local build can silently
# document a different version of SciBmad than the one you are editing.
print("Instantiating docs Julia environment...")
result = subprocess.run(
    ["julia", "--startup-file=no", f"--project={docs_dir}", "-e",
     "using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()"],
    cwd=project_root
)
if result.returncode != 0:
    exit(1)

# Register the IJulia kernel used by the "runnable" MyST pages. The kernel runs
# against docs/Project.toml, so `using SciBmad` resolves in every executed page.
print("\nInstalling IJulia kernel for runnable documentation pages...")
result = subprocess.run(
    ["julia", "--startup-file=no", f"--project={docs_dir}", "-e",
     'using IJulia; IJulia.installkernel("SciBmad Docs", '
     f'"--project={docs_dir}", "--startup-file=no"; specname="scibmad-docs", '
     'env=Dict("JULIA_NUM_THREADS" => "auto"))'],
    cwd=project_root
)
if result.returncode != 0:
    exit(1)

# Build Documenter first (Sphinx intersphinx needs its objects.inv)
print("Building Documenter.jl documentation...")
result = subprocess.run(
    ["julia", "--startup-file=no", f"--project={docs_dir}", "docs/api/make.jl"],
    cwd=project_root
)
if result.returncode != 0:
    exit(1)

# Install Sphinx dependencies
print("\nInstalling Sphinx dependencies...")
result = subprocess.run(
    ["pip", "install", "-r", "requirements.txt"],
    cwd=docs_dir
)
if result.returncode != 0:
    exit(1)

# Build Sphinx: the Julia version of the site
print("\nBuilding Sphinx documentation...")
result = subprocess.run(
    ["sphinx-build", "-b", "html", "src", "build/html"],
    cwd=docs_dir
)
if result.returncode != 0:
    exit(1)

# The Python version of the site (build/html/python), from the same pages: every Julia example
# has a Python version (docs/_ext/dual_language.py), which runs in a Python kernel with the Python
# package scibmad. It runs SciBmad in the docs Julia environment, so that both versions document
# this checkout. SCIBMAD_DOCS_LANGUAGES=julia skips it; PYSCIBMAD gives the pip requirement of
# scibmad (default: the PySciBmad repository; e.g. a local clone, `-e ../PySciBmad`).
if "python" in os.environ.get("SCIBMAD_DOCS_LANGUAGES", "julia,python").split(","):
    try:
        import scibmad  # noqa: F401
    except ImportError:
        spec = os.environ.get("PYSCIBMAD", "git+https://github.com/bmad-sim/PySciBmad.git")
        print(f"\nInstalling the Python package scibmad ({spec})...")
        result = subprocess.run([sys.executable, "-m", "pip", "install", *spec.split()])
        if result.returncode != 0:
            exit(1)
    print("\nBuilding the Python version of the Sphinx documentation...")
    env = dict(os.environ, SCIBMAD_JULIA=shutil.which("julia") or "julia",
               SCIBMAD_PROJECT=str(docs_dir.resolve()), SCIBMAD_THREADS="auto")
    result = subprocess.run(
        ["sphinx-build", "-b", "html", "-D", "docs_language=python",
         "-d", "build/doctrees-python", "src", "build/html/python"],
        cwd=docs_dir, env=env
    )
    if result.returncode != 0:
        exit(1)

# Combine into gh-pages
print("\nCombining documentation...")
gh_pages = project_root / "gh-pages"
if gh_pages.exists():
    shutil.rmtree(gh_pages)

gh_pages.mkdir()
shutil.copytree(docs_dir / "build" / "html", gh_pages, dirs_exist_ok=True)
shutil.copytree(docs_dir / "api" / "build", gh_pages / "api")

print(f"\nDone! Documentation built in {gh_pages}")
print(f"Open {gh_pages / 'index.html'} in your browser")
