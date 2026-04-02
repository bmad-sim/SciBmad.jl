#!/usr/bin/env python3
"""Build and combine SciBmad.jl documentation."""

import subprocess
import shutil
from pathlib import Path

# Get directories
docs_dir = Path(__file__).parent
project_root = docs_dir.parent

# Build Documenter first (Sphinx intersphinx needs its objects.inv)
print("Building Documenter.jl documentation...")
result = subprocess.run(
    ["julia", f"--project={docs_dir}", "docs/api/make.jl"],
    cwd=project_root
)
if result.returncode != 0:
    exit(1)

# Build Sphinx
print("\nBuilding Sphinx documentation...")
result = subprocess.run(
    ["sphinx-build", "-b", "html", "src", "build/html"],
    cwd=docs_dir
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
