# SciBmad.jl Documentation

This directory contains all documentation for SciBmad.jl, combining narrative documentation (Sphinx/MyST) with API reference (Documenter.jl).

## Building Documentation

### Prerequisites

```bash
# Python dependency
pip install -r docs/requirements.txt
# Julia dependency
julia --project=docs -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
```

### Build All (Combined Documenter + Sphinx) Documentation

```bash
# Build all documentation:
# This builds Documenter first (to generate `objects.inv` for intersphinx), then Sphinx, and combines them into `gh-pages/`.
python docs/build.py

# Open documentation with the command:
start gh-pages/index.html  # Windows
open gh-pages/index.html   # macOS
xdg-open gh-pages/index.html  # Linux
```

### Viewing Documentation Generated With a GitHub Pull Request

There is an "artifact" generated on GitHub when the documentation test is run for a pull request.
This artifact is a zip file containing the documentation and the artifact can be downloaded
to your local machine and viewed. To download, do the following:
- Go to the PR page.
- Click on any one of the tests.
- Near the upper left corner, click on the `Summary` button.
- Near the top, click on the `Artifacts` button.

Note: Artifacts get deleted by GitHub after 90 days.

### Only Build Julia API Reference (Documenter.jl)

If, for some reason, you only want to build the Julia documentation, do the following.
Note: Sphinx uses intersphinx to cross-reference into the API docs, so Documenter's
`objects.inv` must exist before building Sphinx below.

```bash
julia --project=docs docs/api/make.jl
```

Output: `docs/api/build/`

### Only Build Narrative Documentation (Sphinx)

If, for some reason, you only want to build Sphinx/Myst, do the following.
Note: Documenter's documentation must exist before building Sphinx.
```bash
cd docs
sphinx-build -b html src build/html
```

Output: `docs/build/html/`

## Directory Structure

```
docs/
├── src/                    # Narrative documentation (Sphinx/MyST)
│   ├── conf.py            # Sphinx configuration
│   ├── index.md           # Main landing page
│   ├── getting-started.md # Installation and basic usage
│   ├── user-guide/        # Detailed usage guides
│   ├── examples/          # Practical examples
│   ├── developer-guide/   # Contributing guidelines
│   ├── _static/           # CSS, images, and other static files
│   └── _templates/        # Custom HTML templates
├── api/                    # API reference (Documenter.jl)
│   ├── src/
│   │   ├── index.md       # API reference landing page
│   │   └── main-docs.md   # Redirect to main docs
│   └── make.jl            # Documenter build script
├── requirements.txt        # Python dependencies (Sphinx)
├── Project.toml           # Julia dependencies (Documenter)
└── README.md              # This file
```

## Contributing to Documentation

### Where to Add Content

| Type of Content | Location | Format |
|----------------|----------|--------|
| Installation guide | `src/getting-started.md` | Markdown (MyST) |
| Usage tutorials | `src/user-guide/*.md` | Markdown (MyST) |
| Examples | `src/examples/*.md` | Markdown (MyST) |
| Contributing guide | `src/developer-guide/*.md` | Markdown (MyST) |
| API docstrings | Source code (`src/*.jl`) | Julia docstrings |
| API organization | `api/src/index.md` | Markdown |

### Writing Narrative Documentation

Narrative docs use **MyST Markdown**, an enhanced Markdown with Sphinx directives.

**Basic example:**
```markdown
    # Section Title

    Regular markdown text with [links](https://example.com).

    ## Subsection

    ```julia
    # Code example
    qf = Quadrupole(Kn1=0.36, L=0.5)
    ```
```

**Math:**
Inline math: $E = mc^2$

Display math:
$$
\int_0^\infty e^{-x^2} dx = \frac{\sqrt{\pi}}{2}
$$

**Admonitions:**
```{note}
This is a note box.
```

```{warning}
This is a warning box.
```

**Resources:**
- [MyST Markdown Guide](https://myst-parser.readthedocs.io/)
- [Sphinx Documentation](https://www.sphinx-doc.org/)

### Embedding Jupyter notebooks

Jupyter notebooks are rendered with [MyST-NB](https://myst-nb.readthedocs.io/).
The `examples/` directory at the repo root is symlinked into `src/examples`, and
notebooks are added to a `{toctree}` like any other page (see `src/examples-index.md`):

```markdown
    ```{toctree}
    Nonlinear Twiss <examples/julia/nonlinear-twiss.ipynb>
    ```
```

Notes:
- Notebooks are committed **already executed** and rendered with their stored outputs
  (`nb_execution_mode = "off"` in `conf.py`) — the build never starts a kernel, so no
  Julia/IJulia is needed in CI.
- A notebook must have its own title (a leading `# Heading` markdown cell) or Sphinx
  won't create a navigation link to it.
- Notebooks that embed images as cell *attachments* (pasted images) don't render in a
  web build; save those as files first or exclude the notebook in `conf.py`.

### Beamlines.jl docstrings

SciBmad is built on [Beamlines.jl](https://github.com/bmad-sim/Beamlines.jl). Its
docstrings live on the Beamlines.jl site and are **not** re-rendered here — we only
cross-reference into them (via `DocumenterInterLinks` in `api/make.jl` and intersphinx
in `conf.py`). Don't add `@autodocs Modules = [Beamlines]` back to `api/src/index.md`.

### Writing API Documentation

API docs are auto-generated from Julia docstrings. Add docstrings to functions in `src/*.jl`:

```julia
    """
        Quadrupole(; Kn1=0.0, L=0.0, kwargs...)

    Create a quadrupole magnet element.

    # Arguments
    - `Kn1::Real`: Normalized quadrupole strength (1/m²)
    - `L::Real`: Length (m)
    - `kwargs...`: Additional LineElement parameters

    # Returns
    - `LineElement` with kind="Quadrupole"

    # Examples
    ```jldoctest
    julia> qf = Quadrupole(Kn1=0.36, L=0.5)
    LineElement(kind="Quadrupole", ...)
    ```
    """
    Quadrupole(; kwargs...) = LineElement(; kind="Quadrupole", kwargs...)
```

The docstrings automatically appear in the API reference.

## Cross-referencing Between Documentation Systems

### MyST → API (intersphinx)

Sphinx's intersphinx extension reads Documenter's `objects.inv` inventory to resolve
cross-references to specific API items. A `doctree-resolved` event handler in `conf.py`
rewrites the absolute URLs to relative paths so links work both locally and deployed.

A minimal Julia domain (`_JuliaDomain`) is registered in `conf.py` so Sphinx recognises
the `jl:type`, `jl:function`, `jl:method`, and `jl:macro` roles from the inventory.

**Link to the API landing page:**
```markdown
{external:doc}`API Reference <index>`
```

**Link to a specific type or function:**
```markdown
{jl:type}`SciBmad.BMultipoleParams`
{jl:function}`Custom text <SciBmad.Quadrupole>`
```

### API → MyST (plain links)

Documenter.jl doesn't support intersphinx, so use plain markdown links with relative
URLs. Since the API docs live under `api/` in the combined site, `../` reaches the
Sphinx site root:

```markdown
[Getting Started](../getting-started.html)
```

Documenter's sidebar also shows a "← Documentation" link (`docs/api/src/main-docs.md`)
that uses a JS redirect back to the main Sphinx docs.

### Sidebar navigation

- **Main docs (Sphinx):** sidebar shows "API Reference →" link
- **API reference (Documenter):** sidebar shows "← Documentation" link

Both systems are deployed as a unified site:
- Main docs at root: `https://bmad-sim.github.io/SciBmad.jl/`
- API reference: `https://bmad-sim.github.io/SciBmad.jl/api/`

## Automatic Deployment

Documentation is automatically built and deployed via GitHub Actions when:
- Code is pushed to `main` branch
- A tag is created
- Manually triggered via workflow dispatch

See `.github/workflows/documentation.yml` for details.

## Local Testing

Always test documentation builds locally before pushing:

1. **Test Documenter build** - Verify docstrings render correctly (`julia --project=docs docs/api/make.jl`)
2. **Test Sphinx build** - Verify no warnings/errors (`cd docs && sphinx-build -b html src build/html`)
3. **Test combined output** - Run `python docs/build.py` and verify cross-links work
4. **Check in browser** - Open `gh-pages/index.html`, verify formatting and navigation

## Questions?

- Check the [Sphinx documentation](https://www.sphinx-doc.org/)
- Check the [Documenter.jl documentation](https://documenter.juliadocs.org/)
- Ask in GitHub Discussions
