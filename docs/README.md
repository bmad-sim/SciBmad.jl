# SciBmad Documentation

This directory contains all documentation for SciBmad, combining narrative documentation (Sphinx/MyST) with API reference (Documenter.jl).

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
│   ├── index.md           # GENERATED landing page (README + toctree.md), gitignored
│   ├── installation.md    # Installing Julia and SciBmad
│   ├── quickstart.md      # Runnable tour of the package
│   ├── contents.md        # Whole-site tree, built by the `{sitetoc}` directive
│   ├── element.md         # Defining a LineElement (incl. parameter groups)
│   ├── beamline.md        # Defining a Beamline
│   ├── defexpr.md         # Deferred expressions and Contexts
│   ├── track.md           # Tracking, callbacks, CPU/GPU parallelization
│   ├── twiss.md           # The twiss docstring
│   ├── tracking-methods.md
│   ├── co.md              # (GPU-)batched closed orbit finder
│   ├── batch.md           # BatchParams / parameter scans
│   ├── timedependent.md   # Time-dependent parameters and ramping
│   ├── parametric-nf.md   # Parametric normal form
│   ├── optimize.md        # Optimization with autodiff
│   ├── dynamic-aperture.md, fma.md, collective.md
│   ├── coordinates.md, sagancavity-physics.md, sagancavity-tracking.md  # physics
│   ├── governance.md      # symlink to ../../GOVERNANCE.md
│   ├── examples/          # symlink to ../../examples (Jupyter notebooks)
│   └── _static/           # CSS, images, and other static files
├── _ext/                   # Sphinx extensions written for this site
│   ├── juliadocstrings.py # the `{docstring}`, `{notebook}` and `{sitetoc}` directives
│   └── docstring_server.jl# Julia side of `{docstring}`
├── toctree.md              # Site navigation, appended to the README
├── api/                    # API reference (Documenter.jl)
│   ├── src/
│   │   ├── index.md       # API reference landing page
│   │   └── main-docs.md   # Redirect to main docs
│   └── make.jl            # Documenter build script
├── requirements.txt        # Python dependencies (Sphinx)
├── Project.toml           # Julia dependencies (Documenter, NonlinearSolve, IJulia)
└── README.md              # This file
```

### The landing page and the navigation

`src/index.md` is **generated** and gitignored: `src/conf.py` writes it on every build by
concatenating the repository `README.md` with `docs/toctree.md`, so the home page of the
site is always a verbatim copy of the README. Add or reorder pages by editing
`docs/toctree.md` — every page must appear in one of its `{toctree}` blocks, or Sphinx will
warn that it is not included in any toctree.

### Page table of contents

Furo shows the sections of the current page in the right-hand sidebar, so pages do not carry
a contents box of their own. The left sidebar shows the same sections too: `conf.py` rebuilds
Furo's navigation tree with the section headings included.

### The Table of Contents page

`src/contents.md` shows the whole site as one fully expanded tree, every page with its
sections and subsections. It is a single directive:

````markdown
```{sitetoc}
```
````

Sphinx's own `{toctree}` cannot do this: it lists only documents no other toctree has claimed,
and it obeys the `:maxdepth:` of the navigation. `{sitetoc}` (in `_ext/juliadocstrings.py`)
instead asks the environment for the global toctree with no depth limit, at
`doctree-resolved` time. It needs no maintenance - new pages appear as soon as they are added
to `docs/toctree.md`.

### Stream outputs from executed pages

`conf.py` replaces myst-nb's `coalesce_streams`. A kernel may split one `println` across two
stream messages (the text and its newline are separate writes); myst-nb merges the pieces as
if each were whole lines, which turns the split into a blank line in the output. The
replacement concatenates the pieces, which is what a byte stream chopped at arbitrary points
requires. The symptom is a race, so it appears in some builds and not others.

## Contributing to Documentation

### Where to Add Content

| Type of Content | Location | Format |
|----------------|----------|--------|
| Installation guide | `src/installation.md` | Markdown (MyST) |
| Usage tutorials | a new `src/*.md`, added to a toctree in `src/index.md` | Markdown (MyST) |
| Tutorial with live output | `src/*.md` with a `kernelspec` header | Runnable MyST (see below) |
| Example notebooks | `examples/**.ipynb` (repo root), listed in `src/examples-index.md` | Jupyter |
| Element parameter reference | Julia docstrings for the parameter group, surfaced by `{docstring}` in `src/element.md` | Julia docstrings |
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

### Runnable pages (executed code blocks)

A page can have its Julia code **executed at build time**, with the real output
spliced in underneath each block — the same thing Documenter.jl's ```` ```@example ````
blocks do on the [Beamlines.jl](https://bmad-sim.github.io/Beamlines.jl/stable/quickstart/)
site. `src/quickstart.md` is the worked example.

A runnable page is an ordinary `.md` file, so it goes in a `{toctree}` next to plain
MyST pages and the two kinds interleave freely. Two things make it runnable:

**1. Front matter** naming a Julia kernel, at the very top of the file:

```yaml
    ---
    jupytext:
      text_representation:
        extension: .md
        format_name: myst
        format_version: 0.13
    kernelspec:
      display_name: Julia
      language: julia
      name: julia
    ---
```

(`name: julia` is remapped by `conf.py` to whichever IJulia kernel is installed.)

**2. `{code-cell}` blocks** instead of plain ```` ```julia ```` fences:

```markdown
    ```{code-cell} julia
    qf = Quadrupole(Kn1=0.36, L=0.5)
    ```
```

All cells on a page share one Julia session, in order, exactly like a single named
Documenter `@example` block. Plain ```` ```julia ```` fences on the same page are still
just displayed, never run — use those for snippets that shouldn't execute
(`Pkg.add`, `include` of a big lattice, …).

**Documenter → MyST equivalents**

| Documenter | Runnable MyST page |
|---|---|
| ```` ```@example name ```` | ```` ```{code-cell} julia ```` (one session per page) |
| ```` ```@setup name ```` | a `{code-cell}` with `:tags: [remove-cell]` |
| `# hide` on a line | `:tags: [remove-input]` (hide code, keep output) or `[remove-output]` |
| ```` ```@repl ```` | one `{code-cell}` per expression |
| an example that should throw | `:tags: [raises-exception]` |

**Converting a whole Documenter page to MyST.** Pages written for Documenter.jl (for
example, drafts that came from a `make.jl`-based build) need these translations as well:

| Documenter | MyST |
|---|---|
| `# [Title](@id label)` | a line `(label)=` immediately *before* `# Title` |
| `[text](@ref label)` | `[text](#label)` — resolves project-wide, across pages |
| `[text](@ref)` to a page | `[text](page.md)` |
| ```` ```math ```` block | `$$ … $$` (the `dollarmath` extension is enabled) |
| ``` ``x`` ``` inline math | `$x$` |
| `!!! note` / `!!! warning` | `:::{note}` … `:::` / `:::{warning}` … `:::` |
| ```` ```@docs ```` block | see below — docstrings are **not** rendered by Sphinx |

Sphinx cannot read Julia docstrings on its own, so ```` ```@docs ```` becomes the
`{docstring}` directive provided by `docs/_ext/juliadocstrings.py`:

````markdown
```{docstring} twiss
```

```{docstring}
track
TrackingResult
```
````

This works for anything reachable from `using SciBmad`, which includes the re-exported
`Beamlines`, `BeamTracking`, `NonlinearNormalForm`, `GTPSA` and `FundamentalFrequencies`
names — so `Beamline`, `DefExpr`, `Bunch`, `naff`, and the parameter groups all render
here without being duplicated into this site's Documenter build.

**How it works.** One long-lived `julia --project=docs` process serves the whole build
(`docs/_ext/docstring_server.jl`). It hands back the *raw* docstring text, which the
extension translates from Documenter-flavoured Markdown to MyST (```` ```jldoctest ````
→ ```` ```julia ````, ```` ```math ```` → `$$`, `` ``x`` `` → `$x$`, `!!! note` →
`:::{note}`, `[text](@ref)` → plain text, and `#` headings → rubrics so they stay out of
the page's table of contents). Successful lookups are cached in the doctree directory;
`SCIBMAD_DOCS_NO_JULIA=1` builds from that cache without starting Julia.

Tags go on the first line of the cell:

```markdown
    ```{code-cell} julia
    :tags: [remove-cell]
    ENV["COLUMNS"] = 100
    ```
```

**How the build runs them**

- `docs/build.py` installs an IJulia kernel named `scibmad-docs` that starts with
  `--project=docs`, so `using SciBmad` resolves in every executed page. `IJulia` is a
  dependency in `docs/Project.toml`.
- `nb_execution_mode = "auto"` in `conf.py` executes only notebooks with missing
  outputs — i.e. the runnable `.md` pages. `nb_execution_excludepatterns = ["*.ipynb"]`
  guarantees the committed, pre-executed `examples/` notebooks are never re-run.
- `nb_execution_raise_on_error = True`: a cell that throws **fails the build**, so
  examples can't silently rot. Tag the cell `raises-exception` if the error is the point.

**Building runnable pages by hand.** `python docs/build.py` sets the kernel up for you.
If you run `sphinx-build` directly, install the kernel once:

```bash
julia --project=docs -e 'using IJulia; IJulia.installkernel("SciBmad Docs", "--project=docs", specname="scibmad-docs")'
```

### Embedding Jupyter notebooks

Jupyter notebooks are rendered with [MyST-NB](https://myst-nb.readthedocs.io/).
The `examples/` directory at the repo root is symlinked into `src/examples`, and
notebooks are added to a `{toctree}` like any other page (see `src/examples-index.md`):

```markdown
    ```{toctree}
    Nonlinear Twiss <examples/julia/nonlinear-twiss.ipynb>
    ```
```

**Embedding a notebook inside another page.** A `{toctree}` entry makes a notebook a page
of its own. To splice one into the middle of an existing page instead — as
`dynamic-aperture.md` does — use the `{notebook}` directive, also from
`docs/_ext/juliadocstrings.py`:

````markdown
```{notebook} examples/julia/dynamic-aperture.ipynb
:skip-heading:
```
````

It renders the notebook's cells and their **stored** outputs (figures included; these are
written to `src/_nbimages/`, which is gitignored) without ever starting a kernel.
`:skip-heading:` drops the notebook's own title, and `:heading-offset: N` demotes its
headings by `N` levels so they nest under the host page.

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
docstrings are **not** re-rendered in this site's *Documenter* build — that only
cross-references into the Beamlines.jl site (via `DocumenterInterLinks` in `api/make.jl`
and intersphinx in `conf.py`), so don't add `@autodocs Modules = [Beamlines]` back to
`api/src/index.md`. The narrative pages are a different matter: `{docstring}` reads
whatever `using SciBmad` can see, so `Beamline`, `DefExpr`, `Context` and the parameter
groups are shown in full where they are being explained.

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
