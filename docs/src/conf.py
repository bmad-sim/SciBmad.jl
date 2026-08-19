# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Project information -----------------------------------------------------
project = 'SciBmad.jl'
copyright = '2025, SciBmad.jl Contributors'
author = 'SciBmad.jl Contributors'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_nb',          # superset of myst_parser that also renders Jupyter notebooks
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',
]

# -- Jupyter notebook handling (myst-nb) -------------------------------------
# Two kinds of notebook content live in this site:
#
#   1. `examples/**.ipynb` -- committed already-executed, rendered from their
#      stored outputs. They are never re-run (see nb_execution_excludepatterns).
#   2. "runnable" MyST pages -- ordinary `.md` files carrying a `kernelspec` in
#      their front matter and `{code-cell}` blocks. These are executed by a Julia
#      kernel at build time and their outputs are spliced into the page, the same
#      way Documenter.jl's ```@example blocks work. They sit in the toctree next
#      to plain MyST pages, so runnable and non-runnable pages interleave freely.
#
# "auto" executes only notebooks that are missing outputs, which is exactly the
# runnable pages (they store no outputs); the exclude pattern below guarantees the
# committed examples are never re-run even if one of their cells has no output.
nb_execution_mode = "auto"
# Matched with PurePosixPath.match against the file path, i.e. from the right:
# "*.ipynb" covers every committed, pre-executed notebook under examples/.
nb_execution_excludepatterns = ["*.ipynb"]
# NOTE: deliberately False. myst-nb's raise_on_error re-raises immediately and drops
# the cell traceback, so a CI failure says only "ExecutionError: <path>". With it off,
# myst-nb logs the full traceback as a warning, and `_fail_on_execution_error` below
# fails the build at the end -- so a broken example still breaks CI, like Documenter,
# but you can see *why*.
nb_execution_raise_on_error = False
nb_execution_show_tb = True
nb_execution_timeout = 600           # seconds per cell; Julia's first `using` is slow
nb_merge_streams = True              # one output block per cell, not one per print

# Runnable pages declare `name: julia` in their kernelspec; map that to whichever
# IJulia kernel is actually installed. `docs/build.py` installs `scibmad-docs`,
# whose kernel runs against `docs/Project.toml`; fall back to any Julia kernel so a
# bare `sphinx-build` still works on a machine with a generic IJulia install.
def _julia_kernel_name(preferred="scibmad-docs"):
    try:
        from jupyter_client.kernelspec import KernelSpecManager
        ksm = KernelSpecManager()
        names = list(ksm.find_kernel_specs())
        if preferred in names:
            return preferred
        for name in names:
            try:
                if ksm.get_kernel_spec(name).language.lower() == "julia":
                    return name
            except Exception:
                continue
    except Exception:
        pass
    return preferred  # nothing installed; let myst-nb report the missing kernel

nb_kernel_rgx_aliases = {"julia.*": _julia_kernel_name()}

numfig = True
bibtex_bibfiles = ['bibliography.bib']
suppress_warnings = [
    "myst.header",               # files whose first heading is H1, not H2
    "mystnb.unknown_mime_type",  # notebook outputs that also carry text/csv or text/tsv (text/plain is rendered)
]

# -- Intersphinx configuration -----------------------------------------------
_docs_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_api_base_url = 'https://bmad-sim.github.io/SciBmad.jl/api/'
intersphinx_mapping = {
    'julia': (_api_base_url,
              (os.path.join(_docs_dir, 'api', 'build', 'objects.inv'),))
}

# Subpath where API docs live in the combined site (gh-pages/api/)
_api_subpath = 'api/'

def _fix_intersphinx_refs(app, doctree, docname):
    """Rewrite intersphinx absolute URLs to relative paths for local browsing."""
    from docutils import nodes
    from posixpath import relpath, dirname

    for node in doctree.traverse(nodes.reference):
        uri = node.get('refuri', '')
        if not uri.startswith(_api_base_url):
            continue
        rel_part = uri[len(_api_base_url):]
        target = _api_subpath + rel_part
        doc_dir = dirname(docname)
        if '#' in target:
            path_part, fragment = target.split('#', 1)
            node['refuri'] = relpath(path_part, doc_dir) + '#' + fragment
        else:
            node['refuri'] = relpath(target, doc_dir)

# -- Minimal Julia domain for intersphinx cross-references -------------------
# Documenter.jl writes jl:function, jl:type, etc. into objects.inv.
# Sphinx needs the domain registered to resolve those roles.
from sphinx.domains import Domain, ObjType
from sphinx.roles import XRefRole

class _JuliaDomain(Domain):
    name = 'jl'
    label = 'Julia'
    object_types = {
        'function': ObjType('function', 'function'),
        'method':   ObjType('method',   'method'),
        'type':     ObjType('type',     'type'),
        'macro':    ObjType('macro',    'macro'),
        'module':   ObjType('module',   'module'),
    }
    roles = {
        'function': XRefRole(),
        'method':   XRefRole(),
        'type':     XRefRole(),
        'macro':    XRefRole(),
        'module':   XRefRole(),
    }
    directives = {}
    initial_data = {'objects': {}}

    def resolve_xref(self, env, fromdocname, builder, typ, target, node, contnode):
        return None  # intersphinx handles external references

    def get_objects(self):
        return iter([])

# -- Fail the build if a runnable page's code errored -------------------------
def _fail_on_execution_error(app, exception):
    """Raise at the end of the build if any executed page had a failing cell."""
    if exception is not None:
        return  # the build already failed for another reason
    failures = []
    for docname, data in getattr(app.env, 'nb_metadata', {}).items():
        exec_data = data.get('exec_data')
        if exec_data and exec_data.get('succeeded') is False:
            failures.append((docname, exec_data))
    if not failures:
        return
    from sphinx.errors import SphinxError
    report = ['Code execution failed in %d runnable page(s):' % len(failures)]
    for docname, exec_data in failures:
        report.append('\n--- %s ---' % docname)
        report.append(exec_data.get('traceback') or str(exec_data.get('error')))
    raise SphinxError('\n'.join(report))

def setup(app):
    app.add_domain(_JuliaDomain)
    app.connect('doctree-resolved', _fix_intersphinx_refs)
    app.connect('build-finished', _fail_on_execution_error)

# MyST Parser configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "linkify",
]

templates_path = ['_templates']
exclude_patterns = [
    'parameters',              # included via other pages, not as standalone docs
    '**/.ipynb_checkpoints',   # Jupyter scratch copies under examples/
]

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'

html_theme_options = {
    'source_repository': 'https://github.com/bmad-sim/SciBmad.jl',
    'source_branch': 'main',
    'source_directory': 'docs/src/',
    'navigation_with_keys': True,
    'sidebar_hide_name': True,
    # Logo shown at the top left of the sidebar (paths relative to html_static_path).
    'light_logo': 'SciBmad-Logo.png',
    'dark_logo': 'SciBmad-Logo-dark.png',
}

html_title = 'SciBmad.jl Documentation'
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['topbar-github.js']

# Sidebar settings with custom external links
html_sidebars = {
    "**": [
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/scroll-start.html",
        "sidebar/navigation.html",
        "sidebar-external-links.html",
        "sidebar/scroll-end.html",
    ]
}

# -- Options for MyST --------------------------------------------------------
myst_heading_anchors = 3
