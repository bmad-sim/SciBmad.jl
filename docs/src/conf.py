# Configuration file for the Sphinx documentation builder.

import os
import sys

# `docs/_ext` holds the Sphinx extensions written for this site.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(_HERE), '_ext'))

# -- Project information -----------------------------------------------------
project = 'SciBmad'
copyright = '2025, SciBmad Contributors'
author = 'SciBmad Contributors'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_nb',          # superset of myst_parser that also renders Jupyter notebooks
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',
    'sphinx_copybutton',
    'juliadocstrings',  # docs/_ext/juliadocstrings.py -- the `{docstring}` directive
]

# -- Copy button on code blocks ----------------------------------------------
# Applied to code *inputs* only. The parent of a rendered output block carries
# class "output" (e.g. <div class="output text_plain ...">), so excluding it
# leaves the button on hand-written fences and on executed `{code-cell}` inputs
# while keeping it off the results, which are not meant to be pasted anywhere.
copybutton_selector = "div:not(.output) > div.highlight pre"

# Nothing in the docs is written as a REPL transcript today, but strip the
# prompts if one ever appears, so the copied text stays runnable.
copybutton_prompt_text = r"julia> |shell> |\(.*\) pkg> |help\?> "
copybutton_prompt_is_regexp = True

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

    def resolve_any_xref(self, env, fromdocname, builder, target, node, contnode):
        # Needed so MyST's "any"-style links (e.g. `[text](#label)`) don't warn about
        # this domain; the domain holds no objects of its own.
        return []

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

# -- Home page: a verbatim copy of the repository README ----------------------
# The landing page is the README, so the two can never drift apart. Sphinx also
# needs the site's `{toctree}` blocks to live in the root document, so they are
# kept in `docs/toctree.md` and appended here; they are `:hidden:`, which puts
# them in the sidebar without adding anything to the rendered page.
_REPO_ROOT = os.path.dirname(os.path.dirname(_HERE))

def _generate_index_page():
    readme = os.path.join(_REPO_ROOT, 'README.md')
    toctree = os.path.join(os.path.dirname(_HERE), 'toctree.md')
    target = os.path.join(_HERE, 'index.md')

    with open(readme, encoding='utf-8') as f:
        content = f.read()
    with open(toctree, encoding='utf-8') as f:
        content = content.rstrip() + '\n\n' + f.read()

    # Only rewrite when something actually changed, so Sphinx does not consider
    # the landing page outdated on every build.
    try:
        with open(target, encoding='utf-8') as f:
            if f.read() == content:
                return
    except FileNotFoundError:
        pass
    with open(target, 'w', encoding='utf-8') as f:
        f.write(content)

_generate_index_page()

# -- Left sidebar: expand the current page's own sections ---------------------
# Furo builds its navigation tree with `titles_only=True`, so the sidebar lists
# page titles and nothing else. Rebuild it with the section headings included, so
# a reader can jump straight to a section of the page they are on. `collapse` has
# to stay False: Sphinx prunes sub-entries before it knows which page is current,
# so collapsing here would drop the very sections we want. Furo's own CSS folds
# the other pages' sections away and leaves the current page's expanded.
def _expand_navigation_tree(app, pagename, templatename, context, doctree):
    if 'toctree' not in context:
        return
    try:
        from furo.navigation import get_navigation_tree
    except ImportError:
        return
    context['furo_navigation_tree'] = get_navigation_tree(
        context['toctree'](
            collapse=False,
            titles_only=False,
            maxdepth=2,
            includehidden=True,
        )
    )

def setup(app):
    app.add_domain(_JuliaDomain)
    app.connect('doctree-resolved', _fix_intersphinx_refs)
    app.connect('build-finished', _fail_on_execution_error)
    # Furo is loaded as a theme, i.e. after conf.py's setup(), so its own
    # html-page-context handler is registered last. A higher priority is what
    # makes this one run after it and win.
    app.connect('html-page-context', _expand_navigation_tree, priority=900)

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

html_title = 'SciBmad Documentation'
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
