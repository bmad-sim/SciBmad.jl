# Configuration file for the Sphinx documentation builder.

import os
import sys

# -- Project information -----------------------------------------------------
project = 'SciBmad.jl'
copyright = '2025, SciBmad.jl Contributors'
author = 'SciBmad.jl Contributors'

# -- General configuration ---------------------------------------------------
extensions = [
    'myst_parser',
    'sphinx.ext.githubpages',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinxcontrib.bibtex',
]

numfig = True
bibtex_bibfiles = ['bibliography.bib']
exclude_patterns = ['parameters']
suppress_warnings = ["myst.header"]   # So a file does not cause a "Document headings start at H2, not H1" warning

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

def setup(app):
    app.add_domain(_JuliaDomain)
    app.connect('doctree-resolved', _fix_intersphinx_refs)

# MyST Parser configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
    "deflist",
    "colon_fence",
    "linkify",
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
html_theme = 'furo'

html_theme_options = {
    'source_repository': 'https://github.com/bmad-sim/SciBmad.jl',
    'source_branch': 'main',
    'source_directory': 'docs/src/',
    'navigation_with_keys': True,
    'sidebar_hide_name': False,
}

html_title = 'SciBmad.jl Documentation'
html_static_path = ['_static']
html_css_files = ['custom.css']

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
