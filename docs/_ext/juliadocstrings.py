"""Sphinx directives for pulling Julia content into the MyST narrative pages.

``{docstring}``
    Renders a Julia docstring (see below).

``{notebook}``
    Splices an already-executed Jupyter notebook, outputs and figures included,
    into the middle of a page -- something neither MyST's ``{include}`` (text
    files only) nor a ``{toctree}`` entry (a separate page) can do.

The ``{docstring}`` directive
=============================

Sphinx has no idea how to read Julia docstrings, which is why the narrative pages
used to link out to the Documenter-built API reference instead of showing them.
This extension closes that gap: it keeps a single long-lived Julia process around
for the whole build, asks it for the *raw* text of a docstring, translates the
Documenter-flavoured Markdown into MyST, and parses the result into the page.

Usage (either form, and they can be combined)::

    ```{docstring} twiss
    ```

    ```{docstring}
    track
    TrackingResult
    ```

which is the MyST counterpart of Documenter's ```` ```@docs ```` blocks.

If Julia cannot be started, the extension falls back to a JSON cache written by a
previous successful build, and warns for anything the cache does not cover. Set
``SCIBMAD_DOCS_NO_JULIA=1`` to skip Julia entirely and build from that cache.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import posixpath
import re
import subprocess
from pathlib import Path

from docutils import nodes
from docutils.parsers.rst import directives
from docutils.statemachine import StringList
from sphinx.util import logging
from sphinx.util.docutils import SphinxDirective

logger = logging.getLogger(__name__)

SEP = "\x04SEP\x04"
END = "\x04END\x04"
ERR = "\x04ERR\x04"

_DOCS_DIR = Path(__file__).resolve().parent.parent
_SERVER_JL = Path(__file__).resolve().parent / "docstring_server.jl"


# --------------------------------------------------------------------------------------
# Talking to Julia
# --------------------------------------------------------------------------------------
class JuliaDocstrings:
    """A long-lived ``julia`` subprocess that answers docstring lookups."""

    def __init__(self, cache_path: Path) -> None:
        self.cache_path = cache_path
        self.cache = {}
        self.fresh = {}
        self.proc = None
        self.disabled = os.environ.get("SCIBMAD_DOCS_NO_JULIA") == "1"
        self._load_cache()

    def _load_cache(self) -> None:
        try:
            self.cache = json.loads(self.cache_path.read_text())
        except Exception:
            self.cache = {}

    def save_cache(self) -> None:
        if not self.fresh:
            return
        merged = dict(self.cache)
        merged.update(self.fresh)
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(json.dumps(merged, indent=1, sort_keys=True))
        except Exception as exc:  # pragma: no cover - best effort only
            logger.warning("could not write Julia docstring cache: %s", exc)

    def _start(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        if self.disabled:
            return False
        logger.info("[julia-docstrings] starting Julia to read docstrings...")
        try:
            self.proc = subprocess.Popen(
                [
                    "julia",
                    "--startup-file=no",
                    f"--project={_DOCS_DIR}",
                    str(_SERVER_JL),
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=None,
                text=True,
                bufsize=1,
                cwd=str(_DOCS_DIR.parent),
            )
        except OSError as exc:
            logger.warning(
                "[julia-docstrings] could not start Julia (%s); "
                "falling back to the cached docstrings",
                exc,
            )
            self.disabled = True
            self.proc = None
            return False
        return True

    def fetch(self, name: str):
        """Return ``(kind, [raw_text, ...])`` for ``name``, or ``None``."""
        if name in self.fresh:
            return self.fresh[name]

        if self._start():
            try:
                self.proc.stdin.write(name + "\n")
                self.proc.stdin.flush()
                lines = []
                while True:
                    line = self.proc.stdout.readline()
                    if line == "":  # Julia died
                        raise RuntimeError("the Julia docstring server exited")
                    if line.rstrip("\n") == END:
                        break
                    lines.append(line.rstrip("\n"))
            except Exception as exc:
                logger.warning("[julia-docstrings] lookup of %r failed: %s", name, exc)
                self.proc = None
                self.disabled = True
                lines = None

            if lines is not None:
                if lines and lines[0] == ERR:
                    logger.warning(
                        "[julia-docstrings] no docstring for %r: %s",
                        name,
                        " ".join(lines[1:]).strip(),
                    )
                    return None
                kind = lines[0] if lines else "Function"
                body = "\n".join(lines[1:])
                result = (kind, body.split("\n" + SEP + "\n"))
                self.fresh[name] = result
                return result

        cached = self.cache.get(name)
        if cached is not None:
            return cached[0], cached[1]
        return None

    def shutdown(self) -> None:
        self.save_cache()
        if self.proc is not None and self.proc.poll() is None:
            try:
                self.proc.stdin.close()
                self.proc.wait(timeout=20)
            except Exception:  # pragma: no cover
                self.proc.kill()
        self.proc = None


# --------------------------------------------------------------------------------------
# Documenter-flavoured Markdown -> MyST
# --------------------------------------------------------------------------------------
_FENCE = re.compile(r"^(\s*)(`{3,}|~{3,})\s*(.*?)\s*$")
_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
_ADMONITION = re.compile(r'^!!!\s+(\w+)\s*(?:"(.*)")?\s*$')
_INLINE_MATH = re.compile(r"(?<!`)``([^`\n]+?)``(?!`)")
_REF_LINK = re.compile(r"\[([^\]]*)\]\(@(?:ref|extref|id)[^)]*\)")

_ADMONITIONS = {
    "note",
    "info",
    "tip",
    "warning",
    "danger",
    "compat",
    "todo",
    "details",
}


def _split_signature(lines):
    """Peel the leading indented code block (the signature) off a docstring."""
    i = 0
    while i < len(lines) and not lines[i].strip():
        i += 1
    start = i
    while i < len(lines) and (not lines[i].strip() or lines[i].startswith("    ")):
        i += 1
    # Trailing blank lines belong to the body, not the signature.
    end = i
    while end > start and not lines[end - 1].strip():
        end -= 1
    if end == start:
        return [], lines
    signature = [line[4:] if line.startswith("    ") else line for line in lines[start:end]]
    return signature, lines[end:]


def _convert_inline(line: str) -> str:
    line = _REF_LINK.sub(r"\1", line)
    line = _INLINE_MATH.sub(r"$\1$", line)
    return line


def _dedent(lines, amount=4):
    return [line[amount:] if line.startswith(" " * amount) else line.lstrip() if line.strip() else "" for line in lines]


def convert(text: str) -> str:
    """Translate one raw Julia/Documenter docstring body into MyST Markdown."""
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # ---- fenced code blocks ------------------------------------------------
        m = _FENCE.match(line)
        if m:
            indent, ticks, info = m.groups()
            block = []
            i += 1
            while i < len(lines):
                m2 = _FENCE.match(lines[i])
                if m2 and m2.group(2)[0] == ticks[0] and len(m2.group(2)) >= len(ticks) and not m2.group(3):
                    i += 1
                    break
                block.append(lines[i])
                i += 1
            if info.split(";")[0].strip() == "math":
                # Documenter's ```math blocks are MyST's $$ ... $$
                out.append(f"{indent}$$")
                out.extend(block)
                out.append(f"{indent}$$")
            else:
                lang = info.split(";")[0].strip()
                if lang.startswith("jldoctest") or lang == "doctest":
                    lang = "julia"
                out.append(f"{indent}{ticks}{lang}")
                out.extend(block)
                out.append(f"{indent}{ticks}")
            continue

        # ---- !!! admonitions ---------------------------------------------------
        m = _ADMONITION.match(line)
        if m:
            kind, title = m.group(1).lower(), (m.group(2) or "").strip()
            if kind not in _ADMONITIONS:
                kind = "note"
            body = []
            i += 1
            while i < len(lines) and (not lines[i].strip() or lines[i].startswith("    ")):
                body.append(lines[i])
                i += 1
            while body and not body[-1].strip():
                body.pop()
            out.append(f":::{{{kind}}}" + (f" {title}" if title else ""))
            out.extend(convert("\n".join(_dedent(body))).split("\n"))
            out.append(":::")
            continue

        # ---- headings ----------------------------------------------------------
        m = _HEADING.match(line)
        if m:
            # Rendered as a rubric so docstring headings never enter the page's
            # table of contents or fight with the surrounding section structure.
            out.append("```{rubric} " + _convert_inline(m.group(2)))
            out.append("```")
            i += 1
            continue

        out.append(_convert_inline(line))
        i += 1

    return "\n".join(out)


# --------------------------------------------------------------------------------------
# The directive
# --------------------------------------------------------------------------------------
class DocstringDirective(SphinxDirective):
    """Render one or more Julia docstrings."""

    has_content = True
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    option_spec = {
        "nosignature": directives.flag,
    }

    def run(self):
        names = []
        if self.arguments:
            names.extend(self.arguments[0].split())
        names.extend(name for name in (line.strip() for line in self.content) if name)

        if not names:
            logger.warning(
                "empty {docstring} directive", location=(self.env.docname, self.lineno)
            )
            return []

        server = self.env.app._julia_docstrings
        result = []
        for name in names:
            entry = server.fetch(name)
            container = nodes.container(classes=["julia-docstring"])
            self.set_source_info(container)

            if entry is None:
                logger.warning(
                    "no Julia docstring found for %r", name,
                    location=(self.env.docname, self.lineno),
                )
                container += nodes.paragraph(
                    "", "", nodes.literal(text=name),
                    nodes.Text("  (docstring unavailable in this build)"),
                )
                result.append(container)
                continue

            kind, texts = entry
            header = nodes.paragraph(classes=["julia-docstring-header"])
            header += nodes.literal(text=name, classes=["julia-docstring-name"])
            header += nodes.Text(" — ")
            header += nodes.emphasis(text=kind)
            header += nodes.Text(".")
            container += header

            for text in texts:
                signature, body = _split_signature(text.split("\n"))
                if signature and "nosignature" not in self.options:
                    literal = nodes.literal_block(
                        "\n".join(signature), "\n".join(signature)
                    )
                    literal["language"] = "julia"
                    literal["classes"] = ["julia-docstring-signature"]
                    container += literal
                elif signature:
                    body = signature + [""] + body

                converted = convert("\n".join(body))
                content = StringList(
                    converted.split("\n"), source=f"<docstring:{name}>"
                )
                self.state.nested_parse(content, 0, container)

            result.append(container)

        return result


# --------------------------------------------------------------------------------------
# Embedding an executed notebook inside a page
# --------------------------------------------------------------------------------------
_ANSI = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")
_MD_HEADING = re.compile(r"^(#{1,5})(\s)", re.M)


def _as_text(value):
    return "".join(value) if isinstance(value, list) else (value or "")


class NotebookDirective(SphinxDirective):
    """Render a committed, already-executed notebook inline in the current page.

    The notebook is never executed: its stored outputs (including figures) are
    what get rendered, exactly as for the notebooks that are pages of their own.
    """

    has_content = False
    required_arguments = 1
    final_argument_whitespace = True
    option_spec = {
        "heading-offset": directives.nonnegative_int,
        "skip-heading": directives.flag,
    }

    def run(self):
        relpath, path = self.env.relfn2path(self.arguments[0].strip())
        self.env.note_dependency(path)
        try:
            with open(path, encoding="utf-8") as f:
                notebook = json.load(f)
        except Exception as exc:
            logger.warning(
                "{notebook}: could not read %s (%s)", self.arguments[0], exc,
                location=(self.env.docname, self.lineno),
            )
            return []

        offset = self.options.get("heading-offset", 0)
        skip_heading = "skip-heading" in self.options

        container = nodes.container(classes=["embedded-notebook"])
        self.set_source_info(container)

        for index, cell in enumerate(notebook.get("cells", [])):
            source = _as_text(cell.get("source"))
            if cell.get("cell_type") == "markdown":
                text = source
                if skip_heading:
                    lines = text.split("\n")
                    if lines and lines[0].lstrip().startswith("#"):
                        text = "\n".join(lines[1:]).lstrip("\n")
                    skip_heading = False
                if offset:
                    text = _MD_HEADING.sub(lambda m: "#" * offset + m.group(1) + m.group(2), text)
                if text.strip():
                    self.state.nested_parse(
                        StringList(text.split("\n"), source=path), 0, container
                    )
            elif cell.get("cell_type") == "code":
                if source.strip():
                    literal = nodes.literal_block(source, source)
                    literal["language"] = "julia"
                    container += literal
                for output in cell.get("outputs", []):
                    node = self._render_output(output, relpath, index)
                    if node is not None:
                        container += node

        return [container]

    def _render_output(self, output, relpath, cell_index):
        kind = output.get("output_type")
        if kind == "stream":
            text = _as_text(output.get("text")).rstrip("\n")
            return self._output_block(text) if text else None
        if kind == "error":
            text = _ANSI.sub("", "\n".join(output.get("traceback", []))).rstrip("\n")
            return self._output_block(text) if text else None
        if kind in ("execute_result", "display_data"):
            data = output.get("data") or {}
            for mime in ("image/png", "image/jpeg"):
                if mime in data:
                    uri = self._write_image(data[mime], mime, relpath, cell_index)
                    if uri:
                        return nodes.image(uri=uri, classes=["notebook-output-image"])
            if "text/plain" in data:
                text = _as_text(data["text/plain"]).rstrip("\n")
                return self._output_block(text) if text else None
        return None

    def _output_block(self, text):
        block = nodes.literal_block(text, text)
        block["language"] = "none"
        block["classes"] = ["notebook-output"]
        return block

    def _write_image(self, payload, mime, relpath, cell_index):
        """Write an embedded figure out as a file Sphinx's image handling can pick up."""
        raw = base64.b64decode("".join(payload) if isinstance(payload, list) else payload)
        suffix = ".png" if mime == "image/png" else ".jpg"
        stem = re.sub(r"[^A-Za-z0-9]+", "-", posixpath.splitext(relpath)[0]).strip("-")
        name = f"{stem}-{cell_index}-{hashlib.sha1(raw).hexdigest()[:8]}{suffix}"
        target_dir = Path(self.env.srcdir) / "_nbimages"
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
            (target_dir / name).write_bytes(raw)
        except OSError as exc:  # pragma: no cover - best effort only
            logger.warning("{notebook}: could not write figure %s (%s)", name, exc)
            return None
        # Image URIs are resolved relative to the directory of the current document.
        depth = self.env.docname.count("/")
        return "../" * depth + f"_nbimages/{name}"



# --------------------------------------------------------------------------------------
# `{sitetoc}` - the whole site as one fully expanded tree
# --------------------------------------------------------------------------------------
# Sphinx's own `{toctree}` can only list documents that are not already claimed by
# another toctree, and it renders page titles only. This directive instead asks the
# environment for the *global* toctree with every level expanded, so one page can show
# every section and subsection of the site at once.
class SiteTocNode(nodes.General, nodes.Element):
    pass


class SiteTocDirective(SphinxDirective):
    has_content = False
    option_spec = {"maxdepth": directives.nonnegative_int}

    def run(self):
        node = SiteTocNode()
        # -1 means "no limit". Sphinx treats 0 as "fall back to the `:maxdepth:`
        # of each navigation toctree", which would cut the tree off at the same
        # depth as the sidebar - the opposite of what this page is for.
        node["maxdepth"] = self.options.get("maxdepth", -1)
        node["docname"] = self.env.docname
        return [node]


def _resolve_sitetoc(app, doctree, docname):
    from sphinx.environment.adapters.toctree import global_toctree_for_doc

    for node in list(doctree.findall(SiteTocNode)):
        toc = global_toctree_for_doc(
            app.env,
            node["docname"],
            app.builder,
            collapse=False,
            includehidden=True,
            maxdepth=node["maxdepth"],
            titles_only=False,
        )
        if toc is None:
            node.parent.remove(node)
            continue
        # A `compact_paragraph` is rendered without an element of its own, so a
        # class set on it would never reach the HTML. Wrap it instead - and wrap
        # the node itself rather than re-parenting its children, since a toctree
        # caption is a `title` whose rendering depends on its parent carrying
        # the `toctree` attribute.
        wrapper = nodes.container(classes=["site-toc"])
        wrapper += toc
        node.replace_self(wrapper)


def _skip_sitetoc(self, node):
    raise nodes.SkipNode


# --------------------------------------------------------------------------------------
# Wiring
# --------------------------------------------------------------------------------------
def _init(app):
    app._julia_docstrings = JuliaDocstrings(
        Path(app.doctreedir) / "julia-docstrings.json"
    )


def _finish(app, exception):
    server = getattr(app, "_julia_docstrings", None)
    if server is not None:
        server.shutdown()


def setup(app):
    app.add_directive("docstring", DocstringDirective)
    app.add_directive("notebook", NotebookDirective)
    app.add_directive("sitetoc", SiteTocDirective)
    app.add_node(SiteTocNode, html=(_skip_sitetoc, None))
    app.connect("doctree-resolved", _resolve_sitetoc)
    app.connect("builder-inited", _init)
    app.connect("build-finished", _finish)
    return {
        "version": "1.0",
        "parallel_read_safe": False,
        "parallel_write_safe": True,
    }
