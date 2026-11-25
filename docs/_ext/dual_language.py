"""Julia and Python versions of the documentation from the same pages.

The site is built twice: with ``-D docs_language=julia`` (the default, at the root of the site)
and with ``-D docs_language=python`` (under ``python/``). A switch in the top bar of every page
goes to the same page in the other language (``_static/language-switch.js``), and remembers the
choice.

In the pages:

* Every Julia code cell is followed by its Python version, a ``{code-cell} python`` (the Julia
  build drops it, the Python build drops the Julia cell before it). The same for plain code
  blocks: a ```` ```julia ```` block directly followed by a ```` ```python ```` block.
* A Julia code cell with the tag ``julia-only`` is left out of the Python build.
* In the Python build, Julia's ``true``, ``false`` and ``nothing`` in inline code of the text
  are Python's ``True``, ``False`` and ``None``.
* ``docs_python_rewrites`` in ``conf.py``: (regular expression, replacement) pairs applied to
  the text of the Python build (e.g. links to files that have a Python version).
* Text for one language only: ``:::{only} julia`` ... ``:::`` (or ``python``), and inline the
  roles ``{jl}`code``` and ``{py}`code```, which give inline code in their language's build and
  nothing in the other's (``{jltext}`...``` and ``{pytext}`...``` the same for plain text).

In the Python build the pages run in a Python kernel (the front matter's kernelspec is
replaced), with SciBmad from the Python package scibmad (PySciBmad).
"""

import re

from docutils import nodes

LANGUAGES = ("julia", "python")

_FENCE = re.compile(r"^(?P<indent>[ \t]*)(?P<ticks>`{3,})(?P<info>.*)$")
_PY_KERNEL = """kernelspec:
  display_name: Python 3
  language: python
  name: python3"""
_KERNEL = re.compile(r"kernelspec:\n(?:[ \t]+.*\n?)+")


def _blocks(text):
    """The source as a list of ("text", lines) and ("code", lang, kind, lines) items, where kind
    is "cell" for a {code-cell} and "fence" for a plain fenced block."""
    lines = text.split("\n")
    out = []
    buf = []
    i = 0
    while i < len(lines):
        m = _FENCE.match(lines[i])
        if m:
            info = m.group("info").strip()
            ticks = m.group("ticks")
            indent = m.group("indent")
            j = i + 1
            while j < len(lines) and not (lines[j].strip() == ticks and
                                          lines[j][:len(indent)].strip() == ""):
                j += 1
            block = lines[i:j + 1]
            cm = re.match(r"\{code-cell\}\s*(\w+)", info)
            if cm:
                kind, lang = "cell", cm.group(1).lower()
            elif info.split() and info.split()[0].lower() in LANGUAGES:
                kind, lang = "fence", info.split()[0].lower()
            else:
                kind, lang = None, None
            if kind is None:
                buf.extend(block)
            else:
                if buf:
                    out.append(("text", buf))
                    buf = []
                out.append(("code", lang, kind, block))
            i = j + 1
            continue
        buf.append(lines[i])
        i += 1
    if buf:
        out.append(("text", buf))
    return out


_INLINE = re.compile(r"(?<!`)`([^`\n]+)`(?!`)")
_LITERALS = (("true", "True"), ("false", "False"), ("nothing", "None"))


def _python_inline(line):
    """Julia's literals true, false and nothing in inline code of the text are Python's."""
    def fix(m):
        code = m.group(1)
        for a, b in _LITERALS:
            code = re.sub(rf"(?<![\w.]){a}(?![\w(])", b, code)
        return "`" + code + "`"
    return _INLINE.sub(fix, line)


def _is_blank(item):
    return item[0] == "text" and all(not line.strip() for line in item[1])


def transform(text, language, docname="?", rewrites=()):
    """The page `text` for the build of `language`. `rewrites` are (pattern, replacement) pairs
    applied to the text (not the code) of the Python build, e.g. for links to files that have a
    Python version."""
    items = _blocks(text)
    keep = [True] * len(items)
    missing = []
    for k, item in enumerate(items):
        if item[0] != "code":
            continue
        _, lang, kind, block = item
        # its twin: the next code block of the same kind, with only blank lines in between
        nxt = k + 1
        while nxt < len(items) and _is_blank(items[nxt]):
            nxt += 1
        twin = nxt < len(items) and items[nxt][0] == "code" and items[nxt][2] == kind and \
            items[nxt][1] == "python" and lang == "julia"
        if lang == "julia" and language == "python":
            if twin:
                keep[k] = False
            elif kind == "cell":
                keep[k] = False
                if "julia-only" not in "\n".join(block[1:4]):
                    missing.append(block[1] if len(block) > 1 else "")
        elif lang == "python" and language == "julia":
            prev = k - 1
            while prev >= 0 and _is_blank(items[prev]):
                prev -= 1
            if kind == "cell" or (prev >= 0 and items[prev][0] == "code" and
                                  items[prev][1] == "julia" and items[prev][2] == kind):
                keep[k] = False
    if missing and language == "python":
        raise ValueError(f"{docname}: Julia code cells without a Python version (add a "
                         f"{{code-cell}} python after each, or tag them julia-only): "
                         + "; ".join(repr(m) for m in missing[:5]))
    lines = []
    for k, item in enumerate(items):
        if not keep[k]:
            continue
        if item[0] == "text":
            if language == "python":
                for line in item[1]:
                    line = _python_inline(line)
                    for pattern, repl in rewrites:
                        line = re.sub(pattern, repl, line)
                    lines.append(line)
            else:
                lines.extend(item[1])
        else:
            lines.extend(item[3])
    out = "\n".join(lines)
    if language == "python" and out.startswith("---"):
        end = out.find("\n---", 3)
        if end > 0:
            front = out[:end]
            front = _KERNEL.sub(_PY_KERNEL + "\n", front, count=1)
            out = front + out[end:]
    return out


def _role(language, code=True):
    def role(name, rawtext, text, lineno, inliner, options=None, content=None):
        app_lang = inliner.document.settings.env.config.docs_language
        if app_lang != language:
            return [], []
        return [nodes.literal(rawtext, text) if code else nodes.Text(text)], []
    return role


def _source_read(app, docname, source):
    language = app.config.docs_language
    source[0] = transform(source[0], language, docname, app.config.docs_python_rewrites)


def _config_inited(app, config):
    if config.docs_language not in LANGUAGES:
        raise ValueError(f"docs_language must be one of {LANGUAGES}")
    app.tags.add(config.docs_language)
    app.add_js_file(None, body=f"window.SCIBMAD_DOCS_LANGUAGE = {config.docs_language!r};",
                    priority=100)


def setup(app):
    app.add_config_value("docs_language", "julia", "env")
    app.add_config_value("docs_python_rewrites", [], "env")
    app.connect("config-inited", _config_inited)
    app.connect("source-read", _source_read)
    app.add_role("jl", _role("julia"))
    app.add_role("py", _role("python"))
    app.add_role("jltext", _role("julia", code=False))
    app.add_role("pytext", _role("python", code=False))
    return {"parallel_read_safe": True, "parallel_write_safe": True}
