# Docstring server for the Sphinx `{docstring}` directive (see docs/_ext/juliadocstrings.py).
#
# Reads one identifier per line from stdin and writes that object's *raw* docstring
# text (exactly as written in the source, before Julia's Markdown parser reflows it)
# back to stdout. One long-lived process serves the whole documentation build, so
# SciBmad is loaded only once.
#
# Protocol, per request:
#   <kind>                      e.g. "Function", "Type", "Macro", "Constant"
#   <raw docstring of method 1>
#   \x04SEP\x04                 (only between methods)
#   <raw docstring of method 2>
#   \x04END\x04
# On failure the first line is "\x04ERR\x04" and the body is the error message.

using SciBmad

const SEP = "\x04SEP\x04"
const END = "\x04END\x04"
const ERR = "\x04ERR\x04"

kindof(x::Module)   = "Module"
kindof(x::Type)     = "Type"
kindof(x::Function) = "Function"
kindof(x)           = "Constant"

"""Return `(kind, [raw docstring text, ...])` for the identifier `name`."""
function lookup(name)
  ex = Meta.parse(name)
  if ex isa Expr && ex.head === :macrocall
    binding = Docs.Binding(Main, Symbol(name))
    md = Docs.doc(binding)
    kind = "Macro"
  else
    obj = Core.eval(Main, ex)
    md = Docs.doc(obj)
    kind = kindof(obj)
  end

  # `Docs.doc` stores the unparsed docstrings in `meta[:results]`; prefer those so
  # that indentation, admonitions and math survive verbatim. Fall back to the
  # rendered Markdown if a docstring was attached in some other way.
  results = get(md.meta, :results, nothing)
  texts = if results === nothing || isempty(results)
    [string(md)]
  else
    [join(r.text) for r in results]
  end
  return kind, texts
end

while !eof(stdin)
  name = readline(stdin)
  isempty(strip(name)) && continue
  try
    kind, texts = lookup(strip(name))
    println(kind)
    print(join(texts, string("\n", SEP, "\n")))
    println()
  catch err
    println(ERR)
    println(sprint(showerror, err))
  end
  println(END)
  flush(stdout)
end
