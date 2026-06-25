using Beamlines
using SciBmad
using Documenter
using DocumenterInterLinks

# Resolve `@extref` cross-references to Beamlines.jl types/functions against
# its own published documentation at https://bmad-sim.github.io/Beamlines.jl
links = InterLinks(
    "Beamlines" => "https://bmad-sim.github.io/Beamlines.jl/stable/",
)

DocMeta.setdocmeta!(SciBmad, :DocTestSetup, :(using SciBmad); recursive=true)

# Note: Beamlines is intentionally NOT in `modules`. Its docstrings are documented
# on the Beamlines.jl site; we only cross-reference into it via `links` below so we
# don't duplicate that content here.
makedocs(;
    modules=[SciBmad],
    authors="mattsignorelli <mgs255@cornell.edu> and contributors",
    sitename="SciBmad.jl API Reference",
    format=Documenter.HTML(;
        canonical="https://bmad-sim.github.io/Beamlines.jl",
        edit_link="main",
        assets=String[],
        prettyurls=false, # prettyurls breaks the redirect to the main documentation
    ),
    pages=[
        "← Documentation" => "main-docs.md",
        "API Reference" => "index.md",
    ],
    plugins=[links],
    warnonly=true,  # Don't fail on warnings
)
