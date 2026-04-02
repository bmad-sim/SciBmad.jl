using Beamlines
using Documenter

DocMeta.setdocmeta!(Beamlines, :DocTestSetup, :(using Beamlines); recursive=true)

makedocs(;
    modules=[Beamlines],
    authors="mattsignorelli <mgs255@cornell.edu> and contributors",
    sitename="Beamlines.jl API Reference",
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
    warnonly=true,  # Don't fail on warnings
)
