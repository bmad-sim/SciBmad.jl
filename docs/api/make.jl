using Beamlines
using SciBmad
using Documenter

DocMeta.setdocmeta!(Beamlines, :DocTestSetup, :(using Beamlines); recursive=true)
DocMeta.setdocmeta!(SciBmad, :DocTestSetup, :(using SciBmad); recursive=true)

makedocs(;
    modules=[Beamlines, SciBmad],
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
    warnonly=true,  # Don't fail on warnings
)
