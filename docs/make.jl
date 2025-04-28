using SciBmad
using Documenter

DocMeta.setdocmeta!(SciBmad, :DocTestSetup, :(using SciBmad); recursive=true)

makedocs(;
    modules=[SciBmad],
    authors="mattsignorelli <mgs255@cornell.edu> and contributors",
    sitename="SciBmad.jl",
    format=Documenter.HTML(;
        canonical="https://bmad-sim.github.io/SciBmad.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/bmad-sim/SciBmad.jl",
    devbranch="main",
)
