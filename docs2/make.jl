using Pkg
using Documenter, SciBmad

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)
cp(joinpath(@__DIR__, "..", "GOVERNANCE.md"), joinpath(@__DIR__, "src", "governance.md"); force=true)

makedocs(
  sitename="SciBmad",
  authors = "Matt Signorelli, David Sagan",
  format=Documenter.HTMLWriter.HTML(size_threshold = nothing),
  pages = 
  [
    "Home" => "index.md",
    "Installation" => "installation.md",
    "Quickstart" => "quickstart.md",
    "Defining a LineElement" => "element.md",
    "Defining a Beamline" => "beamline.md",
    "Deferred Expressions and Contexts" => "defexpr.md",
    "Track" => "track.md",
    "Twiss" => "twiss.md",
    "Tracking Methods" => "tracking-methods.md",
    "Collective Effects" => "collective.md",
    "Time-Dependent Parameters and Ramping" => "timedependent.md",
    "Batch Parameters" => "batch.md",
    "(GPU-)Batched Closed Orbit Finder" => "co.md",
    "Parametric Normal Form" => "parametric-nf.md",
    "Optimization with Autodiff" => "optimize.md",
    "Dynamic Aperture" => "dynamic-aperture.md",
    "Frequency Map Analysis" => "fma.md",
    "SciBmad Governance" => "governance.md"
  ]
)

deploydocs(; repo = "github.com/bmad-sim/SciBmad.jl.git")
