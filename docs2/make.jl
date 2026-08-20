using Pkg
using Documenter, SciBmad

cp(joinpath(@__DIR__, "..", "README.md"), joinpath(@__DIR__, "src", "index.md"); force=true)

makedocs(
  sitename="SciBmad",
  authors = "Matt Signorelli",
  format=Documenter.HTMLWriter.HTML(size_threshold = nothing),
  pages = 
  [
    "Home" => "index.md",
    "Installation" => "installation.md",
    "Quickstart" => "quickstart.md",
    "LineElements and Beamlines" => "",
    "Manual" => ["man/a_toc.md",
                 "man/b_descriptor.md", 
                 "man/c_tps.md",
                 "man/d_varsparams.md",
                 "man/e_monoindex.md",
                 "man/f_mono.md",
                 "man/g_gjh.md",
                 "man/h_slice.md",
                 "man/i_methods.md",
                 "man/j_fastgtpsa.md",
                 "man/k_io.md",
                 "man/l_global.md",
                 "man/m_all.md"],
    "For Developers" => "devel.md"
  ]
)

deploydocs(; repo = "github.com/bmad-sim/SciBmad.jl.git")
