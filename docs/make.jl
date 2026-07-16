using Pkg
Pkg.activate(@__DIR__)
# As long as it is not registered, this is nice, in general it locally always
# renders docs of the current version checked out in this repo.
Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../"))
using MultiGridBarrier
using Documenter
using PyPlot
using JuMP   # loads the MultiGridBarrierJuMPExt extension (used by the JuMP page)
using Gmsh   # loads the MultiGridBarrierGmshExt extension (used by the Gmsh page)

DocMeta.setdocmeta!(MultiGridBarrier, :DocTestSetup, :(using MultiGridBarrier); recursive=true)

makedocs(;
    modules=[MultiGridBarrier],
    authors="Sébastien Loisel",
    sitename="MultiGridBarrier.jl $(pkgversion(MultiGridBarrier))",
    warnonly = [:missing_docs, :cross_references, :docs_block],
    format=Documenter.HTML(;
        canonical="https://sloisel.github.io/MultiGridBarrier.jl",
        edit_link="main",
        assets=String[],
        size_threshold=300_000,  # api_guide.html is ~210KB due to embedded examples
    ),
    pages=[
        "Home" => "index.md",
        "Gmsh" => "gmsh.md",
        "JuMP" => "jump.md",
        "Plotting" => "plotting.md",
        "CUDA" => "cuda.md",
        "PyAMG" => "pyamg.md",
        "API Guide" => "api_guide.md",
        "Zoo" => "zoo.md",
        "API Reference" => "reference.md",
    ],
)

deploydocs(;
    repo="github.com/sloisel/MultiGridBarrier.jl",
    devbranch="main",
)
