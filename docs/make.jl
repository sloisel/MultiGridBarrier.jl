using Pkg
Pkg.activate(@__DIR__)
# As long as it is not registered, this is nice, in general it locally always
# renders docs of the current version checked out in this repo.
Pkg.develop(PackageSpec(; path=(@__DIR__) * "/../"))
using MultiGridBarrier
using Documenter
using PyPlot

DocMeta.setdocmeta!(MultiGridBarrier, :DocTestSetup, :(using MultiGridBarrier); recursive=true)

makedocs(;
    modules=[MultiGridBarrier, MultiGridBarrier.Mesh3d],
    authors="Sébastien Loisel",
    sitename="MultiGridBarrier.jl $(pkgversion(MultiGridBarrier))",
    warnonly = [:missing_docs, :cross_references, :docs_block],
    format=Documenter.HTML(;
        canonical="https://sloisel.github.io/MultiGridBarrier.jl",
        edit_link="main",
        assets=String[],
        size_threshold=300_000,  # index.html is ~210KB due to embedded examples
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sloisel/MultiGridBarrier.jl",
    devbranch="main",
)
