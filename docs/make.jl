using MultiGridBarrier
using Documenter

DocMeta.setdocmeta!(MultiGridBarrier, :DocTestSetup, :(using MultiGridBarrier); recursive=true)

makedocs(;
    modules=[MultiGridBarrier],
    authors="Sébastien Loisel",
    sitename="MultiGridBarrier.jl",
    format=Documenter.HTML(;
        canonical="https://sloisel.github.io/MultiGridBarrier.jl",
        edit_link="main",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/sloisel/MultiGridBarrier.jl",
    devbranch="main",
)
