# JuMP front-end tests: cross-validate the jump/ modeling layer against the
# classical API (all six Zoo problems + region-restricted constraints).
# JuMP is a test-only dependency ([extras]/[targets] in Project.toml); the
# package itself does not depend on it.
include(joinpath(@__DIR__, "..", "jump", "test_zoo.jl"))
