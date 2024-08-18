using MultiGridBarrier
using Test
using LinearAlgebra

@testset "MultiGridBarrier.jl" begin
    z = reshape(Float64[-1,-1,-1,1,0,0,2,2],(:,2))
    @test norm(fem1d_solve(Float64,L=1,p=1.0)-z)<1e-6
    z = reshape([2.0,1.0,2.0,1.0,2.0,1.0,0.9629629615502254,2.0,1.0,2.0,1.0,2.0,1.0,0.9629629615502254,2.8284271303696036,0.0,2.0,0.0,2.0,0.0,0.0,2.0,0.0,2.8284271303696036,0.0,2.0,0.0,0.0],(:,2))
    @test norm(fem2d_solve(Float64,L=1,p=1.0)-z)<1e-6
    z = reshape([-1.0,-0.9937184291481691,-1.0606601198186663,-0.28661158923558694,1.0,5.3340857286533046e-8,6.270131188059596e-8,1.0297480733710706e-7,2.999999898325467,5.999999681130596],(:,2))
    @test norm(spectral1d_solve(Float64,n=5,p=1.0)-z)<1e-6
    z = reshape([2.0,1.5,1.0,1.5,2.0,1.5,1.3293564567737008,1.000000018263945,1.3293564567737008,1.5,1.0,1.000000018263945,0.9999999999999992,1.000000018263945,1.0,1.5,1.3293564567737008,1.000000018263945,1.3293564567737006,1.5,2.0,1.5,1.0,1.5,2.0,2.8284271380814046,1.4605935026827006,1.6292452074881896e-7,1.460593502682701,2.8284271380814046,1.4605935026827004,0.9999999768073243,5.33408572865331e-8,0.9999999768073249,1.4605935026827008,1.6292452092368193e-7,5.33408572865331e-8,5.3340857286533106e-8,5.33408572865331e-8,1.6292452052399464e-7,1.4605935026827006,0.9999999768073246,5.334085728653312e-8,0.9999999768073247,1.460593502682701,2.8284271380814046,1.4605935026827008,1.6292452069885802e-7,1.4605935026827015,2.8284271380814046],(:,2))
    @test norm(spectral2d_solve(Float64,n=5,p=1.0)-z)<1e-6
    @test (MultiGridBarrier.fem_precompile(); true)
    @test (MultiGridBarrier.spectral_precompile(); true)
end
