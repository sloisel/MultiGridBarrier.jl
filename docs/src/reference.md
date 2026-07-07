```@meta
CurrentModule = MultiGridBarrier
```

# API reference

The docstrings of the core module. The extension APIs are documented on their
own pages — [Plotting](plotting.md) (`plot`, `MGB3DFigure`, `HTML5anim`),
[JuMP](jump.md) (`MGBModel` and the modeling macros' helpers),
[Gmsh](gmsh.md) (`gmsh_import`), and [PyAMG](pyamg.md) (`amg_pyamg`) — and the
[index](@ref main-index) below links to everything.

## Module

```@autodocs
Modules = [MultiGridBarrier]
Order   = [:module]
Private = false
```

## Types

```@autodocs
Modules = [MultiGridBarrier]
Order   = [:type]
Private = false
Filter = t -> !(nameof(t) in (:MGBModel, :Coef, :EpiPower, :deriv, :integral, :set_start, :mgb_solution, :solver_log, :On, :Broken, :Continuous, :Uniform, :gmsh_import, :amg_pyamg, :MGB3DFigure, :HTML5anim))
```

## Functions

```@autodocs
Modules = [MultiGridBarrier]
Order   = [:function]
Private = false
Filter = t -> !(nameof(t) in (:MGBModel, :Coef, :EpiPower, :deriv, :integral, :set_start, :mgb_solution, :solver_log, :On, :Broken, :Continuous, :Uniform, :gmsh_import, :amg_pyamg, :MGB3DFigure, :HTML5anim))
```

## [Index](@id main-index)

```@index
```
