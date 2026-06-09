# Contributing to MultiGridBarrier.jl

Contributions are welcome — bug reports, feature requests, documentation improvements, and
pull requests.

## Reporting bugs, requesting features, and questions

Please open an issue: <https://github.com/sloisel/MultiGridBarrier.jl/issues>. Usage
questions are welcome there too.

For a bug report, include:

- a minimal example that reproduces the problem,
- the error message or the incorrect output, and
- your Julia version (`versioninfo()`) and the package version (`] status MultiGridBarrier`).

## Contributing code

1. Fork the repository and create a branch for your change.
2. Make the change, adding or updating tests where appropriate.
3. Run the test suite:

   ```julia
   julia --project -e 'using Pkg; Pkg.test()'
   ```

4. Open a pull request describing what the change does and why.

Small, focused pull requests are easier to review and merge.

## License

By contributing, you agree that your contributions will be licensed under the project's
[MIT license](LICENSE).
