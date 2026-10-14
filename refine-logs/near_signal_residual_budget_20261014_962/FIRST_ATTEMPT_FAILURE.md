# Import failure before any scoring

The first CPU invocation exited 1 with `ModuleNotFoundError: No module named trifusion`. The tool added the repository root but omitted its modeling package path. Add that path explicitly, following the existing package location. No candidate metric, NN encoding or optimizer update was produced. Keep the original fixed grid and selector; retain the failed server log at logs/near_signal_residual_budget961. The next invocation uses new output paths.
