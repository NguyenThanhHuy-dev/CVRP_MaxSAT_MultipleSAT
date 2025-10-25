# CVRP_SAT_LAB

This directory provides a clean project structure for the CVRP → MaxSAT pipeline.

Structure created:

- data/ - original instances (place .vrp, .csv, or .txt here). Subfolders A/ and B/ created.
- src/ - main source code (placeholders were created).
- solvers/ - place built solvers here (open-wbo, clasp, maxhs). A README in `solvers/open-wbo` explains linking.
- results/ - experiment outputs (logs, plots, csv).
- scripts/ - useful CLI scripts and conversion utilities.

Notes:
- I kept the repository root unchanged. There is already an `open-wbo/` directory at repository root. You can either move or create a symlink from `CVRP_SAT_LAB/solvers/open-wbo` to the existing `open-wbo/` at the repo root, for example:

  ln -s "$(pwd)/open-wbo" "CVRP_SAT_LAB/solvers/open-wbo"

- Real data files and solver builds were not moved automatically. This keeps changes small and reversible. If you want, I can move/copy them for you.

Next steps:
- Move your instance files into `CVRP_SAT_LAB/data/A` or `B`.
- Place built solver folders (open-wbo) into `CVRP_SAT_LAB/solvers` or create symlinks.
- Implement the small skeleton modules in `src/` (I created minimal placeholders).