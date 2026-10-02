# Heritable short-term cellular memory emerges from stochastic biochemical reaction networks

## Project Abstract
Cells exhibit a mysterious form of selective heritable short-term memory, influencing outcomes as diverse as cell fate decisions in embryos and environmental responses in cancer cells and bacteria. Here, we present a simple theoretical framework explaining how this selective memory can arise from the reactions by which molecules are regulated in cells. Our key insight is that related cells retain more similar molecular concentrations relative to random cells when a greater variance of possible concentration states is created during a single cell generation than is created by cell division across a population. This persistence of molecular similarity down a lineage constitutes a form of heritable short-term memory. We identify the biochemical networks that produce, modify, and degrade molecules as an underexplored source of these additional molecular concentration states. Using experimentally informed simulations, we find that the strength and duration of molecular similarity down a lineage depend on tunable network properties, explaining why some cellular traits persist only briefly while others last generations. These contributions to molecular concentration variance from biochemical reaction networks act in concert with gene expression and other regulatory processes to shape the protein composition of cells. Our framework yields clear, testable predictions for determining how biochemical network architectures drive non-genetic cellular inheritance.

## Repository Structure
- `docs/` - Installation and usage instructions (`setup.md`)
- `src/las_model/simulations/figure_XX/` - One script per experiment, grouped by the figure it feeds; `run_simulations.py` runs them in bulk
- `src/las_model/figures/plot_figures.py` - Draws the main and supplementary figures (`--help` for options)
- `src/las_model/utils/` - Shared code: the cell simulation engine, analysis helpers, experiment saving and configuration

## Requirements
- Python 3.13 or newer
- The Python packages pinned in `pyproject.toml` and `uv.lock` (numpy, scipy, matplotlib, opencv-python, cmapy, python-dotenv), installed with `uv sync`
- A data directory, pointed to by the `ROOT_DIR` environment variable, holding the simulation outputs and the figure schematics
- Roboto font (optional, used for the figure panel labels; see `docs/setup.md`)

## Installation and Usage
See `docs/setup.md` for installing the environment, configuring the data directory, running the simulations and plotting the figures.

## Contact
Contact Allyson Sgro at sgroa@janelia.hhmi.org. 
