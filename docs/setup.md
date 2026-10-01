## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sgrolab/cellsimilaritymodel.git
cd cellsimilaritymodel
```

### 2. Install uv
The project's dependencies are pinned in `pyproject.toml` and `uv.lock` and managed with [uv](https://docs.astral.sh/uv/). If you don't already have it:

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Alternative (using pip):**
```bash
pip install uv
```

### 3. Create the Environment and Install the Package
```bash
uv sync
```
This creates a virtual environment in `.venv` with Python 3.13 (downloading it if necessary), installs the pinned dependencies from `uv.lock`, and installs the `las_model` package itself in editable mode, so `import las_model` works from anywhere once the environment is active.

Activate the environment before running anything:
```bash
# macOS/Linux
source .venv/bin/activate
# Windows
.venv\Scripts\activate
```

If you prefer another environment manager (for example conda), install the dependencies listed in `pyproject.toml` and point Python at the sources with `PYTHONPATH=./src` when running the scripts below.

### 4. Point the Code at Your Data Directory
All simulation outputs and figure inputs live outside the repository, in a directory the code calls `PROJECT_DIR`. It is read from the `ROOT_DIR` environment variable by `src/las_model/utils/config.py`, which also loads a `.env` file from the repository root if one exists (`.env` is git-ignored). Either create that file once:
```bash
echo "ROOT_DIR=/path/to/your/data/directory" > .env
```
or set the variable for each command:
```bash
ROOT_DIR=/path/to/your/data/directory python src/las_model/figures/plot_figures.py
```

The directory must contain a `graphics/` folder with the schematic PNGs that the figures place next to the plots. Simulation outputs are written beneath it by `las_model.utils.output.save_experiment`, one folder per experiment:
```
<ROOT_DIR>/<experiment_directory>/<experiment_name>/<experiment_name>.pickle   # the data
<ROOT_DIR>/<experiment_directory>/<experiment_name>/metadata.json              # the parameters it was run with
```

### 5. Verify the Installation
```bash
python -c "from las_model.utils.config import PROJECT_DIR; print(f'Project directory: {PROJECT_DIR}')"
```
This should print the data directory you configured. If it fails with a `TypeError` mentioning `NoneType`, `ROOT_DIR` is not set.

## Running the Code

### Simulations
Each experiment is one script under `src/las_model/simulations/figure_XX/`, named after the figure it feeds. Its parameters are the `metadata` dictionary at the top of the script, which is also saved alongside the results. Run a script directly, for example:
```bash
python src/las_model/simulations/figure_02/satprod_PprodAsweep.py
```
Parameter sweeps run their points in parallel across all CPU cores. A few supplementary figures reuse another figure's simulation; their folders hold a stub or README pointing at the script to run.

### Figures
`src/las_model/figures/plot_figures.py` draws the main and supplementary figures. Without arguments it opens each figure in a window; with `--save` it writes them to disk instead:
```bash
python src/las_model/figures/plot_figures.py                          # show every figure
python src/las_model/figures/plot_figures.py figure_02 figure_S04     # just these
python src/las_model/figures/plot_figures.py --save figs --format pdf --dpi 300
python src/las_model/figures/plot_figures.py --help
```
Each figure is a function, `figure_01` ... `figure_S21`, so in Spyder or VS Code you can run the file to define them and then call one, e.g. `figure_02()`, from the console. A figure whose data has not been generated yet is reported as failed and the others still run.

Figures S22 to S24 have their own small plotting scripts next to their simulations, `plot_figure_s22.py`, `plot_figure_s23.py` and `plot_figure_s24.py`.

### Roboto Font (optional)
The figure script labels its panels with the Roboto font. If Roboto is not installed, matplotlib falls back to DejaVu Sans and prints a warning for every label:
```
findfont: Font family 'roboto' not found.
```
The figures still render; only the panel letters look different. To install the font:

1. Download the Roboto family (Apache License 2.0) from Google Fonts: https://fonts.google.com/specimen/Roboto
2. Install the `.ttf` files:
   - **macOS:** open each file and click *Install* in Font Book, or copy them to `~/Library/Fonts/`.
   - **Linux:** copy them to `~/.local/share/fonts/` and run `fc-cache -f`.
   - **Windows:** right-click each file and choose *Install*.
3. Delete matplotlib's cached font list so it rescans the system fonts on the next run:
   ```bash
   # macOS
   rm ~/.matplotlib/fontlist-*.json
   # Linux
   rm ~/.cache/matplotlib/fontlist-*.json
   ```
4. Verify that matplotlib can see the font:
   ```bash
   python -c "from matplotlib import font_manager; print(sorted({f.name for f in font_manager.fontManager.ttflist if 'Roboto' in f.name}))"
   ```
   This should print a list containing `'Roboto'`.
