## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/sgrolab/cellsimilaritymodel.git
cd cellsimilaritymodel
```

### 2. Install uv
If you don't already have `uv` installed, install it using one of the following methods:

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

### 3. Create Virtual Environment and Install Dependencies
```bash
# Create a virtual environment with uv
uv venv

# Activate the virtual environment
# On macOS/Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# Install project dependencies
uv pip install -r requirements.txt
```

### 4. Configure Project Settings
Create a configuration file to specify your data directory:
```bash
# Create the utils directory if it doesn't exist
mkdir -p utils

# Create the config.py file
cat > utils/config.py << EOF
"""
Configuration file for project paths and settings
"""
import os
from pathlib import Path

# Set the project directory as the home directory for data
PROJECT_DIR = Path("/path/to/your/data/directory")

# Create data directory if it doesn't exist
PROJECT_DIR.mkdir(parents=True, exist_ok=True)

EOF
```

**Important:** Edit `utils/config.py` and replace `/path/to/your/data/directory` with your actual data directory path.

### 5. Verify Installation
```bash
# Test that the environment is set up correctly
python -c "from utils.config import PROJECT_DIR; print(f'Project directory: {PROJECT_DIR}')"
```

You should see your configured project directory path printed to the console.

### 6. Install the Roboto Font (optional)
The figure script `src/las_model/figures/plot_figures.py` labels its panels with the Roboto font. If Roboto is not installed, matplotlib falls back to DejaVu Sans and prints a warning for every label:
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
