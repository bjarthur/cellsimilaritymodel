# 3 Step Cascade Spatial Simulation
from datetime import datetime
import numpy as np
from las_model.utils import gridfunc as gf
from las_model.utils.config import PROJECT_DIR
from las_model.utils.output import save_experiment

# Experiment metadata
metadata = {
    'experiment_name': 'cascade_spatial',
    'experiment_directory': 'cascade',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'maxCells': 2**10,
    'gridSize': 101,
    'Tcc': 1000,
    'varTcc': 10,
    'circuit': 'cascade',
    'PprodA': 10**-1,
    'kcatA': 10**-2,
    'kcatB': 10**-2,
}

# Pin random seed (gridfunc draws from its module-level rng)
gf.rng = np.random.default_rng(seed=metadata['seed'])

# Initiate and seed grid
grid = gf.Grid(metadata['gridSize'],metadata['gridSize'],metadata['maxCells'])
grid.seed(metadata['circuit'],[metadata['PprodA'],metadata['kcatA'],metadata['kcatB']],metadata['Tcc'],metadata['varTcc'])

# Run grid
grid.run()

# Save results
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data=grid,
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to {exp_dir}")
