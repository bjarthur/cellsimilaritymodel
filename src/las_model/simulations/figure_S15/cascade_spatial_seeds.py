# 3 Step Cascade Spatial Simulation: relatedness curves for different random seeds
from datetime import datetime
import numpy as np
from las_model.utils import gridfunc as gf
from las_model.utils.config import PROJECT_DIR
from las_model.utils.output import save_experiment

# Experiment metadata
metadata = {
    'experiment_name': 'cascade_spatial_seeds',
    'experiment_directory': 'cascade',
    'created': datetime.now().isoformat(),
    'seeds': [1000, 1001, 1002],
    'maxCells': 2**10,
    'gridSize': 101,
    'Tcc': 1000,
    'varTcc': 10,
    'circuit': 'cascade',
    'PprodA': 10**-1,
    'kcatA': 10**-2,
    'kcatB': 10**-2,
    'maxRadius': 9,
    'relatedness_t': 10000,
}

# Relatedness curves stay a list with one (cells, radii) array per seed: the number of 
# cells alive at relatedness_t differs between runs, so they cannot be stacked 
relatedness = []

for seed in metadata['seeds']:

    print(f"Simulating grid for seed = {seed}")

    # Pin random seed (gridfunc draws from its module-level rng)
    gf.rng = np.random.default_rng(seed=seed)

    # Initiate, seed and run grid
    grid = gf.Grid(metadata['gridSize'],metadata['gridSize'],metadata['maxCells'])
    grid.seed(metadata['circuit'],[metadata['PprodA'],metadata['kcatA'],metadata['kcatB']],metadata['Tcc'],metadata['varTcc'])
    grid.run()

    # Compute relatedness curve 
    relatedness.append(grid.calcCollectiveLocalRelatedness(metadata['maxRadius'],metadata['relatedness_t']))

# Save results 
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data=[metadata['seeds'], {'relatedness': relatedness}],
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to {exp_dir}")
