# 3 Step Cascade Spatial Simulation: relatedness, cousin maps, concentration maps, Moran's I
import pickle
from datetime import datetime 
import numpy as np
from las_model.utils.config import PROJECT_DIR
from las_model.utils.output import save_experiment

# Experiment metadata
metadata = {
    'experiment_name': 'cascade_spatial_analyze',
    'experiment_directory': 'cascade',
    'created': datetime.now().isoformat(),
    'source_experiment': 'cascade_spatial',
    'moranI_experiment': 'cascade_spatial_moranI',
    'maxRadius': 9,
    'relatedness_t': 10000,
    'crop': [25,75],
    'cousinCell': 100,
    'cousin_ts': [2000,4000,6000,8000,10000],
    'cousinCells_all': [100,200,300,400,500],
    'cousin_ts_all': [0,2000,4000,6000,8000,10000],
    'molecules': ['A','B','C'],
    'mol_ts': [4000,6000,8000,10000],
    'moranI_shape': 'discdist',
    'moranI_radii': list(range(1,10)),
}

base_dir = PROJECT_DIR / metadata['experiment_directory']
crop = slice(metadata['crop'][0],metadata['crop'][1])

# Load grid from the spatial simulation 
with open(base_dir / metadata['source_experiment'] / f"{metadata['source_experiment']}.pickle",'rb') as f:
    grid = pickle.load(f)

# Collect Moran's I across neighborhood sizes (cheap; fail early if any run is missing)
morIs = []
for r in metadata['moranI_radii']:
    name = f"{metadata['moranI_experiment']}_{metadata['moranI_shape']}_r{r}"
    with open(base_dir / name / f'{name}.pickle','rb') as f:
        morIs.append(pickle.load(f))
morIs = np.stack(morIs,axis=0)

# Compute relatedness curve 
relatedness = grid.calcCollectiveLocalRelatedness(metadata['maxRadius'],metadata['relatedness_t'])

# Pull cousin maps 
def cousinMap(cellNum,t):
    img = 8-grid.cousinMap(cellNum,t)[crop,crop]
    img[np.where(img==10)] = 'NaN'
    return img

cousinmaps = []
for t in metadata['cousin_ts']:
    cousinmaps.append(cousinMap(metadata['cousinCell'],t))

cousinmaps_all = []
for cellNum in metadata['cousinCells_all']:
    cousinmaps_all.append([cousinMap(cellNum,t) for t in metadata['cousin_ts_all']])

# Pull molecular concentration maps 
molConcMaps = []
for molecule in metadata['molecules']:
    molConcMaps.append([grid.getFrame(t,molecule)[crop,crop] for t in metadata['mol_ts']])

# Save results 
results = {
    'relatedness': relatedness,
    'cousinmaps': cousinmaps,
    'cousinmaps_all': cousinmaps_all,
    'molConcMaps': molConcMaps,
    'moranIs': morIs,
}
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data=results,
    metadata=metadata,
    base_dir=base_dir
)
print(f"Experiment saved to {exp_dir}")
