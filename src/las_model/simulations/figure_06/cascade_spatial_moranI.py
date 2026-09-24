# 3 Step Cascade Spatial Simulation: Moran's I for one neighborhood size and weight shape
# usage: python cascade_spatial_moranI.py <neighborhoodsize> <shape>
# <neighborhoodsize> is an int, typically 1-9
# <shape> can be one of 'discdist', 'discstep', 'donut', 'gausdist'
import sys
import pickle
from datetime import datetime 
import numpy as np
from las_model.utils.config import PROJECT_DIR
from las_model.utils.output import save_experiment

neighborhoodsize = int(sys.argv[1])
shape = str(sys.argv[2])

# Experiment metadata
metadata = {
    'experiment_name': f'cascade_spatial_moranI_{shape}_r{neighborhoodsize}',
    'experiment_directory': 'cascade',
    'created': datetime.now().isoformat(),
    'source_experiment': 'cascade_spatial',
    'neighborhoodsize': neighborhoodsize,
    'shape': shape,
    'timestep': 100,
    'molecules': ['A','B','C'],
}

# Load grid from the spatial simulation 
source_dir = PROJECT_DIR / metadata['experiment_directory'] / metadata['source_experiment']
with open(source_dir / f"{metadata['source_experiment']}.pickle",'rb') as f:
    grid = pickle.load(f)

# Calculate Moran's I for each molecule over time 
timepoints = range(0,int(grid.timepoints[-1]),metadata['timestep'])
morIs = np.zeros([5,len(timepoints)])
for i in range(len(timepoints)):
    for j in range(len(metadata['molecules'])):
        morIs[j,i] = grid.calcMoranI(metadata['neighborhoodsize'],timepoints[i],metadata['molecules'][j],metadata['shape'])

# Save results 
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data=morIs,
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to {exp_dir}")
