# 3 Step Cascade Spatial Simulation: Moran's I for every neighborhood size and weight shape
import pickle
from datetime import datetime
import numpy as np
from las_model.utils.config import PROJECT_DIR
from las_model.utils.output import save_experiment

# Experiment metadata
metadata = {
    'experiment_name': 'cascade_spatial_moranI',
    'experiment_directory': 'cascade',
    'created': datetime.now().isoformat(),
    'source_experiment': 'cascade_spatial',
    'neighborhoodsizes': list(range(1,10)),
    'shapes': ['discdist','discstep','donut','gausdist'],
    'timestep': 100,
    'molecules': ['A','B','C'],
}

# Load grid from the spatial simulation
source_dir = PROJECT_DIR / metadata['experiment_directory'] / metadata['source_experiment']
with open(source_dir / f"{metadata['source_experiment']}.pickle",'rb') as f:
    grid = pickle.load(f)

timepoints = range(0,int(grid.timepoints[-1]),metadata['timestep'])

# Calculate Moran's I for each molecule over time, saved as one experiment per shape and size
for shape in metadata['shapes']:
    for neighborhoodsize in metadata['neighborhoodsizes']:
        print(f"Calculating Moran's I for shape={shape}, neighborhoodsize={neighborhoodsize}")

        morIs = np.zeros([5,len(timepoints)])
        for i in range(len(timepoints)):
            for j in range(len(metadata['molecules'])):
                morIs[j,i] = grid.calcMoranI(neighborhoodsize,timepoints[i],metadata['molecules'][j],shape)

        combo_metadata = {
            **metadata,
            'experiment_name': f"{metadata['experiment_name']}_{shape}_r{neighborhoodsize}",
            'neighborhoodsize': neighborhoodsize,
            'shape': shape,
        }

        exp_dir = save_experiment(
            experiment_name=combo_metadata['experiment_name'],
            data=morIs,
            metadata=combo_metadata,
            base_dir=PROJECT_DIR / metadata['experiment_directory']
        )
        print(f"Experiment saved to {exp_dir}")
