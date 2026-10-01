# Saturated Production: effect of bursting on LAS duration 
from datetime import datetime 
import numpy as np 
from las_model.utils import motiffunc as mf
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_offspring_similarity_time
from las_model.utils.output import save_experiment 

# Experiment metadata 
metadata = {
    'experiment_name': 'satprod_burst_time',
    'experiment_directory': 'satprod',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'nCycles': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'prodsat',
    'circuit_burst': 'prodsat_burst',
    'PprodA': 10**-2,
    'kcatA': 10**-1,
    'burstSize': 10,
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

# The bursting cell makes bursts of burstSize molecules at PprodA / burstSize, so the mean production is unchanged 
conditions = {
    'no_burst': (metadata['circuit'], [metadata['PprodA'], metadata['kcatA']]),
    'burst': (metadata['circuit_burst'], [metadata['PprodA'] / metadata['burstSize'], metadata['kcatA'], metadata['burstSize']]),
}

results = {}
for condition, (circuit, params) in conditions.items():

    print(f"Simulating {condition} mother cell")

    # Initialize and run mother cell 
    motherCell = mf.Cell(metadata['Tcc'], metadata['varTcc'], rng)
    motherCell.parameterize(circuit, params)
    motherCell.equilibrate(metadata['nCells_equilibrium'])
    motherCell.run(metadata['nCells'])

    # Offspring similarity over time 
    dsis, drnd, vardsis, vardrnd, normvar = calculate_offspring_similarity_time(motherCell, metadata, rng)

    results[condition] = {
        'vardsis': vardsis,
        'vardrnd': vardrnd,
        'normvar': normvar,
    }

# Save results 
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data=results,
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to {exp_dir}")
