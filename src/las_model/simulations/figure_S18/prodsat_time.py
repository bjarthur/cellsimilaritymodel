#TODO: Run simulation 

# Saturated Production Time Run
from datetime import datetime 
import numpy as np
from las_model.utils import motiffunc as mf
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import simulate_offspring_time, calculate_offspring_differences, calculate_offspring_correlation_time
from las_model.utils.output import save_experiment 

# Experiment metadata 
metadata = {
    'experiment_name': 'satprod_time',
    'experiment_directory': 'satprod/time',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCells_equilibrium': 10,
    'nCycles': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'prodsat',
    'PprodA': 10**-1,
    'kcatA': 10**-1
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

# Initialize and run Mother Cell
motherCell = mf.Cell(metadata['Tcc'],metadata['varTcc'],rng)
motherCell.parameterize(metadata['circuit'],[metadata['PprodA'],metadata['kcatA']])
motherCell.equilibrate(metadata['nCells_equilibrium'])
motherCell.run(metadata['nCells'])

# Simulate offspring, then pairwise differences and correlations over time 
sis1, sis2, rnd1 = simulate_offspring_time(motherCell,metadata,rng)
dsis, drnd, vardsis, vardrnd, normvar = calculate_offspring_differences(sis1, sis2, rnd1)
rsis, rrnd = calculate_offspring_correlation_time(sis1, sis2, rnd1)

results = {
    'dsis': dsis,
    'drnd': drnd,
    'vardsis': vardsis,
    'vardrnd': vardrnd,
    'normvar': normvar,
    'rsis': rsis,
    'rrnd': rrnd,
}

# Save results 
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data = results,
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to {exp_dir}")
