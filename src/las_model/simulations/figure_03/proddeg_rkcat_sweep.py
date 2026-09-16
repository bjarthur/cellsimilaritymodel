# TODO: run this file

# Production and Degradation: Rkcat sweep 
from datetime import datetime 
import numpy as np
from las_model.utils import motiffunc as mf
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_offspring_similarity_time
from las_model.utils.output import save_experiment 

# Experiment metadata
metadata = {
    'experiment_name': 'proddeg_rkcat_sweep',
    'experiment_directory': 'prodanddeg',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 100,
    'nCells_equilibrium': 10,
    'nCycles': 10,
    'Tcc': 1000,
    'varTcc': 0,
    'circuit': 'proddeg',
    'PprodA': 10**-1,
    'kcatAs': list(np.logspace(-1.3,0,20)),
    'PprodB': 10**-1
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

# Calculate Rkcat values 
kcatBs = 2*np.array(metadata['kcatAs'])-0.1
K_Ms = (np.array(metadata['kcatAs']) * metadata['PprodA'] - kcatBs * metadata['PprodB'] /2) * metadata['Tcc']**2

sweep_values = {
    'kcatAs': metadata['kcatAs'],
    'kcatBs': kcatBs,
    'K_Ms': K_Ms
}

# Accumulate results 
results = {
    'dsis': [],
    'drnd': [],
    'vardsis': [],
    'vardrnd': [],
    'normvar': [],
}

# === Iterate over kcatA values and simulate cells =====
for i in range(len(metadata['kcatAs'])):

    # Set kcatA, kcatB, KM values 
    kcatA = metadata['kcatAs'][i]
    kcatB = kcatBs[i]
    K_M = K_Ms[i]

    print(f"Simulating for kcatA = {kcatA}")

    # Initialize and run Mother Cell 
    motherCell = mf.Cell(metadata['Tcc'],metadata['varTcc'],rng)
    motherCell.parameterize(
        metadata['circuit'],
        [metadata['PprodA'], metadata['PprodB'], kcatA, kcatB, K_M]
    )
    motherCell.equilibrate(metadata['nCells_equilibrium'])
    motherCell.run(metadata['nCells'])

    # Calculate offspring similarity 
    dsis, drnd, vardsis, vardrnd, normvar = calculate_offspring_similarity_time(motherCell, metadata, rng)

    # append results 
    results['dsis'].append(dsis)
    results['drnd'].append(drnd)
    results['vardsis'].append(vardsis)
    results['vardrnd'].append(vardrnd)
    results['normvar'].append(normvar)

# Stack results 
results = {k: np.stack(v,axis=0) for k, v in results.items()}

# Save results 
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data = [sweep_values,results],
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to f{exp_dir}")
