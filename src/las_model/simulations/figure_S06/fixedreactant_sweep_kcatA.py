# Fixed Reactant: sweep kcatA and PprodB 
from datetime import datetime 
import numpy as np 
from las_model.utils.fixedreactant import simulate_fixed_reactant
from las_model.utils.config import PROJECT_DIR
from las_model.utils.analyze import calculate_division_differences
from las_model.utils.output import save_experiment 

# Experiment metadata
metadata = {
    'experiment_name': 'fixedreactant_sweep_kcatA',
    'experiment_directory': 'fixed_reactant',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'Tcc': 1000,
    'PprodA': 10**-1,
    'kcatAs': list(np.logspace(-2,2,5)),
    'PprodBs': list(np.logspace(-3,3,31)),
    'Km': 10**3,
}

# Pin random seed 
rng = np.random.default_rng(seed=metadata['seed'])

# Accumulate results
results = {
    'means': [],
    'variances': [],
    'dsis': [],
    'drnd': [],
    'vardsis': [],
    'vardrnd': [],
    'normvar': [],
}

for kcatA in metadata['kcatAs']:
    for PprodB in metadata['PprodBs']:

        print(f"Running simulation for kcatA={kcatA}, PprodB={PprodB}")

        molecules, volume, times, motherMolecules = simulate_fixed_reactant(
            metadata['PprodA'], PprodB, kcatA, metadata['Km'], metadata['Tcc'], metadata['nCells'], rng)

        # Molecule concentration statistics 
        means = np.mean(molecules/volume,axis=1)
        variances = np.var(molecules/volume,axis=1)

        # Division differences from the mother states 
        dsis, drnd, vardsis, vardrnd, normvar = calculate_division_differences(motherMolecules,rng)

        # Store results
        results['means'].append(means)
        results['variances'].append(variances)
        results['dsis'].append(dsis)
        results['drnd'].append(drnd)
        results['vardsis'].append(vardsis)
        results['vardrnd'].append(vardrnd)
        results['normvar'].append(normvar)

# Stack results into a (kcatA, PprodB, ...) grid 
nKcatA, nPprodB = len(metadata['kcatAs']), len(metadata['PprodBs'])
results = {k: np.stack(v,axis=0) for k, v in results.items()}
results = {k: v.reshape(nKcatA, nPprodB, *v.shape[1:]) for k, v in results.items()}

# Save results 
exp_dir = save_experiment(
    experiment_name=metadata['experiment_name'],
    data = [[metadata['kcatAs'],metadata['PprodBs']],results],
    metadata=metadata,
    base_dir=PROJECT_DIR / metadata['experiment_directory']
)
print(f"Experiment saved to {exp_dir}")
