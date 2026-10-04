# Burst Size Sweep, repeated for each total A production rate
from datetime import datetime
import numpy as np
from las_model.utils import motiffunc as mf
from las_model.utils.config import PROJECT_DIR
from las_model.utils.output import save_experiment

# Experiment metadata
metadata = {
    'experiment_name': 'burstSize_prodA',
    'experiment_directory': 'burstSize',
    'created': datetime.now().isoformat(),
    'seed': 1000,
    'nCells': 1000,
    'nCycles_equilibrium': 10,
    'Tcc': 1000,
    'circuit': 'prodsat_burst',
    'burstSizes': list(np.linspace(1,20,20)),
    'prodA_std_exponents': [0, -1, -2],   # prodA_std = 10**exponent
    'kcatA': 10**-2
}

# Set burst size array
burstSizes = np.array(metadata['burstSizes'])
kcatA = metadata['kcatA']

# === Iterate over total A production rates, then burst sizes ====
for exponent in metadata['prodA_std_exponents']:

    # Pin random seed, anew per production rate so each matches a standalone run
    rng = np.random.default_rng(seed=metadata['seed'])

    prodA_std = 10**exponent
    prodAs = prodA_std / burstSizes

    # Initialize arrays to store simulation results
    Aeqs = np.zeros([len(burstSizes)])
    Beqs = np.zeros_like(Aeqs)
    normvarAs = np.zeros_like(Aeqs)
    normvarBs = np.zeros_like(Aeqs)

    for i in range(len(burstSizes)):

        # Set burst size and PprodA value
        burstSize = burstSizes[i]
        prodA = prodAs[i]

        print(f"Simulating prodA_std 10**{exponent} with burst size {burstSize} and prodA {prodA}")

        motherCell = mf.Cell(metadata['Tcc'],0,rng)
        motherCell.parameterize(metadata['circuit'],[prodA,kcatA,burstSize])
        motherCell.equilibrate(metadata['nCycles_equilibrium'])

        # Run simulation
        motherCell.run(metadata['nCells'])

        # Get molecules
        molecules = motherCell.getMolecules()

        Aeqs[i] = np.mean(molecules[0])
        Beqs[i] = np.mean(molecules[1])

        divStates = motherCell.getMotherStates()

        dsis = np.zeros([metadata['nCells'],6])
        drnd = np.zeros([metadata['nCells'],6])

        for k in range(metadata['nCells']):
            cell1 = rng.binomial(divStates[:,k].astype('int'),0.5)
            cell2 = rng.binomial(divStates[:,rng.integers(0,metadata['nCells'])].astype('int'),0.5)

            dsis[k] = divStates[:,k] - 2*cell1
            drnd[k] = cell1 - cell2

        normvarAs[i] = 1-np.var(dsis[:,0],axis=0)/np.var(drnd[:,0],axis=0)
        normvarBs[i] = 1-np.var(dsis[:,1],axis=0)/np.var(drnd[:,1],axis=0)

        print(f"normvarAs[i]: {normvarAs[i]}, normvarBs[i]: {normvarBs[i]}")

    # Save this production rate's sweep as its own experiment
    run_metadata = {
        **metadata,
        'experiment_name': f"{metadata['experiment_name']}-{-exponent}",
        'prodA_std': prodA_std,
    }

    exp_dir = save_experiment(
        experiment_name=run_metadata['experiment_name'],
        data=[burstSizes, prodAs, Aeqs, Beqs, normvarAs, normvarBs],
        metadata=run_metadata,
        base_dir=PROJECT_DIR / metadata['experiment_directory']
    )
    print(f"Saved experiment to {exp_dir}")
