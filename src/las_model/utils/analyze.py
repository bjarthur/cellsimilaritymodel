import os
from concurrent.futures import ProcessPoolExecutor
import numpy as np 
from las_model.utils import motiffunc as mf 

def calculate_division_differences(divStates, rng):
    """
    Calculate division differences for a set of cell division states.

    Parameters:
    -----------
    divStates : np.ndarray
        Array of division states with shape (nVars, nCells).
    rng : np.random.Generator
        Random number generator for stochastic simulations.

    Returns:
    --------
    dsis : np.ndarray
        Array of division state differences for individual cells.
    drnd : np.ndarray
        Array of division state differences for random cell pairs.
    vardsis : np.ndarray
        Variance of dsis across cells.
    vardrnd : np.ndarray
        Variance of drnd across cells.
    normvar : np.ndarray
        Normalized variance (1 - vardsis / vardrnd).
    """

    nCells = divStates.shape[1]
    divStatesInt = divStates.astype('int')

    # One binomial draw per cell
    cell1 = rng.binomial(divStatesInt, 0.5)

    # One random partner index per cell, then gather + draw for all partners at once
    partnerIdx = rng.integers(0, nCells, size=nCells)
    cell2 = rng.binomial(divStatesInt[:, partnerIdx], 0.5)

    dsis = divStates - 2 * cell1
    drnd = cell1 - cell2

    vardsis = np.var(dsis, axis=1)
    vardrnd = np.var(drnd, axis=1)
    normvar = 1 - (vardsis / vardrnd)

    return dsis, drnd, vardsis, vardrnd, normvar

def _simulate_cell_triplet(task_args):
    """Simulate the 3 offspring cells (sis1, sis2, rnd1) for a single cell index."""
    i, sis1_state, sis2_state, rnd1_state, mother_cell, metadata, seed = task_args
    child_rng = np.random.default_rng(seed)

    sis1 = mf.Cell(metadata['Tcc'], metadata['varTcc'], child_rng)
    sis1.inherit(mother_cell, sis1_state)
    sis1.run(metadata['nCycles'])
    molecules_sis1 = sis1.getMolecules()

    sis2 = mf.Cell(metadata['Tcc'], metadata['varTcc'], child_rng)
    sis2.inherit(mother_cell, sis2_state)
    sis2.run(metadata['nCycles'])
    molecules_sis2 = sis2.getMolecules()

    rnd1 = mf.Cell(metadata['Tcc'], metadata['varTcc'], child_rng)
    rnd1.inherit(mother_cell, rnd1_state)
    rnd1.run(metadata['nCycles'])
    molecules_rnd1 = rnd1.getMolecules()

    return i, molecules_sis1, molecules_sis2, molecules_rnd1


def calculate_offspring_similarity_time(motherCell, metadata, rng, num_workers=None):
    # Get division states and create offspring cells 
    divStates = (motherCell.getMotherStates()).astype('int')

    sis1states = rng.binomial(divStates, 0.5)
    sis2states = divStates - sis1states 

    partnerIdx = rng.integers(0, metadata['nCells'], size=metadata['nCells'])
    rnd1states = rng.binomial(divStates[:, partnerIdx], 0.5)

    n_cells = metadata['nCells']
    if num_workers is None:
        num_workers = min(os.cpu_count() or 4, n_cells)

    # Generate statistically independent seeds for each task
    seeds = rng.integers(0, 2**63 - 1, size=n_cells)

    tasks = [
        (i, sis1states[:, i], sis2states[:, i], rnd1states[:, i], motherCell, metadata, seeds[i])
        for i in range(n_cells)
    ]

    mol_sis1 = [None] * n_cells
    mol_sis2 = [None] * n_cells
    mol_rnd1 = [None] * n_cells

    if num_workers == 1:
        for task in tasks:
            i, m_s1, m_s2, m_r1 = _simulate_cell_triplet(task)
            mol_sis1[i] = m_s1
            mol_sis2[i] = m_s2
            mol_rnd1[i] = m_r1
    else:
        print(f"Simulating {n_cells} cells across {num_workers} parallel workers...")
        import multiprocessing as mp
        ctx = mp.get_context('fork')
        with ProcessPoolExecutor(max_workers=num_workers, mp_context=ctx) as executor:
            for i, m_s1, m_s2, m_r1 in executor.map(_simulate_cell_triplet, tasks):
                mol_sis1[i] = m_s1
                mol_sis2[i] = m_s2
                mol_rnd1[i] = m_r1

    # Stack molecule lists and compute pairwise differences
    sis1stack = np.stack(mol_sis1, axis=1)
    sis2stack = np.stack(mol_sis2, axis=1)
    rnd1stack = np.stack(mol_rnd1, axis=1)

    dsis = sis1stack - sis2stack
    drnd = sis1stack - rnd1stack

    vardsis = np.var(dsis, axis=1)
    vardrnd = np.var(drnd, axis=1)
    normvar = 1 - vardsis / vardrnd

    return dsis, drnd, vardsis, vardrnd, normvar

def calcOrder(B,kcat,Km,A):
    """
    Compute local reaction order w.r.t. B via finite-difference in log-log space.
    B, A: scalar or array-like (elementwise-compatible shapes).
    Returns: array (or scalar) of same broadcasted shape.
    """
    B = np.asarray(B, dtype=float)
    A = np.asarray(A, dtype=float)
    
    B1 = B+1 
    rate0 = kcat/2*(A+B+Km-np.sqrt((A+B+Km)**2-4*A*B))
    rate1 = kcat/2*(A+B1+Km-np.sqrt((A+B1+Km)**2-4*A*B1))
       
    logRate0 = np.log10(rate0)
    logRate1 = np.log10(rate1)
    
    logB = np.log10(B)
    logB1 = np.log10(B1)
    
    return (logRate1-logRate0)/(logB1-logB)