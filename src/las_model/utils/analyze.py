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

def simulate_offspring_time(motherCell, metadata, rng):
    """
    Divide every mother cell into two sisters, pair the first sister with a daughter of a
    random other mother, and run all three offspring for metadata['nCycles'] cycles.

    Returns the molecule trajectories sis1, sis2, rnd1, each of shape (nVars, nCells, nTimes).
    """
    # Get division states and create offspring cells 
    divStates = (motherCell.getMotherStates()).astype('int')

    sis1states = rng.binomial(divStates, 0.5)
    sis2states = divStates - sis1states 

    partnerIdx = rng.integers(0, metadata['nCells'], size=metadata['nCells'])
    rnd1states = rng.binomial(divStates[:, partnerIdx], 0.5)

    # preallocate molecules lists
    molcules = {
        'sis1': [],
        'sis2': [],
        'rnd1': []
    }

    # Divide cells and run offspring
    for i in range(metadata['nCells']):
        print(f"Simulating cell {i+1}/{metadata['nCells']}")
        
        sis1 = mf.Cell(metadata['Tcc'],metadata['varTcc'],rng)
        sis1.inherit(motherCell,sis1states[:,i])
        sis1.run(metadata['nCycles'])
        molecules_sis1 = sis1.getMolecules()
        
        sis2 = mf.Cell(metadata['Tcc'],metadata['varTcc'],rng)
        sis2.inherit(motherCell,sis2states[:,i])
        sis2.run(metadata['nCycles'])
        molecules_sis2 = sis2.getMolecules()
        
        rnd1 = mf.Cell(metadata['Tcc'],metadata['varTcc'],rng)
        rnd1.inherit(motherCell,rnd1states[:,i])
        rnd1.run(metadata['nCycles'])
        molecules_rnd1 = rnd1.getMolecules()

        molcules['sis1'].append(molecules_sis1)
        molcules['sis2'].append(molecules_sis2)
        molcules['rnd1'].append(molecules_rnd1)

    # Stack molecule lists 
    sis1stack = np.stack(molcules['sis1'],axis=1)
    sis2stack = np.stack(molcules['sis2'],axis=1)
    rnd1stack = np.stack(molcules['rnd1'],axis=1)

    return sis1stack, sis2stack, rnd1stack


def simulate_offspring_scramble_time(motherCell, metadata, rng):
    """
    Like simulate_offspring_time, but each sister is also rerun as a scrambled copy in which
    the molecules indexed by metadata['scrambled_molecules'] are instead inherited from a
    random other mother (a different one per sister). The random cell rnd1 is shared between
    the inherited and scrambled pairings.

    Returns sis1, sis2, rnd1, sis1scr, sis2scr, each of shape (nVars, nCells, nTimes).
    """
    # Get division states and create offspring cells
    divStates = (motherCell.getMotherStates()).astype('int')
    scrambled = list(metadata['scrambled_molecules'])

    sis1states = rng.binomial(divStates, 0.5)
    sis2states = divStates - sis1states

    partnerIdx = rng.integers(0, metadata['nCells'], size=metadata['nCells'])
    rnd1states = rng.binomial(divStates[:, partnerIdx], 0.5)

    # Redraw the scrambled molecules as daughters of each sister's own random other mother
    sis1scrstates = sis1states.copy()
    newMother1 = rng.integers(0, metadata['nCells'], size=metadata['nCells'])
    sis1scrstates[scrambled] = rng.binomial(divStates[np.ix_(scrambled, newMother1)], 0.5)

    sis2scrstates = sis2states.copy()
    newMother2 = rng.integers(0, metadata['nCells'], size=metadata['nCells'])
    sis2scrstates[scrambled] = rng.binomial(divStates[np.ix_(scrambled, newMother2)], 0.5)

    allstates = [sis1states, sis2states, rnd1states, sis1scrstates, sis2scrstates]

    # preallocate molecules lists
    molecules = [[] for _ in allstates]

    # Divide cells and run offspring
    for i in range(metadata['nCells']):
        print(f"Simulating cell {i+1}/{metadata['nCells']}")

        for states, cell_molecules in zip(allstates, molecules):
            cell = mf.Cell(metadata['Tcc'],metadata['varTcc'],rng)
            cell.inherit(motherCell,states[:,i])
            cell.run(metadata['nCycles'])
            cell_molecules.append(cell.getMolecules())

    # Stack molecule lists
    return tuple(np.stack(cell_molecules,axis=1) for cell_molecules in molecules)


def calculate_offspring_differences(sis1, sis2, rnd1):
    """
    Pairwise differences between the first sister and its sister (dsis) and between the
    first sister and a random cell (drnd), their variances across cells at each time point,
    and the normalized variance 1 - vardsis / vardrnd.
    """
    dsis = sis1 - sis2
    drnd = sis1 - rnd1

    vardsis = np.var(dsis, axis=1)
    vardrnd = np.var(drnd, axis=1)
    normvar = 1 - vardsis / vardrnd

    return dsis, drnd, vardsis, vardrnd, normvar


def calculate_offspring_correlation_time(sis1, sis2, rnd1):
    """
    Pearson correlation across cells, at each time point, between the first sister and its
    sister (rsis) and between the first sister and a random cell (rrnd). Shapes (nVars, nTimes).
    """
    def corr(x, y):
        x = x - x.mean(axis=1, keepdims=True)
        y = y - y.mean(axis=1, keepdims=True)
        return (x * y).sum(axis=1) / np.sqrt((x * x).sum(axis=1) * (y * y).sum(axis=1))

    return corr(sis1, sis2), corr(sis1, rnd1)


def calculate_offspring_similarity_time(motherCell,metadata,rng):
    sis1, sis2, rnd1 = simulate_offspring_time(motherCell, metadata, rng)
    return calculate_offspring_differences(sis1, sis2, rnd1)

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