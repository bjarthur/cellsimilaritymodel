# Production motif with a fixed reactant: the reactant B is held at a fixed
# concentration (its amount tracks cell volume) while the enzyme A and the
# product are simulated stochastically. Used by the figure_S06 sweeps.
import numpy as np

def simulate_fixed_reactant(PprodA, PprodB, kcatA, Km, Tcc, nCells, rng, maxSteps=int(1e9)):
    """Gillespie simulation of nCells generations of one lineage.

    maxSteps preallocates the trajectory buffers; the run raises IndexError if
    the sweep needs more reaction steps than that.

    Returns
    -------
    molecules : (3, nSteps) enzyme, reactant and product amounts at every step
    volume : (nSteps,) cell volume, 1 at birth and 2 at division
    times : (nSteps,) simulation time
    motherMolecules : (3, nCells) molecule amounts just before each division,
        with the reactant rounded to an integer count
    """
    # storage arrays 
    molecules = np.zeros([3,maxSteps])
    volume = np.zeros(maxSteps)
    times = np.zeros_like(volume)

    enzyme_i = PprodA * Tcc
    reactant_i = PprodB * Tcc
    product_i = 3/2 * kcatA/2*(Km+enzyme_i+reactant_i-np.sqrt((Km+enzyme_i+reactant_i)**2-4*enzyme_i*reactant_i))

    molecules[:,0] = [enzyme_i,reactant_i,product_i]
    volume[0] = 1

    # initialize counters
    gen = 0
    n = 1

    while gen < nCells:
        
        while volume[n-1] < 2:
            
            # get enzyme amount from previous timestep 
            A = molecules[0,n-1]
            
            # set amount of substrate based on volume 
            B = volume[n-1] * PprodB * Tcc
            
            # calculate reaction probabilities 
            prodA = PprodA
            prodB = kcatA/2 * (Km+A+B-np.sqrt((Km+A+B)**2-4*A*B))
            
            Rtot = prodA + prodB
            
            # generate random numbers
            r1 = rng.uniform()
            r2 = rng.uniform() * Rtot
            
            # calculate time step
            tau = 1/Rtot*np.log(1/r1)
            
            # pick reaction 
            if r2 < prodA: # produce enzyme 
                molecules[0,n] = molecules[0,n-1] + 1
                molecules[2,n] = molecules[2,n-1]
            else:           # produce product 
                molecules[0,n] = molecules[0,n-1]
                molecules[2,n] = molecules[2,n-1] + 1
        
            # update volume 
            volume[n] = volume[n-1] + tau/Tcc
            
            # update time
            times[n] = times[n-1] + tau
            
            # update amount of substrate 
            molecules[1,n] = volume[n] * PprodB * Tcc
            
            # update counter 
            n+=1
        
        # divide 
        volume[n-1] = 1
        molecules[0,n-1] = rng.binomial(molecules[0,n-1],0.5)
        molecules[1,n-1] = volume[n-1] * PprodB*Tcc
        molecules[2,n-1] = rng.binomial(molecules[2,n-1],0.5)
        
        # update generation counter 
        gen += 1

    # trim zeros (copy so the oversized buffers are released) 
    molecules = molecules[:,0:n].copy()
    volume = volume[0:n].copy()
    times = times[0:n].copy()

    # get mother states 
    motherTimes = np.where(volume==1)[0][1::]-1
    motherMolecules = molecules[:,motherTimes]
    motherMolecules[1] = np.round(motherMolecules[1])

    return molecules, volume, times, motherMolecules
