# Production motif with a fixed reactant: the reactant B is held at a fixed
# concentration (its amount tracks cell volume) while the enzyme A and the
# product are simulated stochastically. Used by the figure_S06 sweeps.
#
# The Gillespie loop is a numba kernel that accumulates the concentration
# statistics as it goes and keeps only the mother-cell states, so memory does
# not grow with the number of reaction steps.
import math
import numba as nb
import numpy as np


@nb.njit(fastmath=True, nogil=True)
def _simulate_lineage(PprodA, PprodB, kcatA, Km, Tcc, nCells, rng, x0, motherMolecules, S1, S2):
    """
    Simulate nCells generations from the initial amounts x0 = (A, B, C) at volume 1.
    Writes the pre-division amounts into motherMolecules (3, nCells) and accumulates
    into S1 and S2 the sums and sums of squares of the concentrations molecules/volume
    of every trajectory sample, shifted by the initial concentrations. Returns the
    number of samples.
    """
    A = x0[0]
    B = x0[1]
    C = x0[2]
    V = 1.0

    # concentrations are accumulated relative to the initial state 
    A0 = A / V
    B0 = B / V
    C0 = C / V
    count = 1

    for gen in range(nCells):

        while V < 2:

            # calculate reaction probabilities 
            prodA = PprodA
            prodB = kcatA/2 * (Km+A+B-math.sqrt((Km+A+B)**2-4*A*B))

            Rtot = prodA + prodB

            # generate random numbers
            r1 = rng.random()
            r2 = rng.random() * Rtot

            # calculate time step
            tau = -math.log(r1)/Rtot

            # pick reaction 
            if r2 < prodA: # produce enzyme 
                A += 1
            else:           # produce product 
                C += 1

            # update volume 
            V += tau/Tcc

            # update amount of substrate 
            B = V * PprodB * Tcc

            # accumulate this sample 
            a = A/V - A0
            b = B/V - B0
            c = C/V - C0
            S1[0] += a
            S1[1] += b
            S1[2] += c
            S2[0] += a*a
            S2[1] += b*b
            S2[2] += c*c
            count += 1

        # mother state just before division: the state after the reaction that
        # pushed V past 2. The earlier array-based engine overwrote that last
        # sample with the newborn state and then located mothers with
        # np.where(volume == 1) - 1, which landed one reaction earlier, so its
        # mother states were low by one molecule of whichever species reacted
        # last (issue #86).
        motherMolecules[0,gen] = A
        motherMolecules[1,gen] = round(B)
        motherMolecules[2,gen] = C

        # divide: the newborn state replaces the last sample 
        a = A/V - A0
        b = B/V - B0
        c = C/V - C0
        S1[0] -= a
        S1[1] -= b
        S1[2] -= c
        S2[0] -= a*a
        S2[1] -= b*b
        S2[2] -= c*c

        V = 1.0
        A = float(rng.binomial(int(A), 0.5))
        B = V * PprodB * Tcc
        C = float(rng.binomial(int(C), 0.5))

        a = A/V - A0
        b = B/V - B0
        c = C/V - C0
        S1[0] += a
        S1[1] += b
        S1[2] += c
        S2[0] += a*a
        S2[1] += b*b
        S2[2] += c*c

    return count


def simulate_fixed_reactant(PprodA, PprodB, kcatA, Km, Tcc, nCells, rng):
    """Gillespie simulation of nCells generations of one lineage.

    Returns
    -------
    means : (3,) mean concentration (amount / volume) of enzyme, reactant and
        product over every sample of the trajectory
    variances : (3,) variance of the same concentrations
    motherMolecules : (3, nCells) molecule amounts just before each division,
        with the reactant rounded to an integer count
    """
    # initial amounts at volume 1 
    enzyme_i = PprodA * Tcc
    reactant_i = PprodB * Tcc
    product_i = 3/2 * kcatA/2*(Km+enzyme_i+reactant_i-np.sqrt((Km+enzyme_i+reactant_i)**2-4*enzyme_i*reactant_i))
    x0 = np.array([enzyme_i, reactant_i, product_i], dtype=np.float64)

    motherMolecules = np.zeros((3, nCells))
    S1 = np.zeros(3)
    S2 = np.zeros(3)

    count = _simulate_lineage(float(PprodA), float(PprodB), float(kcatA), float(Km), float(Tcc),
                              int(nCells), rng, x0, motherMolecules, S1, S2)

    # Variance from running sums, so the trajectory never has to be stored.
    # Expanding sum((x-mu)^2) = sum(x^2) - 2*mu*sum(x) + N*mu^2 with sum(x) = N*mu gives
    #     Var(x) = sum(x^2)/N - (sum(x)/N)^2
    # Taken literally that subtracts two nearly equal numbers when Var(x) << mu^2, so
    # the kernel accumulates deviations d = x - x0 from the initial state instead
    # (variance is shift-invariant, and x0 is close to the mean):
    #     mean = x0 + sum(d)/N        Var = sum(d^2)/N - (sum(d)/N)^2
    # Welford's online update is the usual alternative and is equivalent in exact
    # arithmetic:
    #     mean_k = mean_{k-1} + (x_k - mean_{k-1})/k
    #     M2_k   = M2_{k-1} + (x_k - mean_{k-1})*(x_k - mean_k)       Var = M2_N/N
    # The shifted sums are simpler in the kernel and matched np.var of the stored
    # trajectory to the last bit in testing.
    means = x0 + S1/count
    variances = S2/count - (S1/count)**2

    return means, variances, motherMolecules
