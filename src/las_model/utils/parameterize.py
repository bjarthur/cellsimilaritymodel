"""
Shared parameterize function for the Cell classes in motiffunc.py and gridfunc.py.

Sets circuit rate constants and initial molecule counts on a Cell instance.
"""

import numpy as np


def parameterize_cell(cell, circuit, params):
    if circuit == 'single':
        cell.prodA = params[0]
        cell.A[0] = cell.prodA * cell.Tcc

    elif circuit == 'pos_fb':
        cell.prodA = params[0]
        cell.k1 = params[1]
        cell.k2 = params[2]
        cell.k3 = params[3]

        cell.A[0] = cell.prodA + cell.k1 * cell.Tcc

    elif circuit == 'neg_fb':
        cell.k1 = params[0]
        cell.k2 = params[1]
        cell.k3 = params[2]

        cell.A[0] = cell.prodA * cell.Tcc - cell.k1 * cell.Tcc

    elif circuit == 'bind':
        cell.prodA = params[0]
        cell.prodB = params[1]
        cell.k1 = params[2]

        cell.A[0] = cell.prodA * cell.Tcc
        cell.B[0] = cell.prodB * cell.Tcc

    elif circuit == 'bind2':
        cell.prodA = params[0]
        cell.prodB = params[1]
        cell.k1 = params[2]/(cell.prodA*cell.prodB*cell.Tcc**2)

        cell.A[0] = cell.prodA * cell.Tcc
        cell.B[0] = cell.prodB * cell.Tcc

    elif circuit == 'revbind':
        cell.prodA = params[0]
        cell.prodB = params[1]
        cell.k1 = params[2]/(cell.prodA*cell.prodB*cell.Tcc**2)
        cell.k2 = params[3]/(np.min((cell.prodA,cell.prodB))*cell.Tcc)

        cell.A[0] = cell.prodA * cell.Tcc
        cell.B[0] = cell.prodB * cell.Tcc
        cell.C[0] = 0

    elif circuit == 'prodsat':
        cell.prodA = params[0]
        cell.k1 = params[1]

        cell.A[0] = cell.prodA * cell.Tcc
        cell.B[0] = 3/2 * cell.prodA * cell.k1 * cell.Tcc**2

    elif circuit == 'prodsat_burst':
        cell.prodA = params[0]
        cell.k1 = params[1]
        cell.burstSize = params[2]

        cell.A[0] = cell.prodA * cell.burstSize * cell.Tcc
        cell.B[0] = 3/2 * cell.prodA * cell.burstSize * cell.k1 * cell.Tcc**2

    elif circuit == 'produnsat':
        cell.prodA = params[0]
        cell.prodB = params[1]
        cell.k1 = params[2]
        cell.k2 = params[3]

        Ai = int(cell.prodB * cell.Tcc)
        Bi = int(cell.prodA * cell.Tcc)
        reactionFrac = cell.k1/2*(Ai+Bi+cell.k2-np.sqrt((Ai+Bi+cell.k2)**2-4*Ai*Bi))/(cell.k1*Ai)

        cell.A[0] = int(Bi * reactionFrac)
        cell.B[0] = Ai
        cell.C[0] = Bi - cell.A[0]

    elif circuit == 'prod_fixedB':
        cell.prodA = params[0]
        cell.prodB = params[1]
        cell.k1 = params[2]
        cell.k2 = params[3]

        cell.A[0] = int(cell.prodA *cell.Tcc)
        cell.B[0] = int(cell.prodB * cell.Tcc)
        cell.C[0] = int(4* cell.k1 * cell.Tcc * cell.prodA * cell.Tcc * cell.prodB * cell.Tcc / (cell.k2+cell.prodA * cell.Tcc * cell.prodB * cell.Tcc))

    elif circuit =='cascade':
        cell.prodA = params[0]
        cell.k1 = params[1]
        cell.k2 = params[2]

        cell.A[0] = cell.prodA * cell.Tcc
        cell.B[0] = 3/2 * cell.prodA * cell.k1 * cell.Tcc**2
        cell.C[0] = 3/2 * 3/2 * cell.prodA * cell.k1 * cell.k2 * cell.Tcc**3

    elif circuit == 'proddeg':
        cell.prodA = params[0]
        cell.prodB = params[1]
        cell.k1 = params[2]
        cell.k2 = params[3]
        cell.k3 = params[4]

        cell.A[0] = cell.prodA * cell.Tcc
        cell.B[0] = cell.prodB * cell.Tcc
        cell.C[0] = cell.k3

    elif circuit == 'phos':
        cell.prodA = params[0]
        cell.prodB = params[1]
        cell.k1 = params[2]
        cell.k2 = params[3]

    elif circuit == 'phos_int':
        cell.prodA = params[0]
        cell.prodB = params[1]
        cell.k1 = params[2]
        cell.k2 = params[3]
        cell.k3 = params[4]

        cell.A[0] = cell.prodA * cell.Tcc
        cell.C[0] = cell.prodB * cell.Tcc

    elif circuit == 'phos_cycle':
        cell.prodA = params[0]
        cell.prodB = params[1]
        cell.prodC = params[2]
        cell.k1 = params[3]/(cell.prodA*cell.prodB*cell.Tcc**2)
        cell.k2 = params[4]/(cell.prodC*cell.prodB*cell.Tcc**2)

        cell.A[0] = cell.prodA * cell.Tcc
        cell.B[0] = cell.prodB * cell.Tcc // 2
        cell.C[0] = cell.prodC * cell.Tcc
        cell.D[0] = cell.prodB * cell.Tcc -  cell.B[0]

    elif circuit == 'phos_sat':
        cell.prodA = params[0]
        cell.k1 = params[1]
        cell.k2 = params[2]
        cell.prodB = params[3]
        cell.prodC = params[4]
        cell.k3 = params[5]
        cell.k4 = params[6]

        cell.A[0] = cell.prodA * cell.Tcc
        cell.B[0] = cell.prodB * cell.Tcc // 2
        cell.C[0] = cell.prodC * cell.Tcc
        cell.D[0] = cell.prodB * cell.Tcc -  cell.B[0]

    elif circuit == 'phos2':
        cell.prodA = params[0]
        cell.prodB = params[1]
        cell.k1 = params[2]
        cell.k2 = params[3]
        cell.k3 = params[4]
        cell.k4 = params[5]
        cell.k5 = params[6]
        cell.k6 = params[7]
        cell.k7 = params[8]
        cell.k8 = params[9]

        cell.A[0] = cell.prodA * cell.Tcc
        cell.C[0] = cell.prodB * cell.Tcc

    elif circuit == 'diffTF':
        cell.prodA = params[0]
        cell.prodB = params[1]
        cell.k1 = params[2]/(cell.prodA*cell.prodB*cell.Tcc**2)
        cell.k2 = params[3]

        cell.A[0] = cell.prodA * cell.Tcc / 2
        cell.B[0] = cell.prodB * cell.Tcc / 2
        cell.C[0] = cell.prodA * cell.Tcc / 2
        cell.D[0] = cell.C[0] * cell.k2 * cell.Tcc

    elif circuit == 'cdg':
        cell.prodA = params[0]
        cell.prodB = params[1]
        cell.k1 = params[2]
        cell.k2 = params[3]
        cell.k3 = params[4]
        cell.k4 = params[5]

        cell.A[0] = cell.prodA * cell.Tcc
        cell.B[0] = cell.prodB * cell.Tcc
        cell.C[0] = cell.k3
        cell.D[0] = cell.C[0] * cell.k4 * cell.Tcc

    elif circuit == 'cascade6':
        cell.prodA = params[0]
        cell.k1 = params[1]
        cell.k2 = params[2]
        cell.k3 = params[3]
        cell.k4 = params[4]
        cell.k5 = params[5]

        cell.A[0] = cell.prodA * cell.Tcc
        cell.B[0] = cell.A[0] * (3/2*cell.k1*cell.Tcc)
        cell.C[0] = cell.B[0] * (3/2*cell.k2*cell.Tcc)
        cell.D[0] = cell.C[0] * (3/2*cell.k3*cell.Tcc)
        cell.E[0] = cell.D[0] * (3/2*cell.k4*cell.Tcc)
        cell.F[0] = cell.E[0] * (3/2*cell.k5*cell.Tcc)

    elif circuit =='crazyCascade':
        cell.prodA = params[0]      # PprodA
        cell.k1 = params[1]         # kcatA
        cell.prodB = params[2]      # PprodC
        cell.k2 = params[3]         # kcatC
        cell.k3 = params[4]         # KM,C
        cell.k4 = params[5]         # kcatB
        cell.prodC = params[6]      # PprodE
        cell.k5 = params[7]         # kcatE
        cell.k6 = params[8]         # KM,E
        cell.k7 = params[9]         # kcatD

    # grid_relay (gridfunc.py else branch; never reached by any script)
    # elif circuit == 'grid_relay':
    #     cell.prodA = params[0]
    #     cell.k1 = params[1]
    #     cell.prodC = params[2]
    #     cell.k2 = params[3]
    #     cell.k3 = params[4]

    else: # tcs (motiffunc.py else branch)
        cell.prodA = params[0]
        cell.prodB = params[1]
        cell.k1 = params[2]
        cell.k2 = params[3]
        cell.k3 = params[4]
        cell.k4 = params[5]
