"""
Numba-accelerated Gillespie stochastic simulation algorithm kernels.

Provides JIT-compiled functions for running reaction steps and cell cycles
across all circuit motifs in cell.py at native machine speed.
"""

import math
import numba as nb
import numpy as np


@nb.njit(inline='always', fastmath=True)
def step_reaction(circuit, A, B, C, D, E, F, V, p, rng):
    """
    Execute a single Gillespie reaction event and compute time step tau.
    Returns (A, B, C, D, E, F, tau).
    """
    prodA = p[0]
    prodB = p[1]
    prodC = p[2]
    k1 = p[3]
    k2 = p[4]
    k3 = p[5]
    k4 = p[6]
    k5 = p[7]
    k6 = p[8]
    k7 = p[9]
    k8 = p[10]
    burstSize = p[11]
    Tcc = p[12]

    # --- 1. single ---
    if circuit == 'single':
        Rtot = prodA
        r1 = rng.random()
        tau = -math.log(r1) / Rtot
        A = A + 1

    # --- 2. pos_fb ---
    elif circuit == 'pos_fb':
        A_k3 = A ** k3
        k2_k3 = k2 ** k3
        prodA_eff = prodA + k1 * A_k3 / (k2_k3 + A_k3)
        Rtot = prodA_eff
        r1 = rng.random()
        tau = -math.log(r1) / Rtot
        A = A + 1

    # --- 3. neg_fb ---
    elif circuit == 'neg_fb':
        prodA_eff = k1 / (1.0 + (A / k2) ** k3)
        Rtot = prodA_eff
        r1 = rng.random()
        tau = -math.log(r1) / Rtot
        A = A + 1

    # --- 4. bind & bind2 ---
    elif circuit == 'bind' or circuit == 'bind2':
        r_prodC = k1 * A * B
        Rtot = prodA + prodB + r_prodC
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + prodB:
            B = B + 1
        else:
            A = A - 1
            B = B - 1
            C = C + 1

    # --- 5. revbind ---
    elif circuit == 'revbind':
        r_prodC = k1 * A * B
        revC = k2 * C
        Rtot = prodA + prodB + r_prodC + revC
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + prodB:
            B = B + 1
        elif r2 < prodA + prodB + r_prodC:
            A = A - 1
            B = B - 1
            C = C + 1
        else:
            A = A + 1
            B = B + 1
            C = C - 1

    # --- 6. prodsat ---
    elif circuit == 'prodsat':
        r_prodB = k1 * A
        Rtot = prodA + r_prodB
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        else:
            B = B + 1

    # --- 7. prodsat_burst ---
    elif circuit == 'prodsat_burst':
        r_prodB = k1 * A
        Rtot = prodA + r_prodB
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + burstSize
        else:
            B = B + 1

    # --- 8. produnsat ---
    elif circuit == 'produnsat':
        disc = (k2 + A + B) ** 2 - 4.0 * A * B
        sq = math.sqrt(disc) if disc > 0.0 else 0.0
        r_prodC = (k1 / 2.0) * (k2 + A + B - sq)
        Rtot = prodA + prodB + r_prodC
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + prodB:
            B = B + 1
        else:
            A = A - 1
            C = C + 1

    # --- 9. prod_fixedB ---
    elif circuit == 'prod_fixedB':
        A = prodA * Tcc * V
        disc = (k2 + A + B) ** 2 - 4.0 * A * B
        sq = math.sqrt(disc) if disc > 0.0 else 0.0
        r_prodC = (k1 / 2.0) * (k2 + A + B - sq)
        Rtot = prodB + r_prodC
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodB:
            B = B + 1
        else:
            A = A - 1
            C = C + 1
        A = prodA * Tcc * V

    # --- 10. cascade ---
    elif circuit == 'cascade':
        r_prodB = k1 * A
        r_prodC = k2 * B
        Rtot = prodA + r_prodB + r_prodC
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + r_prodB:
            B = B + 1
        else:
            C = C + 1

    # --- 11. proddeg ---
    elif circuit == 'proddeg':
        r_prodC = k1 * A
        disc = (k3 + B + C) ** 2 - 4.0 * B * C
        sq = math.sqrt(disc) if disc > 0.0 else 0.0
        degC = (k2 / 2.0) * (k3 + B + C - sq)
        Rtot = prodA + prodB + r_prodC + degC
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + prodB:
            B = B + 1
        elif r2 < prodA + prodB + r_prodC:
            C = C + 1
        else:
            C = C - 1

    # --- 12. phos ---
    elif circuit == 'phos':
        r_prodC = prodB
        r_prodB = k1 * A
        r_prodD = k2 * B * C
        Rtot = prodA + r_prodB + r_prodC + r_prodD
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + r_prodB:
            A = A - 1
            B = B + 1
        elif r2 < prodA + r_prodB + r_prodC:
            C = C + 1
        else:
            A = A + 1
            B = B - 1
            C = C - 1
            D = D + 1

    # --- 13. phos_sat ---
    elif circuit == 'phos_sat':
        Bp = D
        disc_c = (k4 + C + Bp) ** 2 - 4.0 * C * Bp
        sq_c = math.sqrt(disc_c) if disc_c > 0.0 else 0.0
        formB = (k3 / 2.0) * (k4 + C + Bp - sq_c)

        disc_a = (k2 + A + B) ** 2 - 4.0 * A * B
        sq_a = math.sqrt(disc_a) if disc_a > 0.0 else 0.0
        formBp = (k1 / 2.0) * (k2 + A + B - sq_a)

        Rtot = prodA + prodB + formB + formBp + prodC
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + prodB:
            B = B + 1
        elif r2 < prodA + prodB + formB:
            B = B + 1
            D = D - 1
        elif r2 < prodA + prodB + formB + formBp:
            B = B - 1
            D = D + 1
        else:
            C = C + 1

    # --- 14. phos_int ---
    elif circuit == 'phos_int':
        Ap = B
        B_spec = C
        ApB = D
        prodAp = k1 * A
        prodApB = k2 * Ap * B_spec
        prodBp = k3 * ApB
        Rtot = prodA + prodAp + prodB + prodApB + prodBp
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + prodAp:
            A = A - 1
            B = B + 1
        elif r2 < prodA + prodAp + prodB:
            C = C + 1
        elif r2 < prodA + prodAp + prodB + prodApB:
            B = B - 1
            C = C - 1
            D = D + 1
        else:
            D = D - 1
            E = E + 1
            A = A + 1

    # --- 15. phos2 ---
    elif circuit == 'phos2':
        A_unphos = A
        Ap = B
        B_unphos = C
        Bp = D
        ApB = E
        ABp = F
        kk = k1
        krevk = k2
        k_bind1 = k3
        krev1 = k4
        kt = k5
        k_bind2 = k6
        krev2 = k7
        kp = k8

        a_phos = kk * A_unphos
        a_dephos = krevk * Ap
        apb_bind = k_bind1 * Ap * B_unphos
        apb_unbind = krev1 * ApB
        tphos = kt * ApB
        abp_bind = k_bind2 * A_unphos * Bp
        abp_unbind = krev2 * ABp
        dephos = kp * ABp

        Rtot = prodA + prodB + a_phos + a_dephos + apb_bind + apb_unbind + tphos + abp_bind + abp_unbind + dephos
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + prodB:
            C = C + 1
        elif r2 < prodA + prodB + a_phos:
            A = A - 1
            B = B + 1
        elif r2 < prodA + prodB + a_phos + a_dephos:
            A = A + 1
            B = B - 1
        elif r2 < prodA + prodB + a_phos + a_dephos + apb_bind:
            B = B - 1
            C = C - 1
            E = E + 1
        elif r2 < prodA + prodB + a_phos + a_dephos + apb_bind + apb_unbind:
            B = B + 1
            C = C + 1
            E = E - 1
        elif r2 < prodA + prodB + a_phos + a_dephos + apb_bind + apb_unbind + tphos:
            E = E - 1
            A = A + 1
            D = D + 1
        elif r2 < prodA + prodB + a_phos + a_dephos + apb_bind + apb_unbind + tphos + abp_bind:
            A = A - 1
            D = D - 1
            F = F + 1
        elif r2 < prodA + prodB + a_phos + a_dephos + apb_bind + apb_unbind + tphos + abp_bind + abp_unbind:
            A = A + 1
            D = D + 1
            F = F - 1
        else:
            A = A + 1
            C = C + 1
            F = F - 1

    # --- 16. phos_cycle ---
    elif circuit == 'phos_cycle':
        Bp = D
        formBp = k1 * A * B
        formB = k3 * C * Bp
        Rtot = prodA + prodB + formB + formBp + prodC
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + prodB:
            B = B + 1
        elif r2 < prodA + prodB + formB:
            B = B + 1
            D = D - 1
        elif r2 < prodA + prodB + formB + formBp:
            B = B - 1
            D = D + 1
        else:
            C = C + 1

    # --- 17. diffTF ---
    elif circuit == 'diffTF':
        r_prodC = k1 * A * B
        r_prodD = k2 * C
        Rtot = prodA + prodB + r_prodC + r_prodD
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + prodB:
            B = B + 1
        elif r2 < prodA + prodB + r_prodC:
            A = A - 1
            B = B - 1
            C = C + 1
        else:
            D = D + 1

    # --- 18. cdg ---
    elif circuit == 'cdg':
        r_prodC = k1 * A
        disc = (k3 + B + C) ** 2 - 4.0 * B * C
        sq = math.sqrt(disc) if disc > 0.0 else 0.0
        degC = (k2 / 2.0) * (k3 + B + C - sq)
        r_prodD = k4 * C
        Rtot = prodA + prodB + r_prodC + degC + r_prodD
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + prodB:
            B = B + 1
        elif r2 < prodA + prodB + r_prodC:
            C = C + 1
        elif r2 < prodA + prodB + r_prodC + degC:
            C = C - 1
        else:
            D = D + 1

    # --- 19. cascade6 ---
    elif circuit == 'cascade6':
        r_prodB = k1 * A
        r_prodC = k2 * B
        r_prodD = k3 * C
        r_prodE = k4 * D
        r_prodF = k5 * E
        Rtot = prodA + r_prodB + r_prodC + r_prodD + r_prodE + r_prodF
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + r_prodB:
            B = B + 1
        elif r2 < prodA + r_prodB + r_prodC:
            C = C + 1
        elif r2 < prodA + r_prodB + r_prodC + r_prodD:
            D = D + 1
        elif r2 < prodA + r_prodB + r_prodC + r_prodD + r_prodE:
            E = E + 1
        else:
            F = F + 1

    # --- 20. crazyCascade ---
    elif circuit == 'crazyCascade':
        r_prodB = k1 * A
        r_prodC = prodB
        disc_b = (k3 + B + C) ** 2 - 4.0 * B * C
        sq_b = math.sqrt(disc_b) if disc_b > 0.0 else 0.0
        degB = (k2 / 2.0) * (k3 + B + C - sq_b)
        r_prodD = k4 * B
        r_prodE = prodC
        disc_d = (k6 + D + E) ** 2 - 4.0 * D * E
        sq_d = math.sqrt(disc_d) if disc_d > 0.0 else 0.0
        degD = (k5 / 2.0) * (k6 + D + E - sq_d)
        r_prodF = k7 * D
        Rtot = prodA + r_prodB + r_prodC + degB + r_prodD + r_prodE + degD + r_prodF
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + r_prodB:
            B = B + 1
        elif r2 < prodA + r_prodB + r_prodC:
            C = C + 1
        elif r2 < prodA + r_prodB + r_prodC + degB:
            B = B - 1
        elif r2 < prodA + r_prodB + r_prodC + degB + r_prodD:
            D = D + 1
        elif r2 < prodA + r_prodB + r_prodC + degB + r_prodD + r_prodE:
            E = E + 1
        elif r2 < prodA + r_prodB + r_prodC + degB + r_prodD + r_prodE + degD:
            D = D - 1
        else:
            F = F + 1

    # --- 21. grid_relay (gridfunc.py else branch; never reached by any script) ---
    # elif circuit == 'grid_relay':
    #     r_prodB = k1 * A
    #     r_prodD = k2 * B * C
    #     r_prodE = k3 * D
    #     Rtot = prodA + r_prodB + prodC + r_prodD + r_prodE
    #     r1 = rng.random()
    #     r2 = rng.random() * Rtot
    #     tau = -math.log(r1) / Rtot
    #     if r2 < prodA:
    #         A = A + 1
    #     elif r2 < prodA + r_prodB:
    #         A = A - 1
    #         B = B + 1
    #     elif r2 < prodA + r_prodB + prodC:
    #         C = C + 1
    #     elif r2 < prodA + r_prodB + prodC + r_prodD:
    #         A = A + 1
    #         B = B - 1
    #         C = C - 1
    #         D = D + 1
    #     else:
    #         E = E + 1

    # --- 22. tcs (motiffunc.py else branch) ---
    else:
        Ap = B
        B_spec = C
        ApB = D
        Bp = E
        prodAp = k1 * A
        prodApB = k2 * Ap * B_spec
        prodBp = k3 * ApB
        prodP = k4 * Bp
        Rtot = prodA + prodAp + prodB + prodApB + prodBp + prodP
        r1 = rng.random()
        r2 = rng.random() * Rtot
        tau = -math.log(r1) / Rtot
        if r2 < prodA:
            A = A + 1
        elif r2 < prodA + prodAp:
            A = A - 1
            B = B + 1
        elif r2 < prodA + prodAp + prodB:
            C = C + 1
        elif r2 < prodA + prodAp + prodB + prodApB:
            B = B - 1
            C = C - 1
            D = D + 1
        elif r2 < prodA + prodAp + prodB + prodApB + prodBp:
            D = D - 1
            E = E + 1
            A = A + 1
        else:
            F = F + 1

    return A, B, C, D, E, F, tau


@nb.njit(fastmath=True, nogil=True)
def run_cycle_numba(
    circuit, A0, B0, C0, D0, E0, F0, V0, t0, growthRate, params, rng,
    t_arr, V_arr, A_arr, B_arr, C_arr, D_arr, E_arr, F_arr, max_steps
):
    """
    Run a full Gillespie cell cycle until V >= 2.0 or max_steps reached.
    Populates pre-allocated buffer arrays and returns (n, t, V, A, B, C, D, E, F, overflow).
    """
    A = A0
    B = B0
    C = C0
    D = D0
    E = E0
    F = F0
    V = V0
    t = t0
    n = 1

    t_arr[0] = t
    V_arr[0] = V
    A_arr[0] = A
    B_arr[0] = B
    C_arr[0] = C
    D_arr[0] = D
    E_arr[0] = E
    F_arr[0] = F

    overflow = False

    while V < 2.0:
        if n >= max_steps:
            overflow = True
            break

        A, B, C, D, E, F, tau = step_reaction(circuit, A, B, C, D, E, F, V, params, rng)
        V += tau * growthRate
        t += tau

        t_arr[n] = t
        V_arr[n] = V
        A_arr[n] = A
        B_arr[n] = B
        C_arr[n] = C
        D_arr[n] = D
        E_arr[n] = E
        F_arr[n] = F
        n += 1

    return n, t, V, A, B, C, D, E, F, overflow


def pack_params(cell):
    """Convert Cell instance parameters into float64 array for Numba."""
    return np.array([
        float(getattr(cell, 'prodA', 0.0)),
        float(getattr(cell, 'prodB', 0.0)),
        float(getattr(cell, 'prodC', 0.0)),
        float(getattr(cell, 'k1', 0.0)),
        float(getattr(cell, 'k2', 0.0)),
        float(getattr(cell, 'k3', 0.0)),
        float(getattr(cell, 'k4', 0.0)),
        float(getattr(cell, 'k5', 0.0)),
        float(getattr(cell, 'k6', 0.0)),
        float(getattr(cell, 'k7', 0.0)),
        float(getattr(cell, 'k8', 0.0)),
        float(getattr(cell, 'burstSize', 1.0)),
        float(getattr(cell, 'Tcc', 0.0)),
        0.0, 0.0, 0.0
    ], dtype=np.float64)


def to_scalar(x):
    """Extract scalar float from float, int, or numpy array."""
    if hasattr(x, '__len__'):
        return float(x[-1])
    return float(x)

