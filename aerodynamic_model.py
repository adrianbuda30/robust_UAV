import casadi as ca
import numpy as np
from math import pi


# Helper functions

def ca_const(x):
    """Convert Python/NumPy numeric to CasADi DM if needed pass through SX/MX."""
    if isinstance(x, (ca.SX, ca.MX, ca.DM)):
        return x
    return ca.DM(x)

def simpson_weights(n, x0, x1):
    """Classic Simpson weights on [x0, x1] for n samples (n must be odd)."""
    assert n % 2 == 1, "Simpson requires an odd number of points"
    w = np.ones(n)
    w[1:-1:2] = 4
    w[2:-1:2] = 2
    h = (x1 - x0) / (n - 1)
    return ca.DM(w * h / 3.0)


def simpson_ca(f_vals, w):
    """Weighted Simpson integral: sum(w .* f). f_vals, w are (n,) vectors."""
    return ca.sum1(ca_const(f_vals) * ca_const(w))


# Integration method across the span that is currently being used
def gauss_nodes_weights(n, a, b):
    """n-point Gauss-Legendre on [a,b]. Returns (nodes DM(n,1), weights DM(n,1))."""
    xk, wk = np.polynomial.legendre.leggauss(n)
    xm = 0.5 * (b + a)
    xr = 0.5 * (b - a)
    return ca.DM(xm + xr * xk), ca.DM(xr * wk)


def softabs_ca(x, eps=1e-6):
    x = ca_const(x)
    return ca.sqrt(x * x + (eps * eps))



def R_sb(angle_of_attack, sideslip_angle, vector):
    ca_cos = ca.cos
    ca_sin = ca.sin
    alpha = ca_const(angle_of_attack)
    beta = ca_const(sideslip_angle)
    v = ca.reshape(ca_const(vector), (3, 1))

    cA, sA = ca_cos(alpha), ca_sin(alpha)
    cB, sB = ca_cos(beta), ca_sin(beta)

    R = ca.vertcat(
        ca.hcat([cA * cB, -cA * sB, -sA]),
        ca.hcat([sB, cB, 0]),
        ca.hcat([sA * cB, -sA * sB, cA]),
    )
    return ca.mtimes(R, v).full().ravel() if isinstance(vector, np.ndarray) else ca.mtimes(R, v)


def R_bs(angle_of_attack, sideslip_angle, vector):
    # transpose of R_sb
    ca_cos = ca.cos
    ca_sin = ca.sin
    alpha = ca_const(angle_of_attack)
    beta = ca_const(sideslip_angle)
    v = ca.reshape(ca_const(vector), (3, 1))

    cA, sA = ca_cos(alpha), ca_sin(alpha)
    cB, sB = ca_cos(beta), ca_sin(beta)

    R = ca.vertcat(
        ca.hcat([cA * cB, -cA * sB, -sA]),
        ca.hcat([sB, cB, 0]),
        ca.hcat([sA * cB, -sA * sB, cA]),
    )
    Rt = R.T
    return ca.mtimes(Rt, v).full().ravel() if isinstance(vector, np.ndarray) else ca.mtimes(Rt, v)



def flat_plate_coeffs(AR_query: float):
    AR_tab = np.array([0.167, 0.333, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 3.0, 4.0, 6.0])

    aLE_tab = np.array([3, 3.64, 4.48, 7.18, 10.2, 13.38, 14.84, 14.49, 9.95, 12.93, 15, 15])
    aTE_tab = np.array([5.9, 15.51, 32.57, 39.44, 48.22, 59.29, 21.55, 7.74, 7.05, 5.26, 6.5, 6.5])

    alphaLE_tab = np.array([59, 58.6, 58.2, 50, 41.53, 26.7, 23.44, 21, 18.63, 14.28, 11.6, 10])
    alphaTE_tab = np.array([59, 58.6, 58.2, 51.85, 41.46, 28.09, 39.4, 35.86, 26.76, 19.76, 16.43, 14])
    alpha_high_S_tab = np.array([49, 54, 56, 48, 40, 29, 27, 25, 24, 22, 22, 20])

    ARq = float(np.clip(float(AR_query), AR_tab[0], AR_tab[-1]))
    aLE = float(np.interp(ARq, AR_tab, aLE_tab))
    aTE = float(np.interp(ARq, AR_tab, aTE_tab))
    alpha_LE = float(np.deg2rad(np.interp(ARq, AR_tab, alphaLE_tab)))
    alpha_TE = float(np.deg2rad(np.interp(ARq, AR_tab, alphaTE_tab)))
    alpha_high_S = float(np.deg2rad(np.interp(ARq, AR_tab, alpha_high_S_tab)))

    return dict(aLE=aLE, aTE=aTE, alpha_LE=alpha_LE, alpha_TE=alpha_TE, alpha_high_S=alpha_high_S)


def flatPlate_model(alpha, Cl_alpha, alpha_0, alphadot, AR, c, cf, V, delta_f=0.0):
    """
    CasADi-friendly version. All scalar inputs may be floats or CasADi SX/MX.
    """
    ca_sin = ca.sin
    ca_cos = ca.cos
    ca_tanh = ca.tanh
    ca_abs = ca.fabs

    pars = flat_plate_coeffs(float(AR))  # constants (ok to compute with NumPy)
    aLE = pars['aLE']
    aTE = pars['aTE']
    alpha_LE = pars['alpha_LE']
    alpha_TE = pars['alpha_TE']
    alpha_high_S = pars['alpha_high_S']

    c = ca_const(c)
    cf = ca_const(cf)
    V = ca.if_else(V == 0, ca.DM(1e-6), ca_const(V))
    Cl_alpha = ca_const(Cl_alpha)
    alpha = ca_const(alpha)
    alpha_0 = ca_const(alpha_0)
    alphadot = ca_const(alphadot)
    delta_f = ca_const(delta_f)

    cf_c = cf / c

    # 3D slope (your formula)
    CL_alpha = 2 * pi * (AR / (AR + 2 * (AR + 4) / (AR + 2)))

    # flap/camber influence
    eta_f = ca.DM(1.0)
    theta_f = ca.acos(2 * cf_c - 1)
    tau_f = 1 - (theta_f - ca_sin(theta_f)) / pi
    delta_CL = CL_alpha * tau_f * eta_f * delta_f

    tauTE = 4.5 * c / V
    tauLE = 0.5 * c / V

    alpha_lift = alpha - alpha_0

    fTE = 0.5 * (1 - ca_tanh(aTE * (softabs_ca(alpha_lift) - tauTE * alphadot - alpha_TE)))
    fLE = 0.5 * (1 - ca_tanh(aLE * (softabs_ca(alpha_lift) - tauLE * alphadot - alpha_LE)))

    Kp = CL_alpha
    Kv = pi

    alpha_eff = alpha_lift

    # Branch 1 (|alpha| < alpha_high_S): attached model
    CL_att = 0.25 * (1 + ca.sqrt(fTE)) ** 2 * (
            Kp * ca_sin(alpha_eff) * (ca_cos(alpha_eff) ** 2)
            + (fLE ** 2) * Kv * ca_abs(ca_sin(alpha_eff)) * ca_sin(alpha_eff) * ca_cos(alpha_eff)
    ) + delta_CL  # effects of flap directly added to lift
    CDi_att = CL_att * ca.tan(alpha_eff)
    CM0_att = -0.25 * (1 + ca.sqrt(fTE)) ** 2 * (
            0.0625 * (-1 + 6 * ca.sqrt(fTE) - 5 * fTE) * Kp * ca_sin(alpha_eff) * ca_cos(alpha_eff)
            + 0.17 * (fLE ** 2) * Kv * ca_abs(ca_sin(alpha_eff)) * ca_sin(alpha_eff)
    )

    # Branch 2 (|alpha| >= alpha_high_S): post-stall
    cprime = ca.sqrt((c - cf) ** 2 + cf ** 2 + 2 * cf * (c - cf) * ca_cos(delta_f))
    alpha_f = ca.asin(ca.if_else(cprime == 0, ca.DM(0.0), cf / cprime * ca_sin(delta_f)))
    alpha_ps = alpha - alpha_0 + alpha_f

    Cd_90 = 1.98 + (-4.26e-2) * (delta_f ** 2) + 2.1e-1 * delta_f
    Cd0_flat = 0.03
    s = ca_sin(alpha_ps)
    c_ = ca_cos(alpha_ps)
    sa = ca_abs(s)
    G = 1.0 / (0.56 + 0.44 * sa) - 0.41 * (1.0 - ca.exp(-17.0 / AR))
    Cn = Cd_90 * s * G  # odd
    Ca = 0.5 * Cd0_flat * c_  # even
    CL_ps = Cn * c_ - Ca * s
    CDi_ps = Cn * s + Ca * c_
    CM0_ps = -Cn * (0.25 - 0.175 * (1.0 - 2.0 * alpha_ps / pi))


    # Use blending model instead of if-else statement for faster computation
    k = 20.0  # sharpness factor
    sigma = 0.5 * (1 + ca.tanh(k * (alpha_high_S - ca.fabs(alpha))))
    CL = sigma * CL_att + (1 - sigma) * CL_ps
    CDi = sigma * CDi_att + (1 - sigma) * CDi_ps
    CM0 = sigma * CM0_att + (1 - sigma) * CM0_ps

    return CL, CDi, CM0


# ---------- vertical surface ----------

def calculate_force_vertical(U, V, W, p, r, N, cg, data, wing_data, delta_f, alpha_dot=0.0, rho=1.225):

    b = data.get('b', 5.0)
    c_root = data.get('c_root', 1.0)
    c_tip = data.get('c_tip', 0.5)
    p_root = np.array(data.get('p_root', np.array([0, 0, 0])), dtype=float)
    AR = 4 * b / (c_root + c_tip)
    CD0 = data.get('CD0', 0.02)
    Cl_alpha = data.get('Cl_alpha', 2 * pi)
    sweep_q = data.get('sweep_quater', 0.0)
    cf_c = data.get('cf_c', 0.3)
    alpha_0 = data.get('alpha_0', 0.0)

    p_root_quater = p_root + np.array([c_root * 0.25, 0, 0], dtype=float)

    # 5-point Gauss-Legendre scheme
    z0, z1 = float(p_root[2]), float(p_root[2] + b)
    z_nodes, wS = gauss_nodes_weights(N, z0, z1)
    z = np.array(z_nodes.full()).ravel()

    secY = []
    secX = []
    secNm = []
    secLm = []

    Uc, Vc, Wc = ca_const(U), ca_const(V), ca_const(W)
    pc, rc = ca_const(p), ca_const(r)
    cgc = ca_const(cg).reshape((3, 1))
    rho_c = ca_const(rho)
    sweep_q_c = ca_const(sweep_q)

    for zi in z:
        # chord (with sweep correction)
        chord = (c_root + (c_tip - c_root) * (abs(zi) / b)) / np.cos(sweep_q)
        chord_c = ca.DM(chord)

        p_section = p_root_quater + np.array([zi * np.sin(sweep_q), 0, zi], dtype=float)
        moment_arm = ca_const(p_section).reshape((3, 1)) - cgc

        # local velocities (body)
        V_rot = ca.cross(ca.vertcat(pc, ca.DM(0), rc), moment_arm)
        u_sec = Uc
        v_sec = Vc + moment_arm[2] * pc - rc * moment_arm[0]
        w_sec = Wc

        Ve2 = u_sec * u_sec + v_sec * v_sec + w_sec * w_sec
        Ve = ca.sqrt(Ve2 + 1e-16)
        alpha = ca.atan2(w_sec, u_sec)
        arg = v_sec / Ve
        arg = ca.fmin(1.0, ca.fmax(-1.0, arg))
        beta = ca.asin(arg)

        VN = Ve * ca.sin(beta)
        VC = Ve * (ca.cos(alpha) * ca.cos(beta) * ca.cos(sweep_q_c) +
              ca.sin(alpha) * ca.cos(beta) * ca.sin(sweep_q_c))
        V_inf = ca.sqrt(VN * VN + VC * VC + 1e-16)

        alpha_h = ca.atan2(VN, VC)

        CL, CDi, CM0 = flatPlate_model(alpha_h, Cl_alpha, alpha_0, alpha_dot, AR,
                                       chord_c, cf_c * chord_c, V_inf, delta_f)
        CD = ca.DM(CD0) + CDi

        qS = 0.5 * rho_c * (V_inf * V_inf) * chord_c
        L = qS * CL
        D = qS * CD
        M = 0.5 * rho_c * (V_inf * V_inf) * (chord_c * chord_c) * CM0

        F_body = R_sb(alpha_h, 0, ca.vertcat(D, 0, L))  # 3x1
        M_body = R_sb(alpha_h, 0, ca.vertcat(0, M, 0))  # 3x1

        Y_i = F_body[2]
        X_i = -F_body[0] * ca.cos(sweep_q_c)
        N_i = Y_i * moment_arm[0] + M_body[1]
        L_i = -Y_i * moment_arm[2]

        secY.append(Y_i)
        secX.append(X_i)
        secNm.append(N_i)
        secLm.append(L_i)

    secY = ca.vertcat(*secY)
    secX = ca.vertcat(*secX)
    secNm = ca.vertcat(*secNm)
    secLm = ca.vertcat(*secLm)

    # integrate
    Y_total = ca.dot(wS, secY)
    X_total = ca.dot(wS, secX)
    N_total = ca.dot(wS, secNm)
    L_total = ca.dot(wS, secLm)

    # coefficients
    Sref = wing_data.get('Sref', 1.0)
    bref = wing_data.get('bref', 1.0)

    Ve = ca.sqrt(ca_const(U) ** 2 + ca_const(V) ** 2 + ca_const(W) ** 2 + 1e-16)
    denomF = 0.5 * rho_c * (Ve * Ve) * Sref
    denomM_b = denomF * bref

    CY = Y_total / denomF
    CX = X_total / denomF
    CL = L_total / denomM_b
    CN = N_total / denomM_b

    return CY, CX, CL, CN, None



def calculate_force_horizontal(U, V, W, p, r, q, cg, N, data, wing_data, delta_f, side,
                               alpha_dot=0.0, delta_sweep=0.0, rho=1.225):
    b = data.get('b', 0.46)
    c_root = data.get('c_root', 0.278)
    c_tip = data.get('c_tip', 0.083)
    AR = 4 * b / (c_root + c_tip)
    CD0 = data.get('CD0', 0.02)
    Cl_alpha = data.get('Cl_alpha', 2 * pi)
    sweep_q = data.get('sweep_quater', 0.0)
    Gamma = data.get('Gamma', 0.0)
    alpha_0 = data.get('alpha_0', 0.0)
    cf_c = data.get('cf_c', 0.5)
    p_root = np.array(data.get('p_root', np.array([0, 0, 0])), dtype=float)
    p_root_q = p_root + np.array([c_root * 0.25, 0, 0], dtype=float)

    if side.lower().startswith('l'):
        y0, y1 = p_root[1] - b, p_root[1]
    else:
        y0, y1 = p_root[1], p_root[1] + b

    y_nodes, wS = gauss_nodes_weights(N, y0, y1)
    y = np.array(y_nodes.full()).ravel()

    secX = []
    secZ = []
    secLm = []
    secMm = []
    secNm = []

    Uc, Vc, Wc = ca_const(U), ca_const(V), ca_const(W)
    pc, rc, qc = ca_const(p), ca_const(r), ca_const(q)
    cgc = ca_const(cg).reshape((3, 1))
    rho_c = ca_const(rho)
    sweep_q_c = ca_const(sweep_q)
    Gamma_c = ca_const(Gamma)

    cosa = ca.cos(delta_sweep)
    sina = ca.sin(delta_sweep)
    Rz = ca.vertcat(
        ca.hcat([cosa, -sina, 0]),
        ca.hcat([sina, cosa, 0]),
        ca.hcat([0, 0, 1])
    )

    for yi in y:
        chord = (c_root + (c_tip - c_root) * (abs(yi) / b)) / np.cos(sweep_q)
        chord_c = ca.DM(chord)

        p_section = p_root_q + np.array([yi * np.sin(sweep_q), yi, yi * np.sin(sweep_q)], dtype=float)
        p_sec_ca = ca_const(p_root_q).reshape((3, 1)) + ca.mtimes(Rz, (
                    ca_const(p_section).reshape((3, 1)) - ca_const(p_root_q).reshape((3, 1))))
        moment_arm = p_sec_ca - cgc

        # local velocities
        u_sec = Uc + moment_arm[1] * rc
        v_sec = Vc
        w_sec = Wc + pc * moment_arm[1] + qc * moment_arm[0]

        Ve2 = u_sec * u_sec + v_sec * v_sec + w_sec * w_sec
        Ve = ca.sqrt(Ve2 + 1e-16)
        alpha = ca.atan2(w_sec, u_sec)
        # beta  = ca.asin(v_sec / Ve)

        arg = v_sec / Ve
        arg = ca.fmin(1.0, ca.fmax(-1.0, arg))
        beta = ca.asin(arg)

        VN = Ve * (ca.sin(beta) * ca.sin(Gamma_c) + ca.sin(alpha) * ca.cos(beta) * ca.cos(Gamma_c))
        VC = Ve * (ca.cos(alpha) * ca.cos(beta) * ca.cos(sweep_q_c)
                   - ca.sin(sweep_q_c) * ca.cos(Gamma_c) * ca.sin(beta)
                   + ca.sin(alpha) * ca.cos(beta) * ca.sin(Gamma_c))
        V_inf = ca.sqrt(VN * VN + VC * VC + 1e-16)
        alpha_h = ca.atan2(VN, VC)

        CL, CDi, CM0 = flatPlate_model(alpha_h, Cl_alpha, alpha_0, alpha_dot, AR,
                                       chord_c, cf_c * chord_c, V_inf, delta_f)
        CD = ca.DM(CD0) + CDi

        qS = 0.5 * rho_c * (V_inf * V_inf) * chord_c
        L = qS * CL
        D = qS * CD
        M = 0.5 * rho_c * (V_inf * V_inf) * (chord_c * chord_c) * CM0

        F_body = R_sb(alpha_h, 0, ca.vertcat(D, 0, L))
        M_body = R_sb(alpha_h, 0, ca.vertcat(0, M, 0))

        X_i = -F_body[0] * ca.cos(sweep_q_c)
        Z_i = -F_body[2]
        L_i = Z_i * moment_arm[1]
        M_i = Z_i * moment_arm[0] + M_body[1]
        N_i = X_i * moment_arm[1]

        secX.append(X_i)
        secZ.append(Z_i)
        secLm.append(L_i)
        secMm.append(M_i)
        secNm.append(N_i)

    # after the loop
    secX = ca.vertcat(*secX)
    secZ = ca.vertcat(*secZ)
    secLm = ca.vertcat(*secLm)
    secMm = ca.vertcat(*secMm)
    secNm = ca.vertcat(*secNm)

    # integrate (no star here)
    X_total = ca.dot(wS, secX)
    Z_total = ca.dot(wS, secZ)
    L_total = ca.dot(wS, secLm)
    M_total = ca.dot(wS, secMm)
    N_total = ca.dot(wS, secNm)

    Sref = wing_data.get('Sref', 1.0)
    bref = wing_data.get('bref', 1.0)
    cref = wing_data.get('cref', 1.0)

    Ve = ca.sqrt(ca_const(U) ** 2 + ca_const(V) ** 2 + ca_const(W) ** 2 + 1e-16)
    denomF = 0.5 * rho_c * (Ve * Ve) * Sref
    denomMb = denomF * bref
    denomMc = denomF * cref

    CX = X_total / denomF
    CZ = Z_total / denomF
    CL = L_total / denomMb
    CM = M_total / denomMc
    CN = N_total / denomMb

    return CX, CZ, CL, CM, CN, None


def compute_totals_coeffs_ca(U, V, W, p, q, r,
                             delta_a, delta_e, delta_r,
                             my_plane, wing_data, cg, N, rho=1.225,
                             aileron_left_sign=-1.0, aileron_right_sign=+1.0):

    seg_vt = my_plane.get_segment('vt')
    seg_ht_left = my_plane.get_segment('left_ht')
    seg_ht_right = my_plane.get_segment('right_ht')
    seg_wing_left = my_plane.get_segment('left_wing')
    seg_wing_right = my_plane.get_segment('right_wing')

    # Vertical tail (rudder)
    CY_vt, CX_vt, CL_vt, CN_vt, _ = calculate_force_vertical(
        U, V, W, p, r, N, cg, seg_vt, wing_data, delta_r, 0.0, rho
    )

    # Horizontal tail (elevator)
    CX_ht_left, CZ_ht_left, CL_ht_left, CM_ht_left, CN_ht_left, _ = calculate_force_horizontal(
        U, V, W, p, r, q, cg, N, seg_ht_left, wing_data, delta_e, 'left', 0.0, 0.0, rho
    )
    CX_ht_right, CZ_ht_right, CL_ht_right, CM_ht_right, CN_ht_right, _ = calculate_force_horizontal(
        U, V, W, p, r, q, cg, N, seg_ht_right, wing_data, delta_e, 'right', 0.0, 0.0, rho
    )

    # Wings (ailerons)
    da_left = aileron_left_sign * delta_a
    da_right = aileron_right_sign * delta_a

    CX_wing_left, CZ_wing_left, CL_wing_left, CM_wing_left, CN_wing_left, _ = calculate_force_horizontal(
        U, V, W, p, r, q, cg, N, seg_wing_left, wing_data, da_left, 'left', 0.0, 0.0, rho
    )
    CX_wing_right, CZ_wing_right, CL_wing_right, CM_wing_right, CN_wing_right, _ = calculate_force_horizontal(
        U, V, W, p, r, q, cg, N, seg_wing_right, wing_data, da_right, 'right', 0.0, 0.0, rho
    )

    # Totals
    CX_total = CX_vt + CX_ht_left + CX_ht_right + CX_wing_left + CX_wing_right
    CY_total = CY_vt
    CZ_total = CZ_ht_left + CZ_ht_right + CZ_wing_left + CZ_wing_right
    CL_total = CL_vt + CL_ht_left + CL_ht_right + CL_wing_left + CL_wing_right
    CM_total = CM_ht_left + CM_ht_right + CM_wing_left + CM_wing_right
    CN_total = CN_vt + CN_ht_left + CN_ht_right + CN_wing_left + CN_wing_right

    return ca.vertcat(CX_total, CY_total, CZ_total, CL_total, CM_total, CN_total)



def build_coeffs_function(my_plane, wing_data, rho=1.225, nquad=3):
    """
    Builds a CasADi symbolic function to compute the total aerodynamic coefficients for a given aircraft configuration.
    This function defines symbolic variables for the aircraft's state and control inputs, computes the aerodynamic coefficients
    using the provided aircraft and wing data, and returns a CasADi function that evaluates these coefficients. The resulting
    function can be used for further symbolic manipulation or numerical evaluation in optimization and control applications.
    Args:
        my_plane: An object containing the aircraft parameters and geometry.
        wing_data: Data structure containing information about the aircraft's wings.
        rho (float, optional): Air density in kg/m^3. Defaults to 1.225.
        nquad (int, optional): Number of quadrature points for numerical integration. Defaults to 3.
    Returns:
        casadi.Function: A CasADi function that takes the state variables (U, V, W, p, q, r), control inputs (da, de, dr),
        and center of gravity vector (cg) as inputs, and outputs the computed aerodynamic coefficients.
    """
    # Define CasADi symbolic variables for the state and control inputs
    U, V, W, p, q, r, da, de, dr = [ca.SX.sym(s) for s in ('U', 'V', 'W', 'p', 'q', 'r', 'da', 'de', 'dr')]
    cg = ca.SX.sym('cg', 3)  # Center of gravity as a 3-element vector

    # Compute the aerodynamic coefficients symbolically
    coeffs = compute_totals_coeffs_ca(U, V, W, p, q, r, da, de, dr,
                                      my_plane, wing_data, cg, nquad, rho)

    # Create a CasADi function for the coefficients, with JIT compilation disabled
    F = ca.Function('coeffs6',
                    [U, V, W, p, q, r, da, de, dr, cg],
                    [coeffs],
                    {'jit': False})
    # Expand the function for efficiency (no compilation)
    F = F.expand()
    return F