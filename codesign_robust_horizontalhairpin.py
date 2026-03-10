import cma
import multiprocessing as mp
import numpy as np
import casadi as ca
import os

import pickle

from dynamics import compute_derivative
from build_plane import plane
from math import pi
from aerodynamic_model import build_coeffs_function

from scipy.linalg import solve_continuous_are, solve_discrete_are
from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat

import random

def compute_real_physics(my_plane, rho=250.0):

    total_m = 0
    total_moment = np.array([0.0, 0.0, 0.0])
    segment_names = ['left_wing', 'right_wing', 'left_ht', 'right_ht', 'vt']
    parts = []

    Ixx = Iyy = Izz = 0

    for name in segment_names:
        s = my_plane.get_segment(name)

        semispan = s['b']
        avg_chord = (s['c_root'] + s['c_tip']) / 2
        thickness = s['tc'] * avg_chord
        volume = semispan * avg_chord * thickness

        m = volume * rho
        x_centroid_local = avg_chord * 0.42

        if name == 'left_wing' or name == 'left_ht':
            cg_seg = s['p_root'] + np.array([x_centroid_local, -semispan / 2, 0])
        elif name == 'right_wing' or name == 'right_ht':
            cg_seg = s['p_root'] + np.array([x_centroid_local, semispan / 2, 0])
        else:
            cg_seg = s['p_root'] + np.array([x_centroid_local, 0, semispan / 2])

        total_m += m
        total_moment += m * cg_seg

        parts.append({'m': m, 'cg': cg_seg, 'b': semispan, 'c': avg_chord, 't': thickness, 'is_v': (name == 'vt')})

    R = 0.05
    L = 0.8
    m_fuse = (np.pi * R ** 2 * L) * rho
    cg_fuse = np.array([L / 2, 0, 0])

    total_m += m_fuse
    total_moment += m_fuse * cg_fuse

    # Final Aircraft CG
    cg_ac = total_moment / total_m

    ixx_f = 0.5 * m_fuse * R ** 2
    iyy_f = (1 / 12) * m_fuse * (3 * R ** 2 + L ** 2)
    izz_f = iyy_f

    ixx_f += m_fuse * (cg_fuse[0] - cg_ac[0]) ** 2
    iyy_f += m_fuse * (cg_fuse[1] - cg_ac[1]) ** 2
    izz_f += m_fuse * (cg_fuse[2] - cg_ac[2]) ** 2

    Ixx += ixx_f
    Iyy += iyy_f
    Izz += izz_f

    m_battery = 0.35
    total_m += m_battery

    for p in parts:

        dx, dy, dz = p['cg'] - cg_ac

        if not p['is_v']:
            ixx_l = (1 / 12) * p['m'] * (p['b'] ** 2 + p['t'] ** 2)
            iyy_l = (1 / 12) * p['m'] * (p['c'] ** 2 + p['t'] ** 2)
            izz_l = (1 / 12) * p['m'] * (p['b'] ** 2 + p['c'] ** 2)
        else:
            ixx_l = (1 / 12) * p['m'] * (p['t'] ** 2 + p['b'] ** 2)
            iyy_l = (1 / 12) * p['m'] * (p['c'] ** 2 + p['b'] ** 2)
            izz_l = (1 / 12) * p['m'] * (p['c'] ** 2 + p['t'] ** 2)

        Ixx += ixx_l + p['m'] * (dy ** 2 + dz ** 2)
        Iyy += iyy_l + p['m'] * (dx ** 2 + dz ** 2)
        Izz += izz_l + p['m'] * (dx ** 2 + dy ** 2)

    return total_m, cg_ac, Ixx, Iyy, Izz


def euler_to_quat(phi, theta, psi):
    cp, sp = np.cos(phi / 2), np.sin(phi / 2)
    ct, st = np.cos(theta / 2), np.sin(theta / 2)
    cy, sy = np.cos(psi / 2), np.sin(psi / 2)
    q0 = cy * ct * cp + sy * st * sp
    q1 = cy * ct * sp - sy * st * cp
    q2 = cy * st * cp + sy * ct * sp
    q3 = sy * ct * cp - cy * st * sp
    return q0, q1, q2, q3


def quaternion_to_rotation_matrix(q):

    if isinstance(q, (ca.MX, ca.SX)):
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        return ca.vertcat(
            ca.horzcat(1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)),
            ca.horzcat(2 * (q1 * q2 - q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 + q0 * q1)),
            ca.horzcat(2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2))
        )
    else:
        q0, q1, q2, q3 = q
        return np.array([
            [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
            [2 * (q1 * q2 - q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 + q0 * q1)],
            [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]
        ])

def inertial_to_body(vec, q):

    Rb = quaternion_to_rotation_matrix(q)
    if isinstance(vec, (ca.MX, ca.SX)):
        return ca.mtimes(Rb, vec)
    else:
        return Rb @ vec


def body_to_inertial(vec, q):

    Rb = quaternion_to_rotation_matrix(q)
    if isinstance(vec, (ca.MX, ca.SX)):
        return ca.mtimes(Rb.T, vec)
    else:
        return Rb.T @ vec


def get_linearization_functions(my_plane, wing_data, coeffs_fun):

    x_13_ref = ca.MX.sym('x_13', 13)
    u_ref = ca.MX.sym('u', 4)
    dx = ca.MX.sym('dx', 12)

    v_b = x_13_ref[0:3] + dx[0:3]
    w_b = x_13_ref[7:10] + dx[6:9]
    p_e = x_13_ref[10:13] + dx[9:12]

    dq = ca.vertcat(1.0, dx[3] / 2, dx[4] / 2, dx[5] / 2)
    q_ref = x_13_ref[3:7]

    def ca_quat_mult(q, p):
        return ca.vertcat(
            q[0] * p[0] - q[1] * p[1] - q[2] * p[2] - q[3] * p[3],
            q[0] * p[1] + q[1] * p[0] + q[2] * p[3] - q[3] * p[2],
            q[0] * p[2] - q[1] * p[3] + q[2] * p[0] + q[3] * p[1],
            q[0] * p[3] + q[1] * p[2] - q[2] * p[1] + q[3] * p[0]
        )

    q_perturbed = ca_quat_mult(q_ref, dq)
    x_perturbed = ca.vertcat(v_b, q_perturbed, w_b, p_e)

    ds_13 = compute_derivative(x_perturbed, u_ref, my_plane, wing_data, coeffs_fun)

    # Map 13D derivative back to 12D error derivative
    ds_12 = ca.vertcat(ds_13[0:3], w_b, ds_13[7:10], ds_13[10:13])

    A_jac = ca.jacobian(ds_12, dx)
    B_jac = ca.jacobian(ds_12, u_ref)

    A_fun = ca.Function('A_fun', [x_13_ref, u_ref], [ca.substitute(A_jac, dx, 0)])
    B_fun = ca.Function('B_fun', [x_13_ref, u_ref], [ca.substitute(B_jac, dx, 0)])

    return A_fun, B_fun

def compute_discrete_lqr_gain(Ac, Bc, Q, R, dt):

    nx = Ac.shape[0]
    Ad = np.eye(nx) + Ac * dt
    Bd = Bc * dt

    try:
        # Solve the Discrete Algebraic Riccati Equation (DARE)
        P = solve_discrete_are(Ad, Bd, Q, R)

        # Compute gain K
        K = np.linalg.inv(R + Bd.T @ P @ Bd) @ (Bd.T @ P @ Ad)
        return K
    except np.linalg.LinAlgError:
        return None


def quat_inv(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def quat_mult(q, p):
    w1, x1, y1, z1 = q
    w2, x2, y2, z2 = p
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def get_von_karman_disturbances(t, Va, seed, sigma_uvw=np.array([1.0, 1.0, 1.0]), L_uvw=np.array([10.0, 10.0, 10.0])):

    rng = np.random.default_rng(seed=seed)
    n_freqs = 40
    omegas = np.logspace(-1, 2, n_freqs)

    phases = rng.uniform(0, 2 * np.pi, (3, n_freqs))

    # Longitudinal PSD (u)
    def psd_u(w, sig, L):
        num = 2 * sig ** 2 * L
        den = np.pi * Va * (1 + (1.339 * L * w / Va) ** 2) ** (5 / 6)
        return num / den

    # Transverse PSD (v and w)
    def psd_trans(w, sig, L):
        num = sig ** 2 * L * (1 + 8 / 3 * (1.339 * L * w / Va) ** 2)
        den = np.pi * Va * (1 + (1.339 * L * w / Va) ** 2) ** (11 / 6)
        return num / den

    gusts = np.zeros(3)
    dw = omegas * 0.05

    for j in range(n_freqs):
        amp_u = np.sqrt(2 * psd_u(omegas[j], sigma_uvw[0], L_uvw[0]) * dw[j])
        gusts[0] += amp_u * np.cos(omegas[j] * t + phases[0, j])

        amp_v = np.sqrt(2 * psd_trans(omegas[j], sigma_uvw[1], L_uvw[1]) * dw[j])
        gusts[1] += amp_v * np.cos(omegas[j] * t + phases[1, j])

        amp_w = np.sqrt(2 * psd_trans(omegas[j], sigma_uvw[2], L_uvw[2]) * dw[j])
        gusts[2] += amp_w * np.cos(omegas[j] * t + phases[2, j])

    return gusts


def objective_function(payload):

    x, eval_id = payload
    semispan = x[0]
    chord = x[1]
    x_wing = x[2]

    print(f"Testing: Semispan={semispan:.4f}, Chord={chord:.4f}, X_pos={x_wing:.4f}")

    max_servo_speed = 8.0
    N = 120

    if not (0.05 <= semispan <= 0.5 and 0.05 <= chord <= 0.5 and 0.05 <= x_wing <= 0.5):
        return 200

    try:
        my_plane = plane()
        my_plane.add_segment('left_wing',
                             {'b': semispan, 'c_tip': chord, 'c_root': chord, 'p_root': np.array([x_wing, -0.05, 0.0]),
                              'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * pi,
                              'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})
        my_plane.add_segment('right_wing',
                             {'b': semispan, 'c_tip': chord, 'c_root': chord, 'p_root': np.array([x_wing, 0.05, 0.0]),
                              'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * pi,
                              'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})
        my_plane.add_segment('left_ht', {'b': 0.15, 'c_tip': 0.1, 'c_root': 0.1, 'p_root': np.array([0.7, -0.05, 0.0]),
                                         'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * pi,
                                         'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})
        my_plane.add_segment('right_ht', {'b': 0.15, 'c_tip': 0.1, 'c_root': 0.1, 'p_root': np.array([0.7, 0.05, 0.0]),
                                          'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * pi,
                                          'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})
        my_plane.add_segment('vt', {'b': 0.15, 'c_tip': 0.1, 'c_root': 0.1, 'p_root': np.array([0.7, 0.0, 0.05]),
                                    'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * pi,
                                    'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})

        real_m, real_cg, real_Ixx, real_Iyy, real_Izz = compute_real_physics(my_plane, rho=30.4)

        prop_eff = 0.65
        k_servo = 0.5

        my_plane.add_mass_properties({
            'mass': real_m,
            'cg': real_cg,
            'Ixx': real_Ixx, 'Iyy': real_Iyy, 'Izz': real_Izz,
            'Ixy': 0.0, 'Ixz': 0.0, 'Iyz': 0.0
        })

        wing_ref = my_plane.get_segment('left_wing')
        c_root_wing = wing_ref.get('c_root', None)
        c_tip_wing = wing_ref.get('c_tip', None)
        bref = wing_ref.get('b', None) * 2
        taper_ref = c_tip_wing / c_root_wing
        Sref = (c_root_wing + c_tip_wing) * bref / 2
        cref = (c_root_wing + c_tip_wing) / 2
        wing_data = {'Sref': Sref, 'bref': bref, 'cref': cref}

        rho = 1.225
        coeffs_fun = build_coeffs_function(my_plane, wing_data, rho=rho, nquad=5)

        opti = ca.Opti()
        time_final = opti.variable()

        opti.subject_to(time_final > 1)
        opti.set_initial(time_final, 10)

        dt = time_final / (N - 1)  # Time step

        # Create separate trajectory variables for each segment
        x_e = opti.variable(N)
        y_e = opti.variable(N)
        z_e = opti.variable(N)
        u_b = opti.variable(N)
        v_b = opti.variable(N)
        w_b = opti.variable(N)
        q0 = opti.variable(N)
        q1 = opti.variable(N)
        q2 = opti.variable(N)
        q3 = opti.variable(N)
        p = opti.variable(N)
        q_ang = opti.variable(N)
        r = opti.variable(N)

        thrust = opti.variable(N)
        delta_e = opti.variable(N)
        delta_a = opti.variable(N)
        delta_r = opti.variable(N)

        # Set initial conditions
        opti.subject_to(x_e[0] == 0)
        opti.subject_to(y_e[0] == 0)
        opti.subject_to(z_e[0] == 0)

        opti.subject_to(u_b[0] == 10)
        opti.subject_to(v_b[0] == 0)
        opti.subject_to(w_b[0] == 0)

        opti.subject_to(q0[0] == 1)
        opti.subject_to(q1[0] == 0)
        opti.subject_to(q2[0] == 0)
        opti.subject_to(q3[0] == 0)

        opti.subject_to(p[0] == 0)
        opti.subject_to(q_ang[0] == 0)
        opti.subject_to(r[0] == 0)

        opti.subject_to(delta_e[0] > -np.deg2rad(25))
        opti.subject_to(delta_a[0] > -np.deg2rad(25))
        opti.subject_to(delta_r[0] > -np.deg2rad(25))

        opti.subject_to(delta_e[0] < np.deg2rad(25))
        opti.subject_to(delta_a[0] < np.deg2rad(25))
        opti.subject_to(delta_r[0] < np.deg2rad(25))

        opti.subject_to(thrust[0] >= 0)
        opti.subject_to(thrust[0] <= 10)

        # Set the guesses
        scale = 20

        x_guess = scale * np.sin(np.linspace(0, 3 / 2 * np.pi, N))
        y_guess = scale * np.cos(np.linspace(0, 3 / 2 * np.pi, N))
        z_guess = np.zeros(N)

        u_guess = 10 * np.ones(N)
        v_guess = np.zeros(N)
        w_guess = np.zeros(N)

        phi = np.deg2rad(20) * np.ones(N)
        theta = np.zeros(N)
        psi = np.linspace(0, 3 / 2 * np.pi, N)
        q0g, q1g, q2g, q3g = euler_to_quat(phi, theta, psi)

        p_guess = np.zeros(N)
        q_guess = np.zeros(N)
        r_guess = np.zeros(N)

        thrust_guess = 5 * np.ones(N)
        delta_e_guess = np.zeros(N)
        delta_a_guess = np.zeros(N)
        delta_r_guess = np.zeros(N)

        # Apply the guesses
        opti.set_initial(x_e, x_guess)
        opti.set_initial(y_e, y_guess)
        opti.set_initial(z_e, z_guess)

        opti.set_initial(u_b, u_guess)
        opti.set_initial(v_b, v_guess)
        opti.set_initial(w_b, w_guess)

        opti.set_initial(q0, q0g)
        opti.set_initial(q1, q1g)
        opti.set_initial(q2, q2g)
        opti.set_initial(q3, q3g)

        opti.set_initial(p, p_guess)
        opti.set_initial(q_ang, q_guess)
        opti.set_initial(r, r_guess)

        opti.set_initial(thrust, thrust_guess)
        opti.set_initial(delta_e, delta_e_guess)
        opti.set_initial(delta_a, delta_a_guess)
        opti.set_initial(delta_r, delta_r_guess)

        # Set final conditions
        opti.subject_to(x_e[-1] == 0)
        opti.subject_to(y_e[-1] == 0)
        opti.subject_to(z_e[-1] == 0)

        opti.subject_to(u_b[-1] == 10)
        opti.subject_to(v_b[-1] == 0)
        opti.subject_to(w_b[-1] == 0)

        opti.subject_to(q0[-1] == 0)
        opti.subject_to(q1[-1] == 0)
        opti.subject_to(q2[-1] == 0)
        opti.subject_to(q3[-1] == 1)

        Va = ca.sqrt(u_b[0] ** 2 + v_b[0] ** 2 + w_b[0] ** 2 + 1e-4)
        energy_total = dt * (
                thrust[0] * Va / prop_eff + k_servo * (delta_e[0] ** 2 + delta_a[0] ** 2 + delta_r[0] ** 2))

        for i in range(N - 1):

            opti.subject_to(delta_e[i + 1] > -np.deg2rad(25))
            opti.subject_to(delta_a[i + 1] > -np.deg2rad(25))
            opti.subject_to(delta_r[i + 1] > -np.deg2rad(25))

            opti.subject_to(delta_e[i + 1] < np.deg2rad(25))
            opti.subject_to(delta_a[i + 1] < np.deg2rad(25))
            opti.subject_to(delta_r[i + 1] < np.deg2rad(25))

            opti.subject_to((delta_e[i + 1] - delta_e[i]) / dt < max_servo_speed)
            opti.subject_to((delta_a[i + 1] - delta_a[i]) / dt < max_servo_speed)
            opti.subject_to((delta_r[i + 1] - delta_r[i]) / dt < max_servo_speed)

            opti.subject_to((delta_e[i + 1] - delta_e[i]) / dt > -max_servo_speed)
            opti.subject_to((delta_a[i + 1] - delta_a[i]) / dt > -max_servo_speed)
            opti.subject_to((delta_r[i + 1] - delta_r[i]) / dt > -max_servo_speed)

            opti.subject_to(thrust[i + 1] >= 0)
            opti.subject_to(thrust[i + 1] <= 10)

            state = ca.vertcat(
                u_b[i], v_b[i], w_b[i],
                q0[i], q1[i], q2[i], q3[i],
                p[i], q_ang[i], r[i],
                x_e[i], y_e[i], z_e[i]
            )

            state_next = ca.vertcat(
                u_b[i + 1], v_b[i + 1], w_b[i + 1],
                q0[i + 1], q1[i + 1], q2[i + 1], q3[i + 1],
                p[i + 1], q_ang[i + 1], r[i + 1],
                x_e[i + 1], y_e[i + 1], z_e[i + 1]
            )

            inputs = ca.vertcat(thrust[i], delta_e[i], delta_a[i], delta_r[i])
            inputs_next = ca.vertcat(thrust[i + 1], delta_e[i + 1], delta_a[i + 1], delta_r[i + 1])

            f = compute_derivative(state, inputs, my_plane, wing_data, coeffs_fun)
            f_next = compute_derivative(state_next, inputs_next, my_plane, wing_data, coeffs_fun)

            opti.subject_to(q0[i + 1] >= -1)
            opti.subject_to(q1[i + 1] >= -1)
            opti.subject_to(q2[i + 1] >= -1)
            opti.subject_to(q3[i + 1] >= -1)

            opti.subject_to(q0[i + 1] <= 1)
            opti.subject_to(q1[i + 1] <= 1)
            opti.subject_to(q2[i + 1] <= 1)
            opti.subject_to(q3[i + 1] <= 1)

            opti.subject_to(q0[i + 1] ** 2 + q1[i + 1] ** 2 + q2[i + 1] ** 2 + q3[i + 1] ** 2 <= 1 + 1e-3)
            opti.subject_to(q0[i + 1] ** 2 + q1[i + 1] ** 2 + q2[i + 1] ** 2 + q3[i + 1] ** 2 >= 1 - 1e-3)

            opti.subject_to(state_next == state + 0.5 * dt * (f + f_next))

            Va = ca.sqrt(u_b[i + 1] ** 2 + v_b[i + 1] ** 2 + w_b[i + 1] ** 2 + 1e-4)
            energy_total += dt * (thrust[i + 1] * Va / prop_eff + k_servo * (
                    delta_e[i + 1] ** 2 + delta_a[i + 1] ** 2 + delta_r[i + 1] ** 2))

        # Objective function: weighted sum of mission time and energy consumed
        opti.minimize(time_final + 0.1 * energy_total)

        # Solver setup and solution
        opti.solver("ipopt",
                    {
                        "print_time": False,
                        "ipopt": {
                            "print_level": 0,
                            "sb": "yes",
                            'tol': 1e-4,
                            'max_iter': 1000,
                        }
                    }
                    )

        solution = opti.solve()

        # Extract results
        x_e_opt = solution.value(x_e)
        y_e_opt = solution.value(y_e)
        z_e_opt = solution.value(z_e)
        q0_opt = solution.value(q0)
        q1_opt = solution.value(q1)
        q2_opt = solution.value(q2)
        q3_opt = solution.value(q3)

        u_b_opt = solution.value(u_b)
        v_b_opt = solution.value(v_b)
        w_b_opt = solution.value(w_b)

        p_opt = solution.value(p)
        q_opt = solution.value(q_ang)
        r_opt = solution.value(r)

        thrust_opt = solution.value(thrust)
        delta_e_opt = solution.value(delta_e)
        delta_a_opt = solution.value(delta_a)
        delta_r_opt = solution.value(delta_r)

        time_final_opt = solution.value(time_final)
        energy_final_opt = solution.value(energy_total)

        X_ref = np.vstack([
            u_b_opt, v_b_opt, w_b_opt, q0_opt, q1_opt, q2_opt, q3_opt, p_opt, q_opt, r_opt,
            x_e_opt, y_e_opt, z_e_opt
        ])

        U_ref = np.vstack([thrust_opt, delta_e_opt, delta_a_opt, delta_r_opt])
        t_ref = np.linspace(0, time_final_opt, N)


        X_interp = interp1d(t_ref, X_ref, kind='linear', axis=1, fill_value="extrapolate")
        U_interp = interp1d(t_ref, U_ref, kind='linear', axis=1, fill_value="extrapolate")

        rng = np.random.default_rng(seed=42)

        ensemble_costs = []
        ensemble_rmse = []
        ensemble_success = []
        num_evals = 100

        for i in range(num_evals):

            sigma_des = 0.1
            semispan_noisy = semispan + sigma_des * semispan * rng.standard_normal()
            chord_noisy = chord + sigma_des * chord * rng.standard_normal()
            x_wing_noisy = x_wing + sigma_des * x_wing * rng.standard_normal()

            my_plane_noisy = plane()
            my_plane_noisy.add_segment('left_wing', {'b': semispan_noisy, 'c_tip': chord_noisy, 'c_root': chord_noisy,
                                                     'p_root': np.array([x_wing_noisy, -0.05, 0.0]),
                                                     'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * pi,
                                                     'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})
            my_plane_noisy.add_segment('right_wing', {'b': semispan_noisy, 'c_tip': chord_noisy, 'c_root': chord_noisy,
                                                      'p_root': np.array([x_wing_noisy, 0.05, 0.0]),
                                                      'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * pi,
                                                      'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})
            my_plane_noisy.add_segment('left_ht',
                                       {'b': 0.15, 'c_tip': 0.1, 'c_root': 0.1, 'p_root': np.array([0.7, -0.05, 0.0]),
                                        'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * pi,
                                        'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})
            my_plane_noisy.add_segment('right_ht',
                                       {'b': 0.15, 'c_tip': 0.1, 'c_root': 0.1, 'p_root': np.array([0.7, 0.05, 0.0]),
                                        'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * pi,
                                        'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})
            my_plane_noisy.add_segment('vt', {'b': 0.15, 'c_tip': 0.1, 'c_root': 0.1, 'p_root': np.array([0.7, 0.0, 0.05]),
                                              'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * pi,
                                              'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})


            real_m_noisy, real_cg_noisy, real_Ixx_noisy, real_Iyy_noisy, real_Izz_noisy = compute_real_physics(
                my_plane_noisy, rho=30.4)

            my_plane_noisy.add_mass_properties({
                'mass': real_m_noisy,
                'cg': real_cg_noisy,
                'Ixx': real_Ixx_noisy, 'Iyy': real_Iyy_noisy, 'Izz': real_Izz_noisy,
                'Ixy': 0.0, 'Ixz': 0.0, 'Iyz': 0.0
            })

            wing_ref_noisy = my_plane_noisy.get_segment('left_wing')
            c_root_wing_noisy = wing_ref_noisy.get('c_root', None)
            c_tip_wing_noisy = wing_ref_noisy.get('c_tip', None)
            bref_noisy = wing_ref_noisy.get('b', None) * 2
            taper_ref_noisy = c_tip_wing_noisy / c_root_wing_noisy
            Sref_noisy = (c_root_wing_noisy + c_tip_wing_noisy) * bref_noisy / 2
            cref_noisy = (c_root_wing_noisy + c_tip_wing_noisy) / 2
            wing_data_noisy = {'Sref': Sref_noisy, 'bref': bref_noisy, 'cref': cref_noisy}

            coeffs_fun_noisy = build_coeffs_function(my_plane_noisy, wing_data_noisy, rho=rho, nquad=5)

            current_x = X_ref[:, 0]
            previous_u = U_ref[:, 0]
            A_f, B_f = get_linearization_functions(my_plane, wing_data, coeffs_fun)

            total_time = t_ref[-1]

            dt_sim = 0.02

            Q = np.diag([10, 10, 10, 1000, 1000, 1000, 100, 100, 100, 10, 10, 10])
            R = np.diag([0.1, 0.1, 0.1, 0.1])

            sim_time = 0.0
            t_step = 0

            energy_total = 0
            energy_total_ref = 0

            X_sim = []
            U_sim = []
            spatial_errors = []

            K_prev = []

            x_sym = ca.MX.sym('x', 13)
            u_sym = ca.MX.sym('u', 4)
            xdot = compute_derivative(x_sym, u_sym, my_plane_noisy, wing_data_noisy, coeffs_fun_noisy)
            f_eval = ca.Function("f_eval", [x_sym, u_sym], [xdot])

            while sim_time <= total_time:

                x_ref = X_interp(sim_time)
                u_ff = U_interp(sim_time)

                A_c = np.array(A_f(current_x, previous_u))
                B_c = np.array(B_f(current_x, previous_u))

                K = compute_discrete_lqr_gain(A_c, B_c, Q, R, dt_sim)

                if K is None:
                    K = K_prev  # Use K from the last successful step
                K_prev = K

                vel_err = current_x[0:3] - x_ref[0:3]
                rate_err = current_x[7:10] - x_ref[7:10]
                pos_err = current_x[10:13] - x_ref[10:13]

                q_curr = current_x[3:7]
                q_ref = x_ref[3:7]

                if np.dot(q_curr, q_ref) < 0:
                    q_ref = -q_ref

                q_err = quat_mult(quat_inv(q_ref), q_curr)
                att_err_vec = 2.0 * q_err[1:4]

                state_error = np.concatenate([vel_err, att_err_vec, rate_err, pos_err])

                u_feedback = -K @ state_error
                u_total = u_ff + u_feedback
                u_total[0] = np.clip(u_total[0], 0, 10)
                u_total[1:4] = np.clip(u_total[1:4], -np.deg2rad(25), np.deg2rad(25))

                v_ref_mag = np.linalg.norm(x_ref[0:3])
                w_gust = get_von_karman_disturbances(sim_time, v_ref_mag, seed=i)
                v_wind_inertial = np.array(w_gust)

                def get_wind_relative_xdot(x_in, u_in):
                    v_wind_body = np.array(inertial_to_body(v_wind_inertial, x_in[3:7])).flatten()

                    x_app = np.copy(x_in)
                    x_app[0:3] = x_in[0:3] - v_wind_body

                    xdot_out = np.array(f_eval(ca.DM(x_app), ca.DM(u_in))).flatten()
                    omega_b = x_in[7:10]

                    coriolis_correction = np.cross(omega_b, v_wind_body)

                    xdot_out[0:3] -= coriolis_correction

                    v_ground_body = x_in[0:3]
                    xdot_out[10:13] = np.array(body_to_inertial(v_ground_body, x_in[3:7])).flatten()

                    return xdot_out


                k1 = get_wind_relative_xdot(current_x, u_total)
                k2 = get_wind_relative_xdot(current_x + 0.5 * dt_sim * k1, u_total)
                k3 = get_wind_relative_xdot(current_x + 0.5 * dt_sim * k2, u_total)
                k4 = get_wind_relative_xdot(current_x + dt_sim * k3, u_total)
                current_x = current_x + (dt_sim / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
                previous_u = u_total

                dists = np.linalg.norm(x_ref[10:13] - current_x[10:13], axis=0)
                spatial_errors.append(np.min(dists))

                X_sim.append(current_x)
                U_sim.append(u_total)
                sim_time += dt_sim
                t_step += 1

                u, v, w = current_x[0:3]
                Va = ca.sqrt(u ** 2 + v ** 2 + w ** 2 + 1e-4)
                energy_total += dt_sim * (
                            u_total[0] * Va / prop_eff + k_servo * (u_total[1] ** 2 + u_total[2] ** 2 + u_total[3] ** 2))

                u_ref, v_ref, w_ref = x_ref[0:3]
                Va_ref = ca.sqrt(u_ref ** 2 + v_ref ** 2 + w_ref ** 2 + 1e-4)
                energy_total_ref += dt_sim * (
                            u_ff[0] * Va_ref / prop_eff + k_servo * (u_ff[1] ** 2 + u_ff[2] ** 2 + u_ff[3] ** 2))


            X_final = X_sim[-1]

            final_pos = X_final[10:13]
            target_pos = np.array([0, 0, 0])
            dist_to_gate = np.linalg.norm(final_pos - target_pos)

            gate_limit = 1
            gate_success = dist_to_gate < gate_limit

            RMSE = np.sqrt(np.mean(np.square(spatial_errors)))

            if gate_success == False:
                cost = 200.0
            else:
                cost = sim_time + 0.1 * energy_total + 10 * RMSE

            ensemble_costs.append(cost)
            ensemble_rmse.append(RMSE)
            ensemble_success.append(1.0 if gate_success else 0.0)

        total_cost = np.mean(ensemble_costs)
        total_rmse = np.mean(ensemble_rmse)
        success_rate = np.mean(ensemble_success)
        rmse_std = np.std(ensemble_rmse)
        cost_std = np.std(ensemble_costs)
        success_std = np.std(ensemble_success)

        # Create a unique filename based on parameters
        filename = f"run_{eval_id:05d}_SUCCESS_S{semispan:.4f}_C{chord:.4f}_X{x_wing:.4f}.npz"
        filepath = os.path.join(results_dir, filename)

        np.savez_compressed(
            filepath,
            X_ref=X_ref,
            U_ref=U_ref,
            t_ref=t_ref,
            time_final=time_final_opt,
            energy_final=energy_final_opt,
            cost_final=time_final_opt + 0.1 * energy_final_opt,
            ensembled_cost=total_cost,
            ensembled_RMSE=total_rmse,
            success_rate=success_rate,
            all_costs=np.array(ensemble_costs),
            all_rmse=np.array(ensemble_rmse),
            all_success=np.array(ensemble_success),
            std_costs=cost_std,
            std_rmse=rmse_std,
            std_success=success_std,
            success_flag=1.0,
            params=np.array([semispan, chord, x_wing])
        )

        print(
            f"Mission completed in {time_final_opt:.4f} s, energy: {energy_final_opt:.4f} J and mean cost: {total_cost:.4f}, with a semispan of: {semispan:.4f} m, a chord of: {chord:.4f} m, and a wing position of: {x_wing:.4f} m")

        return float(total_cost)


    except Exception as e:
        print(f"!CMA-ES could not find a solution! : {e}")
        filename = f"run_{eval_id:05d}_FAIL_S{semispan:.4f}_C{chord:.4f}_X{x_wing:.4f}.npz"
        filepath = os.path.join(results_dir, filename)
        np.savez_compressed(
            filepath,
            X_ref=np.zeros((13, 1)),
            U_ref=np.zeros((4, 1)),
            t_ref=np.array([0]),
            time_final=10.0,
            energy_final=1000.0,
            cost_final=200.0,
            ensembled_cost=200.0,
            ensembled_RMSE=10.0,
            success_rate=0.0,
            all_costs=np.array([200.0]),
            all_rmse=np.array([10.0]),
            all_success=np.array([0.0]),
            std_costs=0.0,
            std_rmse=0.0,
            std_success=0.0,
            success_flag=0.0,
            params=np.array([semispan, chord, x_wing])
        )
        return 200.0


if __name__ == '__main__':

    results_dir = "results/results_robust_codesign_horizontalhairpin"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    x0 = [0.25, 0.25, 0.25]

    sigma0 = 0.15

    opts = {
        'bounds': [[0.05, 0.05, 0.05], [0.5, 0.5, 0.5]],
        'maxfevals': 4800,
        'popsize': 32,
        'seed': 1,
        'tolfun': 1e-4,
        'tolx': 1e-4,
    }
    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    logger = cma.CMADataLogger(os.path.join(results_dir, "cma_")).register(es)

    num_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', 7))
    total_evals = 0

    with mp.Pool(processes=num_cores) as pool:
        while not es.stop():
            solutions = es.ask()
            payloads = []
            for x in solutions:
                payloads.append((x, total_evals))
                total_evals += 1

            costs = pool.map(objective_function, payloads)
            es.tell(solutions, costs)
            es.disp()

            logger.add()

    res = es.result
    logger.plot()

    with open(os.path.join(results_dir, "final_es_object.pkl"), "wb") as f:
        pickle.dump(es, f)

    print(f"Optimal Semispan: {res[0][0]}")
    print(f"Optimal Chord:    {res[0][1]}")
    print(f"Optimal X_Wing:   {res[0][2]}")
    print(f"Minimum Cost:     {res[1]}")