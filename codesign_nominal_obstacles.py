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

        dt = time_final / (N - 1)

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
        opti.subject_to(thrust[0] <= 5)

        # Set the guesses
        x_guess = np.linspace(0, 60, N)
        y_guess = 4 * np.sin(np.pi * x_guess / 30)
        z_guess = np.zeros(N)

        u_guess = 10 * np.ones(N)
        v_guess = np.zeros(N)
        w_guess = np.zeros(N)

        thrust_guess = 4 * np.ones(N)
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

        opti.set_initial(thrust, thrust_guess)
        opti.set_initial(delta_e, delta_e_guess)
        opti.set_initial(delta_a, delta_a_guess)
        opti.set_initial(delta_r, delta_r_guess)

        # Set final conditions
        opti.subject_to(x_e[-1] == 60)
        opti.subject_to(y_e[-1] == 0)
        opti.subject_to(z_e[-1] == 0)

        opti.subject_to(q0[-1] == 1)
        opti.subject_to(q1[-1] == 0)
        opti.subject_to(q2[-1] == 0)
        opti.subject_to(q3[-1] == 0)

        opti.subject_to(u_b[-1] == 10)
        opti.subject_to(v_b[-1] == 0)
        opti.subject_to(w_b[-1] == 0)

        Va = ca.sqrt(u_b[0] ** 2 + v_b[0] ** 2 + w_b[0] ** 2 + 1e-4)
        energy_total = dt * (thrust[0] * Va / prop_eff + k_servo * (delta_e[0] ** 2 + delta_a[0] ** 2 + delta_r[0] ** 2))

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
            opti.subject_to(thrust[i + 1] <= 5)

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

            # Obstacle avoidance constraints
            dist_sq1 = (x_e[i + 1] - 20) ** 2 + (y_e[i + 1] - 0) ** 2
            opti.subject_to(dist_sq1 > (4 + 0.5 * bref) ** 2)
            dist_sq2 = (x_e[i + 1] - 40) ** 2 + (y_e[i + 1] - 0) ** 2
            opti.subject_to(dist_sq2 > (4 + 0.5 * bref) ** 2)

            opti.subject_to(state_next == state + 0.5 * dt * (f + f_next))

            Va = ca.sqrt(u_b[i + 1] ** 2 + v_b[i + 1] ** 2 + w_b[i + 1] ** 2 + 1e-4)
            energy_total += dt * (thrust[i + 1] * Va / prop_eff + k_servo * (delta_e[i + 1] ** 2 + delta_a[i + 1] ** 2 + delta_r[i + 1] ** 2))

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

        # Create a unique filename based on parameters
        filename = f"run_{eval_id:05d}_SUCCESS_S{semispan:.4f}_C{chord:.4f}_X{x_wing:.4f}.npz"
        filepath = os.path.join(results_dir, filename)


        total_cost = time_final_opt + 0.1 * energy_final_opt

        np.savez_compressed(
            filepath,
            X_ref=X_ref,
            U_ref=U_ref,
            t_ref=t_ref,
            time_final=time_final_opt,
            energy_final=energy_final_opt,
            cost_final=time_final_opt + 0.1 * energy_final_opt,
            ensembled_cost=total_cost,
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
            success_flag=0.0,
            params=np.array([semispan, chord, x_wing])
        )

        return 200.0


if __name__ == '__main__':

    results_dir = "results/results_nominal_codesign_obstacles"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    x0 = [0.25, 0.25, 0.25]

    sigma0 = 0.15

    opts = {
        'bounds': [[0.05, 0.05, 0.05], [0.5, 0.5, 0.5]],
        'maxfevals': 3200,
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