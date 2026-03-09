import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.integrate import solve_ivp
from casadi import *
import numpy as np
from scipy.spatial.transform import Slerp
from matplotlib import pyplot as plt
from aerodynamic_model import compute_totals_coeffs_ca, build_coeffs_function


# Rotation functions
def angle_to_quaternion(yaw, pitch, roll):
    """
    Quaternion from aerospace Euler angles (yaw, pitch, roll).
    Works for NumPy floats or CasADi SX/MX.
    Output format: [q0, q1, q2, q3] scalar-first.
    """
    if isinstance(yaw, (MX, SX)) or isinstance(pitch, (MX, SX)) or isinstance(roll, (MX, SX)):
        cy = cos(yaw * 0.5)
        sy = sin(yaw * 0.5)
        cp = cos(pitch * 0.5)
        sp = sin(pitch * 0.5)
        cr = cos(roll * 0.5)
        sr = sin(roll * 0.5)
        q0 = cy * cp * cr + sy * sp * sr
        q1 = cy * cp * sr - sy * sp * cr
        q2 = sy * cp * sr + cy * sp * cr
        q3 = sy * cp * cr - cy * sp * sr
        return vertcat(q0, q1, q2, q3)
    else:
        r = R.from_euler('ZYX', [yaw, pitch, roll])
        q = r.as_quat()  # [x, y, z, w]
        return np.array([q[3], q[0], q[1], q[2]])  # [w, x, y, z]


def quaternion_to_rotation_matrix(q):
    """
    Convert quaternion [q0, q1, q2, q3] to rotation matrix.
    Works with NumPy or CasADi types.
    """
    if isinstance(q, (MX, SX)):
        q0, q1, q2, q3 = q[0], q[1], q[2], q[3]
        return vertcat(
            horzcat(1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)),
            horzcat(2 * (q1 * q2 - q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 + q0 * q1)),
            horzcat(2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2))
        )
    else:
        q0, q1, q2, q3 = q
        return np.array([
            [1 - 2 * (q2 ** 2 + q3 ** 2), 2 * (q1 * q2 + q0 * q3), 2 * (q1 * q3 - q0 * q2)],
            [2 * (q1 * q2 - q0 * q3), 1 - 2 * (q1 ** 2 + q3 ** 2), 2 * (q2 * q3 + q0 * q1)],
            [2 * (q1 * q3 + q0 * q2), 2 * (q2 * q3 - q0 * q1), 1 - 2 * (q1 ** 2 + q2 ** 2)]
        ])


def inertial_to_body(vec, q):
    """
    Transform vector from inertial frame to body frame.
    Works for NumPy or CasADi types.
    """
    Rb = quaternion_to_rotation_matrix(q)
    if isinstance(vec, (MX, SX)):
        return mtimes(Rb, vec)
    else:
        return Rb @ vec


def body_to_inertial(vec, q):
    """
    Transform vector from body frame to inertial frame.
    Works for NumPy or CasADi types.
    """
    Rb = quaternion_to_rotation_matrix(q)
    if isinstance(vec, (MX, SX)):
        return mtimes(Rb.T, vec)
    else:
        return Rb.T @ vec


def compute_derivative(state, control, my_plane, wing_data, coeffs_fun):
    """
    Computes the dynamics of a fixed-wing aircraft in inertial frame.

    Parameters:
    state: np.ndarray
        State vector [ u, v, w, q0, q1, q2, q3, p, q, r]
    control: np.ndarrayx
        Control vector [thrust, alpha_left, alpha_right]

    Returns:
    np.ndarray
        Derivative of the state vector
    """
    G = 9.81
    rho = 1.225  # sea level air density in kg/m^3

    cg = my_plane.get_mass_properties('cg')
    mass = my_plane.get_mass_properties('mass')
    Ixx = my_plane.get_mass_properties('Ixx')
    Iyy = my_plane.get_mass_properties('Iyy')
    Izz = my_plane.get_mass_properties('Izz')
    Ixz = my_plane.get_mass_properties('Ixz')

    # --- Extract state variables ---
    u = state[0]
    v = state[1]
    w = state[2]

    quaternion = state[3:7]  # still works as a slice (returns MX/DM vector)

    p = state[7]
    q = state[8]
    r = state[9]

    thrust = control[0]
    delta_e = control[1]
    delta_a = control[2]
    delta_r = control[3]

    if isinstance(quaternion[0], (MX, SX)):
        g_vec = vertcat(0, 0, G)
        g_body = inertial_to_body(g_vec, quaternion)
    else:
        g_body = inertial_to_body(np.array([0, 0, G]), quaternion)

    Ve = sqrt(u * u + v * v + w * w)
    qbar = 0.5 * 1.225 * Ve * Ve
    Sref = wing_data['Sref']
    bref = wing_data['bref']
    cref = wing_data['cref']

    C = coeffs_fun(u, v, w, p, q, r, delta_a, delta_e, delta_r, cg)

    if isinstance(C, (list, tuple)):
        C = C[0]  # now C is a 6x1 CasADi vector

    # Extract components
    CX, CY, CZ, CL, CM, CN = C[0], C[1], C[2], C[3], C[4], C[5]

    thrust_vec = vertcat(thrust, 0, 0)

    F_body = vertcat(CX, CY, CZ) * (qbar * Sref) + thrust_vec
    L = CL * (qbar * Sref * bref)
    M = CM * (qbar * Sref * cref)
    N = CN * (qbar * Sref * bref)

    a_body = F_body / mass

    udot = a_body[0] + g_body[0] + r * v - q * w
    vdot = a_body[1] + g_body[1] + p * w - r * u
    wdot = a_body[2] + g_body[2] + q * u - p * v


    Lprimeprime = L + Ixz * p * q - (Izz - Iyy) * r * q
    Nprime = N - (Iyy - Ixx) * p * q - Ixz * r * q

    pdot = (Lprimeprime * Izz - Nprime * Ixz) / (Ixx * Izz - Ixz ** 2)
    qdot = (M - (Ixx - Izz) * p * r - Ixz * (p ** 2 - r ** 2)) / Iyy
    rdot = (Nprime * Ixx + Lprimeprime * Ixz) / (Ixx * Izz - Ixz ** 2)

    if isinstance(quaternion[0], (MX, SX)):
        update_matrix = vertcat(
            horzcat(0, -p, -q, -r),
            horzcat(p, 0, r, -q),
            horzcat(q, -r, 0, p),
            horzcat(r, q, -p, 0)
        )
        quarterniondot = 0.5 * mtimes(update_matrix, quaternion)
    else:
        update_matrix = np.array([
            [0, -p, -q, -r],
            [p, 0, r, -q],
            [q, -r, 0, p],
            [r, q, -p, 0]
        ])
        quarterniondot = 0.5 * update_matrix @ quaternion


    if isinstance(u, (MX, SX)):
        v_body = vertcat(u, v, w)
        xyzdot = body_to_inertial(v_body, quaternion)
        xdot = vertcat(
            udot, vdot, wdot,
            quarterniondot,
            pdot, qdot, rdot,
            xyzdot
        )
    else:
        v_body = np.array([u, v, w])
        xyzdot = body_to_inertial(v_body, quaternion)

        u_v_w_dot = np.array([udot, vdot, wdot]).flatten()
        q_dot = np.array(quarterniondot).flatten()
        p_q_r_dot = np.array([pdot, qdot, rdot]).flatten()
        pos_dot = np.array(xyzdot).flatten()

        xdot = np.concatenate((
            u_v_w_dot,
            q_dot,
            p_q_r_dot,
            pos_dot
        ))

    return xdot
