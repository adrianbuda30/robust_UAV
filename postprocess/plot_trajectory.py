import casadi as ca
import numpy as np

import matplotlib.pyplot as plt
import os
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec


def rotation_matrix_to_euler_xyz(R):

    r20 = R[2, 0]
    r21 = R[2, 1]
    r22 = R[2, 2]
    r10 = R[1, 0]
    r00 = R[0, 0]
    r12 = R[1, 2]
    r11 = R[1, 1]

    epsilon = 1e-6

    is_singular_pos = ca.logic_and(ca.fabs(r20 - 1.0) < epsilon, r20 > 0)
    is_singular_neg = ca.logic_and(ca.fabs(r20 + 1.0) < epsilon, r20 < 0)
    is_regular = ca.logic_not(ca.logic_or(is_singular_pos, is_singular_neg))

    theta_regular = ca.asin(-r20)
    phi_regular = ca.atan2(r21, r22 + 1e-6)
    psi_regular = ca.atan2(r10, r00 + 1e-6)

    # Singular case: R[2,0] == -1 (gimbal lock at +90 deg pitch)
    theta_pos = ca.pi / 2
    phi_pos = -ca.atan2(-r12, r11 + 1e-6)
    psi_pos = 0

    # Singular case: R[2,0] == +1 (gimbal lock at -90 deg pitch)
    theta_neg = -ca.pi / 2
    phi_neg = ca.atan2(-r12, r11 + 1e-6)
    psi_neg = 0

    theta = ca.if_else(is_regular, theta_regular,
                       ca.if_else(is_singular_pos, theta_pos, theta_neg))
    phi = ca.if_else(is_regular, phi_regular,
                     ca.if_else(is_singular_pos, phi_pos, phi_neg))
    psi = ca.if_else(is_regular, psi_regular,
                     ca.if_else(is_singular_pos, psi_pos, psi_neg))

    return phi, theta, psi

def load_reference_traj(traj_file):
    if os.path.exists(traj_file):
        try:
            d = np.load(traj_file)
            t_opt = d["t_opt"]
            X_opt = d["X_opt"]
            U_opt = d["U_opt"]
            return t_opt, X_opt, U_opt
        except Exception as e:
            print("Failed loading .npz:", e)
    return t_opt, X_opt, U_opt

def extract_plot_data(X_ref):
    x_e = X_ref[10, :]
    y_e = X_ref[11, :]
    z_e = X_ref[12, :]

    q0_opt = X_ref[3, :]
    q1_opt = X_ref[4, :]
    q2_opt = X_ref[5, :]
    q3_opt = X_ref[6, :]

    phi_opt = np.zeros(len(q0_opt))
    theta_opt = np.zeros(len(q0_opt))
    psi_opt = np.zeros(len(q0_opt))

    for index in range(len(q0_opt)):
        R = np.array([[1 - 2 * (q2_opt[index] ** 2 + q3_opt[index] ** 2),
                       2 * (q1_opt[index] * q2_opt[index] - q0_opt[index] * q3_opt[index]),
                       2 * (q1_opt[index] * q3_opt[index] + q0_opt[index] * q2_opt[index])],
                      [2 * (q1_opt[index] * q2_opt[index] + q0_opt[index] * q3_opt[index]),
                       1 - 2 * (q1_opt[index] ** 2 + q3_opt[index] ** 2),
                       2 * (q2_opt[index] * q3_opt[index] - q0_opt[index] * q1_opt[index])],
                      [2 * (q1_opt[index] * q3_opt[index] - q0_opt[index] * q2_opt[index]),
                       2 * (q2_opt[index] * q3_opt[index] + q0_opt[index] * q1_opt[index]),
                       1 - 2 * (q1_opt[index] ** 2 + q2_opt[index] ** 2)]])

        phi_opt[index], theta_opt[index], psi_opt[index] = rotation_matrix_to_euler_xyz(R)

    return x_e, y_e, z_e, theta_opt, phi_opt, psi_opt


def naca0012_profile(x):
    """Returns the half-thickness of NACA 0012."""
    return 0.6 * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x ** 2 + 0.2843 * x ** 3 - 0.1015 * x ** 4)


def plot_fixed_wing_snapshots_obstacles(
        ax, x_e_vals, y_e_vals, z_e_vals,
        theta_vals, phi_vals, psi_vals,
        wing_span=0.75, wing_chord=0.25,
        tail_span=0.3, tail_chord=0.1,
        tail_height=0.15, fuse_radius=0.03,
        scale=5.0, num_snapshots=6
):

    ws, wc = wing_span * scale, wing_chord * scale
    ts, tc, th = tail_span * scale, tail_chord * scale, tail_height * scale
    r, fl = fuse_radius * scale, 0.8 * scale

    ax.plot(x_e_vals, y_e_vals, z_e_vals, label="Trajectory", lw=1.5, color='#0072B2', alpha=0.5, ls=':')

    indices = np.linspace(0, len(x_e_vals) - 1, num_snapshots, dtype=int)

    final_x = x_e_vals[-1]
    final_y = y_e_vals[-1]
    final_z = z_e_vals[-1]
    gate_radius = 5.0

    gate_theta = np.linspace(0, 2 * np.pi, 50)
    gy = final_y + gate_radius * np.cos(gate_theta)
    gz = final_z + gate_radius * np.sin(gate_theta)
    gx = np.full_like(gy, final_x)

    ax.plot(gx, gy, gz, color='green', linewidth=1.5, alpha=0.6, zorder=10, label='Target Gate')

    for frame in indices:
        x, y, z = x_e_vals[frame], y_e_vals[frame], z_e_vals[frame]
        th_f, ph_f, ps_f = -theta_vals[frame], -phi_vals[frame], psi_vals[frame]

        # Rotation Matrix (Body to Inertia)
        cr, cp, cy = np.cos(ph_f), np.cos(th_f), np.cos(ps_f)
        sr, sp, sy = np.sin(ph_f), np.sin(th_f), np.sin(ps_f)
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])

        def transform(X, Y, Z):
            points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()])
            transformed = R @ points + np.array([[x], [y], [z]])
            return transformed[0].reshape(X.shape), transformed[1].reshape(Y.shape), transformed[2].reshape(Z.shape)

        fx = np.linspace(fl / 2, -fl / 2, 10)
        ftheta = np.linspace(0, 2 * np.pi, 12)
        FX, FTHETA = np.meshgrid(fx, ftheta)
        FY, FZ = r * np.cos(FTHETA), r * np.sin(FTHETA)
        AX, AY, AZ = transform(FX, FY, FZ)
        ax.plot_surface(AX, AY, AZ, color='#D55E00', alpha=0.2 + 0.1 * frame / 15, shade=True)

        def add_naca_part(x_off, chord, span, z_off, color, is_vert=False):
            xl = np.linspace(0, 1, 10)
            sl = np.linspace(-span / 2, span / 2, 10)
            X_loc, S_loc = np.meshgrid(xl, sl)
            Z_thick = naca0012_profile(X_loc) * chord
            X_b = x_off + (0.5 - X_loc) * chord

            if not is_vert:
                Y_b = S_loc
                TX_u, TY_u, TZ_u = transform(X_b, Y_b, z_off + Z_thick)
                ax.plot_surface(TX_u, TY_u, TZ_u, color=color, shade=True, alpha=0.2 + 0.1 * frame / 15)
                TX_l, TY_l, TZ_l = transform(X_b, Y_b, z_off - Z_thick)
                ax.plot_surface(TX_l, TY_l, TZ_l, color=color, shade=True, alpha=0.2 + 0.1 * frame / 15)
            else:
                Z_b = z_off + (S_loc + span / 2)
                TX_l, TY_l, TZ_l = transform(X_b, Z_thick, Z_b)
                ax.plot_surface(TX_l, TY_l, TZ_l, color=color, shade=True, alpha=0.2 + 0.1 * frame / 15)
                TX_r, TY_r, TZ_r = transform(X_b, -Z_thick, Z_b)
                ax.plot_surface(TX_r, TY_r, TZ_r, color=color, shade=True, alpha=0.2 + 0.1 * frame / 15)

        # Draw Parts
        add_naca_part(0.15 * scale, wc, ws, 0, '#D55E00')  # Main Wing
        tx_pos = -fl / 2 + tc / 2
        add_naca_part(tx_pos, tc, ts, 0, '#D55E00')  # Horizontal Tail
        add_naca_part(tx_pos, tc, th, r, '#D55E00', is_vert=True)  # Vertical Tail



    padding = 0.2 * scale
    def add_obstacle(center_x, center_y, radius, height):
        z_grid = np.linspace(np.min(z_e_vals) - padding, np.max(z_e_vals) + padding, 20)
        theta_grid = np.linspace(0, 2 * np.pi, 30)
        Z_cyl, THETA_cyl = np.meshgrid(z_grid, theta_grid)
        X_cyl = center_x + radius * np.cos(THETA_cyl)
        Y_cyl = center_y + radius * np.sin(THETA_cyl)

        ax.plot_surface(X_cyl, Y_cyl, Z_cyl, color='lightgrey', alpha=0.3, edgecolor='lightgrey', lw=0.1)

    add_obstacle(20, 0, 4, 20)
    add_obstacle(40, 0, 4, 20)


    ax.set_xlabel('X [m]')
    ax.set_ylabel('Y [m]')
    ax.set_zlabel('Z [m]')

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.view_init(elev=30, azim=-120)

    ax.set_xlim(np.min(x_e_vals) - padding, np.max(x_e_vals) + padding)
    ax.set_ylim(np.min(y_e_vals) - padding, np.max(y_e_vals) + padding)
    ax.set_zlim(np.min(z_e_vals) - padding, np.max(z_e_vals) + padding)
    ax.set_box_aspect((5, 1, 0.8))

    ax.set_axis_off()
    ax.set_aspect('auto')
    ax.dist = 3



def plot_fixed_wing_snapshots(
        ax, x_e_vals, y_e_vals, z_e_vals,
        theta_vals, phi_vals, psi_vals,
        wing_span=0.75, wing_chord=0.25,
        tail_span=0.3, tail_chord=0.1,
        tail_height=0.15, fuse_radius=0.03,
        scale=5.0, num_snapshots=6
):


    ws, wc = wing_span * scale, wing_chord * scale
    ts, tc, th = tail_span * scale, tail_chord * scale, tail_height * scale
    r, fl = fuse_radius * scale, 0.8 * scale

    ax.plot(x_e_vals, y_e_vals, z_e_vals, label="Trajectory", lw=1.5, color='#0072B2', alpha=0.5, ls=':')

    indices = np.linspace(0, len(x_e_vals) - 1, num_snapshots, dtype=int)

    final_x = x_e_vals[-1]
    final_y = y_e_vals[-1]
    final_z = z_e_vals[-1]
    gate_radius = 5.0

    gate_theta = np.linspace(0, 2 * np.pi, 50)
    gy = final_y + gate_radius * np.cos(gate_theta)
    gz = final_z + gate_radius * np.sin(gate_theta)
    gx = np.full_like(gy, final_x)

    ax.plot(gx, gy, gz, color='green', linewidth=1.5, alpha=0.6, zorder=10, label='Target Gate')


    for frame in indices:
        x, y, z = x_e_vals[frame], y_e_vals[frame], z_e_vals[frame]
        th_f, ph_f, ps_f = -theta_vals[frame], -phi_vals[frame], psi_vals[frame]

        # Rotation Matrix (Body to Inertia)
        cr, cp, cy = np.cos(ph_f), np.cos(th_f), np.cos(ps_f)
        sr, sp, sy = np.sin(ph_f), np.sin(th_f), np.sin(ps_f)
        R = np.array([
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]
        ])

        def transform(X, Y, Z):
            points = np.vstack([X.flatten(), Y.flatten(), Z.flatten()])
            transformed = R @ points + np.array([[x], [y], [z]])
            return transformed[0].reshape(X.shape), transformed[1].reshape(Y.shape), transformed[2].reshape(Z.shape)

        fx = np.linspace(fl / 2, -fl / 2, 10)
        ftheta = np.linspace(0, 2 * np.pi, 12)
        FX, FTHETA = np.meshgrid(fx, ftheta)
        FY, FZ = r * np.cos(FTHETA), r * np.sin(FTHETA)
        AX, AY, AZ = transform(FX, FY, FZ)
        ax.plot_surface(AX, AY, AZ, color='#D55E00', alpha=0.2 + 0.1 * frame / 15, shade=True)

        def add_naca_part(x_off, chord, span, z_off, color, is_vert=False):
            xl = np.linspace(0, 1, 10)
            sl = np.linspace(-span / 2, span / 2, 10)
            X_loc, S_loc = np.meshgrid(xl, sl)
            Z_thick = naca0012_profile(X_loc) * chord
            X_b = x_off + (0.5 - X_loc) * chord

            if not is_vert:
                Y_b = S_loc
                TX_u, TY_u, TZ_u = transform(X_b, Y_b, z_off + Z_thick)
                ax.plot_surface(TX_u, TY_u, TZ_u, color=color, shade=True, alpha=0.2 + 0.1 * frame / 15)
                TX_l, TY_l, TZ_l = transform(X_b, Y_b, z_off - Z_thick)
                ax.plot_surface(TX_l, TY_l, TZ_l, color=color, shade=True, alpha=0.2 + 0.1 * frame / 15)
            else:
                Z_b = z_off + (S_loc + span / 2)
                TX_l, TY_l, TZ_l = transform(X_b, Z_thick, Z_b)
                ax.plot_surface(TX_l, TY_l, TZ_l, color=color, shade=True, alpha=0.2 + 0.1 * frame / 15)
                TX_r, TY_r, TZ_r = transform(X_b, -Z_thick, Z_b)
                ax.plot_surface(TX_r, TY_r, TZ_r, color=color, shade=True, alpha=0.2 + 0.1 * frame / 15)

        # Draw Parts
        add_naca_part(0.15 * scale, wc, ws, 0, '#D55E00')  # Main Wing
        tx_pos = -fl / 2 + tc / 2
        add_naca_part(tx_pos, tc, ts, 0, '#D55E00')  # Horizontal Tail
        add_naca_part(tx_pos, tc, th, r, '#D55E00', is_vert=True)  # Vertical Tail

    ax.set_xlabel('X (North) [m]')
    ax.set_ylabel('Y (East) [m]')
    ax.set_zlabel('Z (Up) [m]')

    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    ax.view_init(elev=30, azim=-105)

    padding = 0.5 * scale
    ax.set_xlim(np.min(x_e_vals) - padding, np.max(x_e_vals) + padding)
    ax.set_ylim(np.min(y_e_vals) - padding, np.max(y_e_vals) + padding)
    ax.set_zlim(np.min(z_e_vals) - padding, np.max(z_e_vals) + padding)
    ax.set_box_aspect((1, 1, 1))
    ax.set_aspect('equal')

    ax.set_axis_off()



traj_file_obstacles = "../results/optimal_traj_obstacles.npz"
traj_file_vertical = "../results/optimal_traj_verticalreversal.npz"
traj_file_circular = "../results/optimal_traj_horizontalhairpin.npz"

t_obstacles, X_obstacles, U_obstacles = load_reference_traj(traj_file_obstacles)
t_vertical, X_vertical, U_vertical = load_reference_traj(traj_file_vertical)
t_circular, X_circular, U_circular = load_reference_traj(traj_file_circular)

x_e_obs, y_e_obs, z_e_obs, th_obs, ph_obs, ps_obs = extract_plot_data(X_obstacles)
x_e_vert, y_e_vert, z_e_vert, th_vert, ph_vert, ps_vert = extract_plot_data(X_vertical)
x_e_circ, y_e_circ, z_e_circ, th_circ, ph_circ, ps_circ = extract_plot_data(X_circular)



fig1 = plt.figure(figsize=(8, 3))
ax1 = fig1.add_subplot(111, projection='3d')

plot_fixed_wing_snapshots_obstacles(
    ax1, x_e_obs, y_e_obs, -z_e_obs, th_obs, ph_obs, ps_obs,
)

plt.tight_layout()
plt.savefig("3d_trajectory_obstacles.pdf",
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=True)
plt.show()


fig2 = plt.figure(figsize=(8, 3))
ax2 = fig2.add_subplot(111, projection='3d')

plot_fixed_wing_snapshots(
    ax2, x_e_vert, y_e_vert, -z_e_vert, th_vert, ph_vert, ps_vert,
)

plt.tight_layout()
plt.savefig("3d_trajectory_verticalreversal.pdf",
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=True)

plt.show()


fig3 = plt.figure(figsize=(8, 3))
ax3 = fig3.add_subplot(111, projection='3d')

plot_fixed_wing_snapshots(
    ax3, x_e_circ, y_e_circ, -z_e_circ, th_circ, ph_circ, ps_circ,
)


plt.tight_layout()
plt.savefig("3d_trajectory_horizontalhairpin.pdf",
            dpi=300,
            bbox_inches='tight',
            pad_inches=0.1,
            transparent=True)

plt.show()