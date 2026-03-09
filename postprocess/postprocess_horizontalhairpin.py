import numpy as np
import os
import glob
import sys

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from mpl_toolkits.mplot3d import Axes3D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from build_plane import plane



def draw_drone_parameterized(ax, color, span, chord, x_wing, title=""):

    total_m = 0
    total_moment = np.array([0.0, 0.0, 0.0])

    my_plane = plane()
    my_plane.add_segment('left_wing',
                         {'b': span / 2, 'c_tip': chord, 'c_root': chord, 'p_root': np.array([x_wing, -0.05, 0.0]),
                          'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * np.pi,
                          'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})
    my_plane.add_segment('right_wing',
                         {'b': span / 2, 'c_tip': chord, 'c_root': chord, 'p_root': np.array([x_wing, 0.05, 0.0]),
                          'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * np.pi,
                          'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})
    my_plane.add_segment('left_ht', {'b': 0.15, 'c_tip': 0.1, 'c_root': 0.1, 'p_root': np.array([0.7, -0.05, 0.0]),
                                     'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * np.pi,
                                     'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})
    my_plane.add_segment('right_ht', {'b': 0.15, 'c_tip': 0.1, 'c_root': 0.1, 'p_root': np.array([0.7, 0.05, 0.0]),
                                      'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * np.pi,
                                      'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})
    my_plane.add_segment('vt', {'b': 0.15, 'c_tip': 0.1, 'c_root': 0.1, 'p_root': np.array([0.7, 0.0, 0.05]),
                                'sweep_quater': 0.0, 'Gamma': 0.0, 'Cl_alpha': 2.0 * np.pi,
                                'cf_c': 0.3, 'CD0': 0.02, 'tc': 0.12})

    segment_names = ['left_wing', 'right_wing', 'left_ht', 'right_ht', 'vt']
    parts = []

    Ixx = Iyy = Izz = 0

    for name in segment_names:
        s = my_plane.get_segment(name)

        semispan = s['b']
        avg_chord = (s['c_root'] + s['c_tip']) / 2
        thickness = s['tc'] * avg_chord
        volume = semispan * avg_chord * thickness

        rho = 30.4
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

    lw = my_plane.get_segment('left_wing')
    rw = my_plane.get_segment('right_wing')
    lht = my_plane.get_segment('left_ht')
    rht = my_plane.get_segment('right_ht')


    S_w = (lw['c_root'] + lw['c_tip']) * lw['b']
    taper_w = lw['c_tip'] / lw['c_root']
    c_mac_w = (lw['c_tip'] + lw['c_root']) / 2
    x_ac_w = lw['p_root'][0] + 0.25 * c_mac_w

    S_h = (lht['c_root'] + lht['c_tip']) * lht['b']
    taper_h = lht['c_tip'] / lht['c_root']
    c_mac_h = (lht['c_tip'] + lht['c_root']) / 2

    x_ac_h = lht['p_root'][0] + 0.25 * c_mac_h

    eta = 0.9
    tail_arm = x_ac_h - x_ac_w
    x_np = x_ac_w + (S_h / S_w) * tail_arm * eta
    x_cg = cg_ac[0]

    fus_l, fus_w = 0.8, 0.1
    tail_s, tail_c = 0.3, 0.1


    ax.add_patch(patches.Rectangle((-fus_l / 2, -fus_w / 2), fus_l, fus_w,
                                   color=color, alpha=0.9, zorder=2))
    wing = patches.Rectangle((-fus_l / 2 + x_wing, -span / 2), chord, span,
                             color=color, alpha=0.7, zorder=3)
    ax.add_patch(wing)

    ax.add_patch(patches.Rectangle((fus_l / 2 - tail_c, -tail_s / 2), tail_c, tail_s,
                                   color=color, alpha=0.7, zorder=1))

    ax.scatter(-fus_l / 2 + x_np + 0.05, 0, marker='o', s=100, facecolors='none',
               edgecolors='#F5F5F5', linewidth=3, zorder=5, label='NP')

    ax.scatter(-fus_l / 2 + x_cg + 0.05, 0, marker='x', s=100, color='#F5F5F5',
               linewidth=3, zorder=5, label='CG')

    ax.set_xlim([-0.5, 0.5])
    ax.set_ylim([-0.5, 0.5])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(title, fontsize=10, fontweight='bold')

def plot_ensemble_top_view(ref_nocontrol, ref_control, ensemble_nc, ensemble_c):
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['xtick.labelsize'] = 7
    plt.rcParams['ytick.labelsize'] = 7

    fig = plt.figure(figsize=(3.5, 4))
    fig.suptitle("Horizontal Hairpin", fontsize=10, fontweight='bold', y=0.97)

    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.05)

    color_rob = '#0072B2'
    color_nom = '#D55E00'
    gate_color = '#009E73'

    ax1 = fig.add_subplot(gs[0, :])

    ax1.plot(ref_nocontrol[10, :], ref_nocontrol[11, :], color=color_nom,
            linestyle='--', linewidth=2, label='Reference (Nominal)', zorder=4)
    ax1.plot(ref_control[10, :], ref_control[11, :], color=color_rob,
            linestyle='--', linewidth=2, label='Reference (Robust)', zorder=4)


    for i in range(90):
        lbl_nc = 'Individual Runs (Nominal)' if i == 0 else ""
        ax1.plot(ensemble_nc[i, 10, :], ensemble_nc[i, 11, :],
                color=color_nom, alpha=0.15, linewidth=0.5, zorder=3, label=lbl_nc)

    for i in range(90):
        lbl_c = 'Individual Runs (Robust)' if i == 0 else ""
        ax1.plot(ensemble_c[i, 10, :], ensemble_c[i, 11, :],
                color=color_rob, alpha=0.15, linewidth=0.5, zorder=3, label=lbl_c)

    gate_x, gate_y = ref_control[10, -1], ref_control[11, -1]
    gate_radius = 1.0

    ax1.vlines(gate_x, gate_y - gate_radius, gate_y + gate_radius,
              colors=gate_color, linewidth=3, label='Target Gate', zorder=6)
    ax1.hlines([gate_y - gate_radius, gate_y + gate_radius], gate_x - 0.5, gate_x + 0.5,
              colors=gate_color, linewidth=1.5, zorder=5)
    ax1.plot(gate_x, gate_y, 'o', color=gate_color, markersize=4, zorder=7)

    ax1.set_ylabel('Y (m)', fontsize=8)
    ax1.set_xlim([0, 25])
    ax1.set_ylim([-2.5, 7.5])
    ax1.yaxis.set_label_coords(-0.1, 0.5)
    ax1.grid(True, linestyle=':', alpha=0.5)
    ax1.set_aspect('equal')


    ax2 = fig.add_subplot(gs[1, :], sharex=ax1)

    ax2.plot(ref_nocontrol[10, :], -ref_nocontrol[12, :], color=color_nom,
            linestyle='--', linewidth=2, label='Reference (Nominal)', zorder=4)
    ax2.plot(ref_control[10, :], -ref_control[12, :], color=color_rob,
            linestyle='--', linewidth=2, label='Reference (Robust)', zorder=4)

    for i in range(90):
        lbl_nc = 'Individual Runs (Nominal)' if i == 0 else ""
        ax2.plot(ensemble_nc[i, 10, :], -ensemble_nc[i, 12, :],
                color=color_nom, alpha=0.15, linewidth=0.5, zorder=3, label=lbl_nc)

    for i in range(90):
        lbl_c = 'Individual Runs (Robust)' if i == 0 else ""
        ax2.plot(ensemble_c[i, 10, :], -ensemble_c[i, 12, :],
                color=color_rob, alpha=0.15, linewidth=0.5, zorder=3, label=lbl_c)


    gate_x, gate_y = ref_control[10, -1], -ref_control[12, -1]
    gate_radius = 1.0

    ax2.vlines(gate_x, gate_y - gate_radius, gate_y + gate_radius,
              colors=gate_color, linewidth=3, label='Target Gate', zorder=6)
    ax2.hlines([gate_y - gate_radius, gate_y + gate_radius], gate_x - 0.5, gate_x + 0.5,
              colors=gate_color, linewidth=1.5, zorder=6)
    ax2.plot(gate_x, gate_y, 'o', color=gate_color, markersize=4, zorder=7)


    ax2.set_xlabel('X (m)', fontsize=8)
    ax2.set_ylabel('Z (m)', fontsize=8)
    ax2.yaxis.set_label_coords(-0.1, 0.5)
    ax2.set_xlim([0, 25])
    ax2.set_ylim([-3, 10])
    ax2.grid(True, linestyle=':', alpha=0.5)
    ax2.set_aspect('equal')

    plt.tight_layout()
    plt.savefig("robustness_horiz_states_top.svg", dpi=300, bbox_inches='tight')
    plt.show()

    def save_single_drone(color, span, chord, x_w, filename):
        fig, ax = plt.subplots(figsize=(2, 2))
        draw_drone_parameterized(ax, color, span, chord, x_w)
        ax.axis('off')
        fig.patch.set_alpha(0)
        plt.savefig(filename, dpi=300, bbox_inches='tight', transparent=True)
        plt.show()

    save_single_drone(color_nom, 0.996, 0.310, 0.245, "drone_nominal_horiz.svg")
    save_single_drone(color_rob, 0.894, 0.324, 0.324, "drone_robust_horiz.svg")


def load_from_single_npz(file_path):
    data = np.load(file_path, allow_pickle=True)
    trajectories = []

    run_keys = sorted([k for k in data.files if k.startswith('run_')],
                      key=lambda x: int(x.split('_')[1]))

    for key in run_keys:
        run_data = data[key].item()
        trajectories.append(run_data['X_sim'])

    x_ref = data['X_ref']

    return trajectories, x_ref


def synchronize_trajectories(traj_list):
    if not traj_list: return None

    max_len = max(t.shape[1] for t in traj_list)
    padded_list = []

    for t in traj_list:
        curr_len = t.shape[1]
        if curr_len < max_len:
            last_state = t[:, -1:]
            padding = np.tile(last_state, (1, max_len - curr_len))
            t_padded = np.hstack([t, padding])
            padded_list.append(t_padded)
        else:
            padded_list.append(t)

    return np.array(padded_list)

path_n = "../results/robustness_analysis/evaluation_nominal_horizontalhairpin.npz"
path_r = "../results/robustness_analysis/evaluation_robust_horizontalhairpin.npz"

ensemble_r_list, X_ref_r = load_from_single_npz(path_r)
ensemble_n_list, X_ref_n = load_from_single_npz(path_n)

ensemble_r = synchronize_trajectories(ensemble_r_list)
ensemble_n = synchronize_trajectories(ensemble_n_list)

plot_ensemble_top_view(
    ref_nocontrol=X_ref_n,
    ref_control=X_ref_r,
    ensemble_nc=ensemble_n,
    ensemble_c=ensemble_r
)
