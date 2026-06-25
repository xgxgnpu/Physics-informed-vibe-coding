"""Generate comparison plots for SV-SNN acceleration experiment."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAVE_DIR = os.path.join(SCRIPT_DIR, "saved_data")
FIG_DIR = os.path.join(SCRIPT_DIR, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 14, 'mathtext.fontset': 'stix',
    'axes.labelsize': 16, 'axes.titlesize': 16, 'axes.linewidth': 2.0,
    'xtick.labelsize': 12, 'ytick.labelsize': 12,
    'xtick.major.width': 1.5, 'ytick.major.width': 1.5,
    'lines.linewidth': 2.5, 'legend.fontsize': 10,
    'legend.frameon': True, 'legend.edgecolor': 'black', 'legend.fancybox': False,
    'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'savefig.pad_inches': 0.05,
})

METHOD_ORDER = ['SVSNN_accel', 'SVSNN_original', 'SPINN', 'SIREN', 'FourierPINN', 'PINN']
DISPLAY_NAMES = {
    'SVSNN_accel': 'SV-SNN\n(accelerated)',
    'SVSNN_original': 'SV-SNN\n(original)',
    'SPINN': 'SPINN',
    'SIREN': 'SIREN',
    'FourierPINN': 'FourierPINN',
    'PINN': 'PINN',
}
COLORS = {
    'SVSNN_accel': '#D62728',
    'SVSNN_original': '#FF9896',
    'SPINN': '#1F77B4',
    'SIREN': '#2CA02C',
    'FourierPINN': '#FF7F0E',
    'PINN': '#9467BD',
}

summaries = {}
histories = {}
for name in METHOD_ORDER:
    with open(os.path.join(SAVE_DIR, f"{name}_summary.json")) as f:
        summaries[name] = json.load(f)
    h = np.load(os.path.join(SAVE_DIR, f"{name}_history.npz"))
    histories[name] = {k: h[k] for k in h.files}

# --- Fig 1: Convergence curves ---
fig, ax = plt.subplots(figsize=(10, 6))
for name in METHOD_ORDER:
    h = histories[name]
    epochs = h['eval_epochs']
    errs = h['l2_error']
    ax.semilogy(epochs, errs, color=COLORS[name], label=name.replace('_', ' '), linewidth=2.5)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Relative $L_2$ Error', fontweight='bold')
ax.set_title('Convergence Comparison — 1D Heat ($\\kappa=20\\pi$)', fontweight='bold')
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
for sp in ax.spines.values():
    sp.set_linewidth(2.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig1_convergence.png'))
plt.close(fig)
print("Saved fig1_convergence.png")

# --- Fig 2: Speed vs Accuracy bar chart ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
names = [DISPLAY_NAMES[n] for n in METHOD_ORDER]
colors = [COLORS[n] for n in METHOD_ORDER]

best_l2 = [summaries[n]['best_l2_error'] for n in METHOD_ORDER]
bars1 = ax1.bar(names, best_l2, color=colors, edgecolor='black', linewidth=1.5)
ax1.set_yscale('log')
ax1.set_ylabel('Best Relative $L_2$ Error', fontweight='bold')
ax1.set_title('(a) Accuracy Comparison', fontweight='bold')
for bar, v in zip(bars1, best_l2):
    ax1.text(bar.get_x() + bar.get_width()/2, v * 1.5, f'{v:.1e}',
             ha='center', va='bottom', fontsize=8, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for sp in ax1.spines.values():
    sp.set_linewidth(2.0)

ms_per_epoch = [summaries[n]['ms_per_epoch'] for n in METHOD_ORDER]
bars2 = ax2.bar(names, ms_per_epoch, color=colors, edgecolor='black', linewidth=1.5)
ax2.set_ylabel('ms / epoch', fontweight='bold')
ax2.set_title('(b) Training Speed Comparison', fontweight='bold')
for bar, v in zip(bars2, ms_per_epoch):
    ax2.text(bar.get_x() + bar.get_width()/2, v + 0.1, f'{v:.2f}',
             ha='center', va='bottom', fontsize=9, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)
for sp in ax2.spines.values():
    sp.set_linewidth(2.0)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig2_speed_accuracy.png'))
plt.close(fig)
print("Saved fig2_speed_accuracy.png")

# --- Fig 3: Speedup & Parameter efficiency ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

time_accel = summaries['SVSNN_accel']['total_time_sec']
time_orig = summaries['SVSNN_original']['total_time_sec']
ms_accel = summaries['SVSNN_accel']['ms_per_epoch']
ms_orig = summaries['SVSNN_original']['ms_per_epoch']

labels = ['SV-SNN\n(original)', 'SV-SNN\n(accelerated)']
vals = [ms_orig, ms_accel]
cols = [COLORS['SVSNN_original'], COLORS['SVSNN_accel']]
bars = ax1.bar(labels, vals, color=cols, edgecolor='black', linewidth=1.5, width=0.5)
for bar, v in zip(bars, vals):
    ax1.text(bar.get_x() + bar.get_width()/2, v + 0.1, f'{v:.2f} ms',
             ha='center', va='bottom', fontsize=12, fontweight='bold')
speedup = ms_orig / ms_accel
ax1.set_ylabel('ms / epoch', fontweight='bold')
ax1.set_title(f'(a) SV-SNN Speedup: {speedup:.2f}x', fontweight='bold')
ax1.grid(axis='y', alpha=0.3)
for sp in ax1.spines.values():
    sp.set_linewidth(2.0)

params_list = [summaries[n]['total_params'] for n in METHOD_ORDER]
best_l2_list = [summaries[n]['best_l2_error'] for n in METHOD_ORDER]
for i, name in enumerate(METHOD_ORDER):
    ax2.scatter(params_list[i], best_l2_list[i], color=COLORS[name],
                s=200, edgecolors='black', linewidths=1.5, zorder=5,
                label=name.replace('_', ' '))
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_xlabel('Number of Parameters', fontweight='bold')
ax2.set_ylabel('Best Relative $L_2$ Error', fontweight='bold')
ax2.set_title('(b) Parameter Efficiency', fontweight='bold')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
for sp in ax2.spines.values():
    sp.set_linewidth(2.0)

fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig3_speedup_efficiency.png'))
plt.close(fig)
print("Saved fig3_speedup_efficiency.png")

# --- Fig 4: SV-SNN only: accel vs original convergence ---
fig, ax = plt.subplots(figsize=(8, 5))
for name in ['SVSNN_accel', 'SVSNN_original']:
    h = histories[name]
    ax.semilogy(h['eval_epochs'], h['l2_error'], color=COLORS[name],
                label=name.replace('_', ' '), linewidth=2.5)
ax.set_xlabel('Epoch', fontweight='bold')
ax.set_ylabel('Relative $L_2$ Error', fontweight='bold')
ax.set_title('SV-SNN: Accelerated vs Original Convergence', fontweight='bold')
ax.legend(fontsize=12)
ax.grid(alpha=0.3)
for sp in ax.spines.values():
    sp.set_linewidth(2.0)
fig.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'fig4_svsnn_comparison.png'))
plt.close(fig)
print("Saved fig4_svsnn_comparison.png")

print("\nAll plots saved to:", FIG_DIR)
