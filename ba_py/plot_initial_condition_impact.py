import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "text.usetex": True,          # Use LaTeX for all text
    "font.family": "serif",       # Or any LaTeX-supported font
    "legend.frameon": True,
})



# combined_results = {
#     'initial_noise': [[0,0], [0.5, 0.25], [1.0, 0.5], [1.5, 0.75], [2.0, 1.0], [4.0, 2.0]],
#     'ate': [[0.417094, 1.41608, 2.49765, 0.970215],
#             [0.441105, 1.4232, 2.50803, 0.962721],
#             [0.42998, 1.4058, 2.51497, 0.971655],
#             [0.443136, 1.43074, 2.52679, 0.991855],
#             [0.435093, 1.46285, 2.48604, 0.955731],
#             [0.626338, 1.51342, 2.63685, 0.937372]],
#     'num_iter': [[7, 5, 5, 6],
#                  [6, 7, 6, 7],
#                  [11, 15, 7, 8],
#                  [14, 23, 24, 23],
#                  [26, 26, 27, 27],
#                  [39, 42, 44, 48]]
# }

combined_results = {
    'initial_noise': [[0,0],
                      [1.0, 0.5],
                      [2.0, 1.0],
                      [3.0, 1.5],
                      [4.0, 2.0],
                      [5.0, 2.5]],
    'ate': [[0.417094, 1.41608, 2.49765, 0.970215],
            [0.42998, 1.4058, 2.51497, 0.971655],
            [0.435093, 1.46285, 2.48604, 0.955731],
            [0.469427, 1.3971, 2.54792, 0.987848],
            [0.626338, 1.51342, 2.63685, 0.937372],
            [0.650984, 1.64281, 2.62944, 1.07676]],
    'num_iter': [[7, 5, 5, 6],
                 [11, 15, 7, 8],
                 [26, 26, 27, 27],
                 [21, 34, 29, 25],
                 [39, 42, 44, 48],
                 [79, 61, 95, 82]]
}


n_filled = len(combined_results['ate'])
labels = combined_results['initial_noise'][:n_filled]
avg_ate = [np.mean(v) for v in combined_results['ate']]
avg_iter = [np.mean(v) for v in combined_results['num_iter']]

print(avg_ate)

n_filled = len(combined_results['ate'])
noise_labels = [f'{a} / {b}' for a, b in combined_results['initial_noise'][:n_filled]]
x = np.arange(n_filled)
avg_ate = [np.mean(v) for v in combined_results['ate']]
avg_iter = [np.mean(v) for v in combined_results['num_iter']]


fig, ax1 = plt.subplots(figsize=(6, 3))
color_ate = '#3266ad'
color_iter = '#e07b30'
ax1.plot(x, avg_ate, color=color_ate, marker='o', linewidth=3, label=r'avg ATE')
ax1.set_xlabel(r'initial noise (translation ($m$) / rotation ($^{\circ}$))', fontsize=14)
ax1.set_ylabel(r'ATE', color=color_ate, fontsize=14)
ax1.tick_params(axis='y', labelcolor=color_ate, labelsize=12)
ax2 = ax1.twinx()
ax2.plot(x, avg_iter, color=color_iter, marker='o', linewidth=3,
    linestyle='--', label=r'avg num.\ of iterations')
ax2.set_ylabel(r'num. iterations', color=color_iter, fontsize=14)
ax2.tick_params(axis='y', labelcolor=color_iter, labelsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(noise_labels, rotation=20, ha='center', fontsize=14)
ax1.tick_params(axis='x', labelsize=12)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.set_ylim(1.2, 1.6)
ax2.set_ylim(0, 100)
fig.tight_layout()
plt.savefig('/home/dl/Downloads/drba_init_conditions.pdf', bbox_inches='tight')
plt.show()