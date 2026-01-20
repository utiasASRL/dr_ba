import sys
import numpy as np
import matplotlib.pyplot as plt
import os.path as osp

csv_path = sys.argv[1]
if not osp.exists(csv_path):
    print("CSV file does not exist:", csv_path)
    sys.exit(1)

# Check if a second argument is provided for output path
save_result = True
if len(sys.argv) < 3:
    save_result = False
else:
    output_path = sys.argv[2]

data = np.loadtxt(csv_path, delimiter=",", skiprows=1)

cost = data[:, 0]
ate = data[:, 1]
rmse_x = data[:, 2]
rmse_y = data[:, 3]
rmse_yaw = data[:, 4]

fig, ax1 = plt.subplots()

color_x = 'tab:blue'
color_y = 'tab:orange'
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Translational Error (m)', color=color_x)
ax1.plot(rmse_x, marker='o', color=color_x)
ax1.plot(rmse_y, marker='x', color=color_x)
ax1.tick_params(axis='y', labelcolor=color_x)
# Add legend for x and y
ax1.legend(['Translational Error X', 'Translational Error Y'], loc='upper left')
ax1.grid()
ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:red'
ax2.set_ylabel('Rotational Error (deg)', color=color)  # we already handled the x-label with ax1
ax2.plot(rmse_yaw, marker='o', color=color)
ax2.tick_params(axis='y', labelcolor=color)
fig.tight_layout()  # otherwise the right y-label is slightly clipped

if save_result:
    print("Saving error plots to:", output_path)
    plt.savefig(osp.join(output_path, 'rmse.png'), dpi=300, bbox_inches="tight")

# Plot cost history
# Apply log to cost for better visualization
plt.figure()
plt.plot(cost, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Cost')
plt.title('Cost History')
plt.grid()
if save_result:
    plt.savefig(osp.join(output_path, 'cost.png'), dpi=300, bbox_inches="tight")

# Plot ATE history
plt.figure()
plt.plot(ate, marker='o')
plt.xlabel('Iteration')
plt.ylabel('ATE (m)')
plt.title('ATE History')
plt.grid()
if save_result:
    plt.savefig(osp.join(output_path, 'ate.png'), dpi=300, bbox_inches="tight")
plt.show()

plt.close()
