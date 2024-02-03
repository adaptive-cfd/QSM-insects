import numpy as np 
import matplotlib.pyplot as plt 
import csv
import debug as db

def converge(file):
    file_to_read = open(file, 'r')
    csvreader = csv.reader(file_to_read)
    info_to_plot = []
    for row in csvreader: 
        current_row = [float(row[0]), float(row[1]), float(row[2])]
        info_to_plot.append(current_row)
    # info_to_plot = np.array(info_to_plot)
    file_to_read.close()
    return np.array(info_to_plot)

# K_final = converge('convergence?/K_final.csv')
# plt.loglog(K_final[:, 0], K_final[:, 1], label='K_final_analytical')
# plt.loglog(K_final[:, 0], K_final[:, 2], label='K_final_numerical')
# plt.title('log-log plot of K_final .vs. number of blades')
# plt.xlabel('number of blades []')
# plt.ylabel('K_final []')
# plt.legend()
# plt.show()

# x0_0 = converge('convergence?/x0[0].csv')
# plt.loglog(x0_0[:, 0], x0_0[:, 1], label='x0[0]_analytical')
# plt.loglog(x0_0[:, 0], x0_0[:, 2], label='x0[0]_numerical')
# plt.title('log-log plot of x0[0] .vs. number of blades')
# plt.xlabel('number of blades []')
# plt.ylabel('x0[0] []')
# plt.legend()
# plt.show()

# x0_1 = converge('convergence?/x0[1].csv')
# plt.loglog(np.abs(x0_1[:, 0]), np.abs(x0_1[:, 1]), label='x0[1]_analytical')
# plt.loglog(np.abs(x0_1[:, 0]), np.abs(x0_1[:, 2]), label='x0[1]_numerical')
# plt.title('log-log plot of x0[1] .vs. number of blades')
# plt.xlabel('number of blades []')
# plt.ylabel('x0[1] []')
# plt.legend()
# plt.show()

# x0_2 = converge('convergence?/x0[2].csv')
# plt.loglog(x0_2[:, 0], x0_2[:, 1], label='x0[2]_analytical')
# plt.loglog(x0_2[:, 0], x0_2[:, 2], label='x0[2]_numerical')
# plt.title('log-log plot of x0[2] .vs. number of blades')
# plt.xlabel('number of blades []')
# plt.ylabel('x0[2] []')
# plt.legend()
# plt.show()

# x0_3 = converge('convergence?/x0[3].csv')
# plt.loglog(np.abs(x0_3[:, 0]), np.abs(x0_3[:, 1]), label='x0[3]_analytical')
# plt.loglog(np.abs(x0_3[:, 0]), np.abs(x0_3[:, 2]), label='x0[3]_numerical')
# plt.title('log-log loglog of x0[3] .vs. number of blades')
# plt.xlabel('number of blades []')
# plt.ylabel('x0[3] []')
# plt.legend()
# plt.show()

K_final_copy = converge('convergence?/K_final_copy.csv')
plt.loglog(K_final_copy[:, 0], K_final_copy[:, 1], label='K_final_copy_analytical')
plt.loglog(K_final_copy[:, 0], K_final_copy[:, 2], label='K_final_copy_numerical')
plt.title('log-log plot of K_final_copy .vs. number of blades')
plt.xlabel('number of blades []')
plt.ylabel('K_final_copy []')
plt.legend()
plt.show()

x0_0_copy = converge('convergence?/x0[0]_copy.csv')
plt.loglog(x0_0_copy[:, 0], x0_0_copy[:, 1], label='x0[0]_copy_analytical')
plt.loglog(x0_0_copy[:, 0], x0_0_copy[:, 2], label='x0[0]_copy_numerical')
plt.title('log-log plot of x0[0]_copy .vs. number of blades')
plt.xlabel('number of blades []')
plt.ylabel('x0[0]_copy []')
plt.legend()
plt.show()

x0_1_copy = converge('convergence?/x0[1]_copy.csv')
plt.loglog(np.abs(x0_1_copy[:, 0]), np.abs(x0_1_copy[:, 1]), label='x0[1]_copy_analytical')
plt.loglog(np.abs(x0_1_copy[:, 0]), np.abs(x0_1_copy[:, 2]), label='x0[1]_copy_numerical')
plt.title('log-log plot of x0[1]_copy .vs. number of blades')
plt.xlabel('number of blades []')
plt.ylabel('x0[1]_copy []')
plt.legend()
plt.show()

x0_2_copy = converge('convergence?/x0[2]_copy.csv')
plt.loglog(x0_2_copy[:, 0], x0_2_copy[:, 1], label='x0[2]_copy_analytical')
plt.loglog(x0_2_copy[:, 0], x0_2_copy[:, 2], label='x0[2]_copy_numerical')
plt.title('log-log plot of x0[2]_copy .vs. number of blades')
plt.xlabel('number of blades []')
plt.ylabel('x0[2]_copy []')
plt.legend()
plt.show()

x0_3_copy = converge('convergence?/x0[3]_copy.csv')
plt.loglog(np.abs(x0_3_copy[:, 0]), np.abs(x0_3_copy[:, 1]), label='x0[3]_copy_analytical')
plt.loglog(np.abs(x0_3_copy[:, 0]), np.abs(x0_3_copy[:, 2]), label='x0[3]_copy_numerical')
plt.title('log-log loglog of x0[3]_copy .vs. number of blades')
plt.xlabel('number of blades []')
plt.ylabel('x0[3]_copy []')
plt.legend()
plt.show()