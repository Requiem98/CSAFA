from libraries import *
import utilities as ut
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


light_cpu_times_hat = ut.read_object("Data/descriptors_data/light_cpu_times_hat.pkl")
heavy_cpu_times_hat = ut.read_object("Data/descriptors_data/heavy_cpu_times_hat.pkl")
light_times_hat = ut.read_object("Data/descriptors_data/light_times_hat.pkl")
heavy_times_hat = ut.read_object("Data/descriptors_data/heavy_times_hat.pkl")
n_samples = ut.read_object("Data/descriptors_data/n_samples.pkl")



vp_cpu = np.abs((light_cpu_times_hat*100)/heavy_cpu_times_hat - 100)
vp_wall = np.abs((light_times_hat*100)/heavy_times_hat - 100)





def set_spines(axs, idx):
    axs[idx].spines['left'].set_linewidth(2)
    axs[idx].spines['bottom'].set_linewidth(2)
    axs[idx].spines['left'].set_color('black')
    axs[idx].spines['bottom'].set_color('black')
    
    axs[idx].grid(linewidth=3)
    axs[idx].tick_params(axis='both', which='major', labelsize=15)


fig, axs = plt.subplots(2, 1, figsize=(10, 15))
fig.tight_layout(pad=8.0)

axs[0].set_title('Variance percentage of CPU Execution time', fontsize = 20)
axs[0].set_xlabel('Numbers of descriptors', fontsize=20)
axs[0].set_ylabel('Percentage', fontsize=16)
#axs[0].set_xscale('symlog')
axs[0].plot(n_samples,vp_cpu, linewidth=3)    
set_spines(axs, 0)


axs[1].set_title('Variance percentage of Wall Execution time', fontsize = 20)
axs[1].set_xlabel('Numbers of descriptors', fontsize=20)
axs[1].set_ylabel('Percentage', fontsize=20)
#axs[1].set_xscale('symlog')
axs[1].plot(n_samples,vp_wall, linewidth=3)
set_spines(axs, 1)

fig.savefig('plots/vp_execution_times.png', dpi=300)
