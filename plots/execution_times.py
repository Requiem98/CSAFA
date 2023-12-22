from libraries import *
import utilities as ut
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

light_cpu_times_hat = ut.read_object("Data/descriptors_data/light_cpu_times_hat.pkl")
heavy_cpu_times_hat = ut.read_object("Data/descriptors_data/heavy_cpu_times_hat.pkl")
light_times_hat = ut.read_object("Data/descriptors_data/light_times_hat.pkl")
heavy_times_hat = ut.read_object("Data/descriptors_data/heavy_times_hat.pkl")
n_samples = ut.read_object("Data/descriptors_data/n_samples.pkl")


def set_spines(axs, idx):
    axs[idx].spines['left'].set_linewidth(2)
    axs[idx].spines['bottom'].set_linewidth(2)
    axs[idx].spines['left'].set_color('black')
    axs[idx].spines['bottom'].set_color('black')
    
    axs[idx].grid(linewidth=3)
    axs[idx].tick_params(axis='both', which='major', labelsize=15)
    
    axs[idx].legend(loc="upper left", fontsize = 15, facecolor="white", framealpha=1.0, edgecolor = "black")


fig, axs = plt.subplots(2, 1, figsize=(10, 15))
fig.tight_layout(pad=8.0)

axs[0].set_title('CPU Execution time', fontsize = 20)
axs[0].set_xlabel('Numbers of descriptors', fontsize=20)
axs[0].set_ylabel('Execution time', fontsize=16)
axs[0].set_xscale('symlog')
axs[0].plot(n_samples,light_cpu_times_hat, linewidth=3, label="size 512")    
axs[0].plot(n_samples,heavy_cpu_times_hat, linewidth=3, label="size 4096")  
set_spines(axs, 0)


axs[1].set_title('Wall Execution time', fontsize = 20)
axs[1].set_xlabel('Numbers of descriptors', fontsize=20)
axs[1].set_ylabel('Execution time', fontsize=20)
axs[1].set_xscale('symlog')
axs[1].plot(n_samples,light_times_hat, linewidth=3, label="size 512")    
axs[1].plot(n_samples,heavy_times_hat, linewidth=3, label="size 4096")  
set_spines(axs, 1)

fig.savefig('plots/execution_times.png', dpi=300)





