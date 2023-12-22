from libraries import *
import utilities as ut
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

n_samples = ut.read_object("Data/descriptors_data/n_samples.pkl")
l_memorysize = np.array(ut.read_object("Data/descriptors_data/l_memorysize.pkl"))/1000
h_memorysize = np.array(ut.read_object("Data/descriptors_data/h_memorysize.pkl"))/1000


def set_spines(axs, idx):
    axs.spines['left'].set_linewidth(2)
    axs.spines['bottom'].set_linewidth(2)
    axs.spines['left'].set_color('black')
    axs.spines['bottom'].set_color('black')
    
    axs.grid(linewidth=3)
    axs.tick_params(axis='both', which='major', labelsize=15)

    axs.legend(loc="upper left", fontsize = 15, facecolor="white", framealpha=1.0, edgecolor = "black")


fig, axs = plt.subplots(1, 1, figsize=(10, 15))
fig.tight_layout(pad=8.0)

axs.set_xlabel('Numbers of descriptors', fontsize=20)
axs.set_ylabel('Memory size of descriptor matrix (GB)', fontsize=16)
axs.set_xscale('symlog')
axs.plot(n_samples,l_memorysize, linewidth=3, label="size 512")    
axs.plot(n_samples,h_memorysize, linewidth=3, label="size 4096")  
set_spines(axs, 0)

fig.savefig('plots/memory_size.png', dpi=300)
