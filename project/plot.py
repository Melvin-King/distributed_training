import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np

def plot(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

# Fill the data points here
if __name__ == '__main__':
    single_mean, single_std = None, None
    device0_mean, device0_std =  None, None
    device1_mean, device1_std =  None, None
    plot([single_mean, device0_mean, device1_mean],
        [single_std, device0_std, device1_std],
        ['Single GPU', 'Data Parallel - GPU0', 'Data Parallel - GPU1'],
        'DP.png')

    
    npp2_mean, npp2_std = None, None
    npp4_mean, npp4_std = None, None
    pp_mean, pp_std = None, None
    plot([npp2_mean, npp4_mean, pp_mean],
        [npp2_std, npp4_std, pp_std],
        ['Naive Pipeline - 2GPUs', 'Naive Pipeline - 4GPUs', 'Improved Pipeline'],
        'PP.png')