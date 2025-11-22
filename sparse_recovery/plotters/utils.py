import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend import Legend
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerLine2D, HandlerTuple


#from mpl_toolkits.axes_grid.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import os

FIGSIZE = (6, 4)
FIGSIZE_SMALL = (6, 4)
FIGSIZE_MEDIUM = (8, 6)
FIGSIZE_LARGE = (15, 10)

LINEWIDTH = 2.0
FONTSIZE = 12

LABEL_FONTSIZE = 20
TICK_LABEL_FONTSIZE = 15

MARKERSIZE = 10

########################################################################################
########################################################################################

def get_training_phases(all_steps, train_accs, test_accs, min_acc=0.05, max_acc=0.99):
    t_1, t_2, t_3, t_4 = None, None, None, None
    if train_accs is not None :
        for t, acc in zip(all_steps, train_accs):
            if acc>min_acc : t_1 = t; break
        for t, acc in zip(all_steps, train_accs):
            if acc>=max_acc : t_2 = t; break
    if test_accs is not None :
        for t, acc in zip(all_steps, test_accs):
            if acc>min_acc : t_3 = t; break
        for t, acc in zip(all_steps, test_accs):
            if acc>=max_acc : t_4 = t; break
    return t_1, t_2, t_3, t_4

def find_closest_step(step, all_steps) :
    if step is None: return None
    all_steps_step_index = {k:v for v, k in enumerate(all_steps)}
    candidates = np.array(list(all_steps_step_index.keys()))
    # Find the closest checkpoint step with acc compute
    closest = candidates[np.abs(candidates - step).argmin()]
    index = all_steps_step_index[closest]
    closest_step = all_steps[index]
    # train_acc = train_accs[index]
    # val_acc = test_accs[index]
    return closest_step

########################################################################################
########################################################################################

def get_twin_axis(
    ax=None, color_1="k", color_2="k", no_twin=False,
    axis = "x",
    linewidth=0.8, # 0.3 # major
    linewidth_minor=0.2, # 0.2 # minor
    alpha=0.7,
    alpha_minor=0.3,
    ) :

    assert axis in ["x", "y"], "axis must be 'x' or  'y'"
    axis_second = "y" if axis == "x" else "x"

    color='black' # major : 'k' 'gray' 'black' ...
    color_minor='gray' # minor
    linestyle="-"
    linestyle_minor='--'

    if ax is None :
        R, C = 1, 1
        #figsize=(C*15, R*10)
        figsize=(C*6, R*4)
        fig, ax1 = plt.subplots(figsize=figsize)
    else :
        ax1 = ax
        fig = None

    ax1.grid(axis=axis, linestyle=linestyle, which='major', color=color, linewidth=linewidth, alpha=alpha)
    ax1.grid(axis=axis, linestyle=linestyle_minor, color=color_minor, which='minor', linewidth=linewidth_minor, alpha=alpha_minor)
    ax1.grid(axis=axis_second, linestyle=linestyle, which='major', color=color_1, linewidth=linewidth, alpha=alpha)
    ax1.grid(axis=axis_second, linestyle=linestyle_minor, color=color_minor, which='minor', linewidth=linewidth_minor, alpha=alpha_minor)
    plt.minorticks_on()

    if no_twin : return fig, ax1, None

    ax2 = ax1.twinx() if axis == "x" else ax1.twiny()    
    ax2.grid(axis=axis_second, linestyle=linestyle, which='major', color=color_2, linewidth=linewidth, alpha=alpha)
    ax2.grid(axis=axis_second, linestyle=linestyle_minor, color=color_minor, which='minor', linewidth=linewidth_minor, alpha=alpha_minor)
    plt.minorticks_on()

    return fig, ax1, ax2

########################################################################################
########################################################################################

def add_legend(element, legend_elements, labels,
              #bbox_to_anchor=(0., 1.02, 1., .102), # top
              #bbox_to_anchor=(0.5, 1.09), # top
              #bbox_to_anchor=(0.5, 1.05), # top, on the line
              loc='best',
               ):

    # legend_elements_train = (Line2D([0], [0], linestyle='-', color=color_train))
    # legend_elements_val = (Line2D([0], [0], linestyle='-', color=color_val))
    # legend_elements = [legend_elements_train, legend_elements_val]
    # labels = [train_label, val_label]

    # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html
    # locations : 'best', 'upper right', 'upper left', 'lower left', 'lower right', 'right', 'center left', 'center right', 'lower center', 'upper center', 'center'
    element.legend(legend_elements, labels,
                    loc=loc,
                    #ncol=2,
                    #bbox_to_anchor=bbox_to_anchor, fancybox=True, shadow=True,
                    handler_map={tuple: HandlerTuple(ndivide=None)}, handlelength=2,
                    fontsize='large',
                )
    

########################################################################################
########################################################################################

def plot_loss_accs(
    statistics, 
    train_test_metrics_names = ["loss"], # "loss", "accuracy", etc.
    other_metrics_names = [], # norms, etc.
    multiple_runs=False, log_x=False, log_y=False, 
    figsize=FIGSIZE, linewidth=LINEWIDTH, fontsize=FONTSIZE,
    fileName=None, filePath=None, show=True
    ):

    metrics_names = train_test_metrics_names + other_metrics_names
    cols = min(5, len(metrics_names))
    rows = (len(metrics_names) + cols - 1) // cols  # Ceiling division to determine number of rows
    fig = plt.figure(figsize=(cols * figsize[0], rows * figsize[1]))
    color_1 = 'tab:blue' # #1f77b4
    color_2 = 'tab:red' # #d62728
    
    same_steps = False
    if multiple_runs :
        all_steps = statistics["all_steps"]
        same_steps = all(len(steps) == len(all_steps[0]) for steps in all_steps) # Check if all runs have the same number of steps
        if same_steps :
            all_steps = np.array(all_steps[0]) + 1e-0 # Add 1e-0 to avoid log(0)
        else :
            all_steps = [np.array(steps) + 1e-0 for steps in all_steps] # Add 1e-0 to avoid log(0)
            color_indices = np.linspace(0, 1, len(all_steps))
            colors = plt.cm.viridis(color_indices)
    else :
        all_steps = np.array(statistics["all_steps"]) + 1e-0

    for i, key in enumerate(train_test_metrics_names) :
        ax = fig.add_subplot(rows, cols, i+1)
        if multiple_runs :
            zs = np.array(statistics["train"][key])
            if same_steps :
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                #ax.errorbar(all_steps, zs_mean, yerr=zs_std, fmt=f'-', color=color_1, label=f"Train", lw=linewidth)
                ax.plot(all_steps, zs_mean, '-', color=color_1, label=f"Train", lw=linewidth)
                ax.fill_between(all_steps, zs_mean-zs_std, zs_mean+zs_std, color=color_1, alpha=0.2)
            else :  
                for j, z in enumerate(zs) :
                    ax.plot(all_steps[j], z, '-', color=colors[j], label=f"Train", lw=linewidth/2)

            zs = np.array(statistics["test"][key])
            if same_steps :
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                ax.plot(all_steps, zs_mean, '-', color=color_2, label=f"Eval", lw=linewidth)
                ax.fill_between(all_steps, zs_mean-zs_std, zs_mean+zs_std, color=color_2, alpha=0.2)
            else :  
                for j, z in enumerate(zs) :
                    ax.plot(all_steps[j], z, '--', color=colors[j], label=f"Eval", lw=linewidth/2)

        else :
            ax.plot(all_steps, statistics["train"][key], "-", color=color_1,  label=f"Train", lw=linewidth) 
            ax.plot(all_steps, statistics["test"][key], "-", color=color_2,  label=f"Eval", lw=linewidth) 

        if log_x : ax.set_xscale('log')
        #if log_y : ax.set_yscale('log')
        if log_y and key=="loss" : ax.set_yscale('log') # No need to log accuracy
        ax.tick_params(axis='y', labelsize='x-large')
        ax.tick_params(axis='x', labelsize='x-large')
        ax.set_xlabel("Training Steps (t)", fontsize=fontsize)
        if key=="accuracy": s = "Accuracy"
        elif key=="loss": s = "Loss"
        else: s = key#.capitalize()

        #ax.set_ylabel(s, fontsize=fontsize)
        ax.set_title(s, fontsize=fontsize)
        ax.grid(True)
        if multiple_runs and (not same_steps) :
            legend_elements = [Line2D([0], [0], color='k', lw=linewidth, linestyle='-', label='Train'),
                            Line2D([0], [0], color='k', lw=linewidth, linestyle='--', label='Eval')]
            ax.legend(handles=legend_elements, fontsize=fontsize)
        else :
            ax.legend(fontsize=fontsize)

    for i, key in enumerate(other_metrics_names, start=len(train_test_metrics_names)) :
        ax = fig.add_subplot(rows, cols, i+1)
        if multiple_runs :
            zs = np.array(statistics[key])
            if same_steps :
                zs_mean, zs_std = np.mean(zs, axis=0), np.std(zs, axis=0)
                #ax.errorbar(all_steps, zs_mean, yerr=zs_std, fmt=f'-', color=color_1, label=None, lw=linewidth)
                ax.plot(all_steps, zs_mean, '-', color=color_1, label=None, lw=linewidth)
                ax.fill_between(all_steps, zs_mean-zs_std, zs_mean+zs_std, color=color_1, alpha=0.2)
            else :  
                for j, z in enumerate(zs) :
                    ax.plot(all_steps[j], z, '-', color=colors[j], label=None, lw=linewidth/2)

        else :
            ax.plot(all_steps, statistics[key], "-", color=color_1,  label=None, lw=linewidth) 

        if log_x : ax.set_xscale('log')
        if log_y : ax.set_yscale('log')
        ax.tick_params(axis='y', labelsize='x-large')
        ax.tick_params(axis='x', labelsize='x-large')
        ax.set_xlabel("Training Steps (t)", fontsize=fontsize)
        s = key#.capitalize()
        #ax.set_ylabel(s, fontsize=fontsize)
        ax.set_title(s, fontsize=fontsize)
        ax.grid(True)
        #ax.legend(fontsize=fontsize)

    ## Adjust layout at the end
    fig.tight_layout()
    ## Adjust spacing manually if needed (left, bottom, right, top, wspace between subplots, hspace between subplots)
    # plt.subplots_adjust(left=0.025, bottom=None, right=0.95, top=None, wspace=0.20, hspace=None)

    if fileName is not None and filePath is not None :
        os.makedirs(filePath, exist_ok=True)
        plt.savefig(f"{filePath}/{fileName}"  + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

    if show : plt.show()
    else : plt.close()