import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
from scipy.interpolate import interp2d

import os
DIR_PATH_FIGURES="../figures"

def can_find_exponent(base, n):
    """Return an integer i such that base^i = n"""
    if n <= 0 or base <= 0:
        return None  # n and base must be positive
    
    log_result = np.emath.logn(base, n)
    
    # Check if the log result is an integer (or very close to one)
    if np.isclose(log_result, round(log_result)):
        return int(round(log_result))
    else:
        return None

def find_a_and_i(base, n):
    """Return two integers (a, i) such that a * base^i = n."""
    if n <= 0 or base <= 0:
        return None, None  # n and base must be positive
    
    # Start by finding the logarithm of n with respect to base
    log_result = np.emath.logn(base, n)
    i = int(np.floor(log_result))  # We take the floor of the log to find the closest integer i
    
    # Calculate the corresponding a
    a = n / (base ** i)
    
    # Check if a is an integer
    if np.isclose(a, round(a)) and a > 0:
        return int(round(a)), i
    else:
        return None, None

def select_elements_with_step(L, step):
    """
    Select elements from a list with a specified step, ensuring the first and last elements are included.

    Parameters:
    L (list): The input list to select elements from.
    step (int): The step size between selected elements.

    Returns:
    list: A list of elements selected from the input list, with the specified step, including the first and last elements.
    """
    # Select elements separated by the step
    selected_elements = list(L[::step])
    
    # Ensure the last element is included if it's not already
    if L[-1] not in selected_elements:
        selected_elements.append(L[-1])
    
    return selected_elements

def custom_imshow(
    img_data, ax=None, fig=None, add_text=False, n_decimals=2,
    hide_ticks_and_labels=False,
    xticklabels=None, yticklabels=None, filter_step_xticks=1, filter_step_yticks=1, log_x=False, log_y=False, base=10,
    rotation_x=0, rotation_y=0, x_label=None, y_label=None, 
    colormesh_kwarg={}, # e.g. colormesh_kwarg = {"shading":'auto', "cmap":'viridis'}
    # Use LogNorm to apply a logarithmic scale
    # colormesh_kwarg={"shading":'auto', "cmap":'viridis', 'norm':LogNorm(vmin=img_data.min(), vmax=img_data.max())},
    imshow_kwarg={},
    colorbar=True, colorbar_label=None,
    label_fontsize=20,
    ticklabel_fontsize=15,
    text_fontsize=15,
    show=True, fileName=None, dpf=None,
    factor_shape=2,
    use_imshow = False,
    ) :

    """"
    Custom plt.imshow / plt.pcolormesh
    This helps to plot the heatmap as a function of many hyperparameters
    """


    H_, W_ = img_data.shape
    if (ax is None) and (fig is None) :
        L, C = 1, 1
        #L_, C_ = L*10, C*15
        #L_, C_ = L*4, C*6
        L_ = L*4
        C_ = int((W_/(factor_shape*H_))*L_)
        figsize=(C_, L_)
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(L, C, 1)

    ########################################################################################################
    ########################################################################################################

    #if not log_x and not log_y:
    if use_imshow :
        img = ax.imshow(img_data, **imshow_kwarg)
    else :    
        if log_x : 
            #x = np.logspace(0, np.log10(W_), W_)-1
            x = np.emath.logn(base, np.arange(W_)+1) 
        else :
            x = np.arange(W_)
        if log_y : 
            #y = np.logspace(0, np.log10(H_), H_)-1
            y = np.emath.logn(base, np.arange(H_)+1) 
        else:
            y = np.arange(H_)

        X, Y = np.meshgrid(x, y)
        interpolation_kind = imshow_kwarg.get('interpolation', None)
        if interpolation_kind is None :
            img = ax.pcolormesh(X, Y, img_data, **colormesh_kwarg)
        else :
            #assert interpolation_kind in ['linear', 'cubic', 'quintic']
            f = interp2d(X, Y, img_data, kind=interpolation_kind)
            new_img_data = f(x, y)
            img = ax.pcolormesh(X, Y, new_img_data, **colormesh_kwarg)

    ########################################################################################################
    ########################################################################################################

    if add_text :
        for (j, i), label in np.ndenumerate(img_data):
            ax.text(i, j, round(label, n_decimals), ha='center', va='center', fontsize=text_fontsize)

    ########################################################################################################
    ########################################################################################################

    # Hide the ticks and labels
    if hide_ticks_and_labels:
        ax.tick_params(left=False, right=False , labelleft=False, labelbottom=False, bottom=False)

    if xticklabels is not None :
        assert len(xticklabels) == W_
        xticks_positions = np.arange(W_)
        if log_x : 
            xticks_positions = np.emath.logn(base, xticks_positions+1) 

            ### TODO
            new_xticklabels = []
            for label in xticklabels :
                try :
                    label = int(label)
                except ValueError: # If not integer, skip
                    new_xticklabels.append("")
                    continue
                i = can_find_exponent(base, label)
                new_xticklabels.append(f"${base}^{i}$" if i is not None else "")

                # a, i = find_a_and_i(base, label)
                # #s = f"{a}\\times" if i is not None and a!=1 else ""
                # s = f"{a}\cdot" if a is not None and a!=1 else ""
                # new_xticklabels.append(f"${s}{base}^{i}$" if i is not None else "")

            xticklabels = new_xticklabels
            ### End TODO
        else :
            xticks_positions = select_elements_with_step(xticks_positions, step=filter_step_xticks)
            xticklabels = select_elements_with_step(xticklabels, step=filter_step_xticks)
        ax.set_xticks(xticks_positions)
        ax.set_xticklabels(xticklabels, rotation=rotation_x, fontsize=ticklabel_fontsize)
        ax.tick_params(axis="x")#, bottom=True, top=False, labelbottom=False, labeltop=True)
    if x_label : ax.set_xlabel(x_label, fontsize=label_fontsize)


    if yticklabels is not None :
        assert len(yticklabels) == H_
        yticks_positions = np.arange(H_)
        if log_y : 
            yticks_positions = np.emath.logn(base, yticks_positions+1) 

            ### TODO
            new_yticklabels = []
            for label in yticklabels :
                try :
                    label = int(label)
                except ValueError: # If not integer, skip
                    new_yticklabels.append("")
                    continue
                i = can_find_exponent(base, label)
                new_yticklabels.append(f"${base}^{i}$" if i is not None else "")

            yticklabels = new_yticklabels
            ### End TODO
        else :
            yticks_positions = select_elements_with_step(yticks_positions, step=filter_step_yticks)
            yticklabels = select_elements_with_step(yticklabels, step=filter_step_yticks)
        ax.set_yticks(yticks_positions)
        ax.set_yticklabels(yticklabels, rotation=rotation_y, fontsize=ticklabel_fontsize)
        ax.tick_params(axis="y")#, bottom=True, top=False, labelbottom=True, labeltop=False)
    if y_label : ax.set_ylabel(y_label, fontsize=label_fontsize)

    ########################################################################################################
    ########################################################################################################

    if colorbar :
        # Create a divider for the existing axes instance
        padding = 0.05
        #padding = 0.02
        divider = make_axes_locatable(ax)
        cax_cb = divider.append_axes("right", size="5%", pad=padding)
        # Add a color bar to the new axes
        cbar = fig.colorbar(img, ax=ax, cax=cax_cb, fraction=0.046, pad=padding, aspect=20)
        # Add label
        cbar.set_label(colorbar_label, rotation=270, labelpad=15, fontsize=label_fontsize)
        # Change the font size of the color bar ticks
        cbar.ax.tick_params(labelsize=ticklabel_fontsize)  # Set the desired font size here

    ########################################################################################################
    ########################################################################################################

    if fileName is not None :
        if dpf is None : dpf=DIR_PATH_FIGURES
        os.makedirs(dpf, exist_ok=True)
        plt.savefig(f"{dpf}/{fileName}"  + '.pdf', dpi=300, bbox_inches='tight', format='pdf')

    if show : plt.show()
    #else : plt.close()

    ########################################################################################################
    ########################################################################################################

    return img

if __name__ == "__main__":

    M, N = 5, 101
    img_data = np.random.normal(size=(M, N))
    xticklabels = range(1, N+1)
    yticklabels = range(1, M+1)
    

    rows, cols = 1, 1
    figsize=(15, 10)
    figsize=(cols*figsize[0], rows*figsize[1])
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(rows, cols, 1)

    img = custom_imshow(
        img_data, ax=ax, fig=fig, add_text=False,
        hide_ticks_and_labels=False, xticklabels=xticklabels, yticklabels=yticklabels, 
        filter_step_xticks=10, filter_step_yticks=1, log_x=False, log_y=False, base=10,
        rotation_x=0, rotation_y=0, 
        x_label="x label", y_label="y label",
        colormesh_kwarg={"shading":'auto', "cmap":'viridis'},
        imshow_kwarg={},
        colorbar=True, colorbar_label='colorbar label',
        show=True, fileName=None, dpf=None,
        use_imshow=False,
    )