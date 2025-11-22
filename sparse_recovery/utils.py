import numpy as np

def sample_iterations_for_plotting(
    train_steps,
    t1,
    t2,
    T,
    include_first=True,
    include_last=True,
    sampling_strategy='linear',
    base=10
):
    """
    Samples T steps from train_steps, ensuring inclusion of first, t₁, t₂, and last steps.
    Additional steps are sampled based on the specified strategy ('linear' or 'log').

    Parameters:
    - train_steps (list or iterable): List of training step identifiers (e.g., iteration numbers).
    - t1 (int or None): Memorization step. Must be in train_steps if not None.
    - t2 (int or None): Generalization step. Must be in train_steps if not None.
    - T (int): Total number of steps to sample.
    - include_first (bool): Whether to include the first step. Default is True.
    - include_last (bool): Whether to include the last step. Default is True.
    - sampling_strategy (str): 'linear' for uniform sampling, 'log' for logarithmic sampling.
    - base (float): Base for logarithmic sampling. Default is 10.

    Returns:
    - sampled_steps (list): List of sampled step identifiers.
    """
    if T <= 0:
        raise ValueError("T must be a positive integer.")

    train_steps = list(train_steps)  # Ensure it's a list
    total_steps = len(train_steps)

    if total_steps == 0:
        raise ValueError("train_steps is empty.")

    # Initialize the set of essential steps
    essential_steps = set()

    if include_first:
        essential_steps.add(train_steps[0])
    if t1 is not None and t1 in train_steps:
        essential_steps.add(t1)
    if t2 is not None and t2 in train_steps:
        essential_steps.add(t2)
    if include_last:
        essential_steps.add(train_steps[-1])

    # Number of additional steps needed
    additional = T - len(essential_steps)

    if additional <= 0:
        # If T is less than or equal to the number of essential steps, return essential steps
        sampled_steps = sorted(essential_steps, key=lambda x: train_steps.index(x))
        return sampled_steps[:T]  # In case T < len(essential_steps)

    # Determine the remaining steps to sample
    remaining_steps = [step for step in train_steps if step not in essential_steps]

    if not remaining_steps:
        # If no remaining steps, return the essential steps
        sampled_steps = sorted(essential_steps, key=lambda x: train_steps.index(x))
        return sampled_steps

    # Sample additional steps based on the specified strategy
    if sampling_strategy == 'linear':
        # Uniformly sample additional steps
        indices = np.linspace(0, len(remaining_steps) - 1, num=additional + 2, dtype=int)[1:-1]
    elif sampling_strategy == 'log':
        # Logarithmically sample additional steps
        if len(remaining_steps) < 2:
            # Not enough points for logarithmic sampling, fallback to linear
            indices = np.linspace(0, len(remaining_steps) - 1, num=additional + 2, dtype=int)[1:-1]
        else:
            # Apply log scaling
            log_min = 0
            log_max = np.log(len(remaining_steps)) / np.log(base)
            log_indices = np.linspace(log_min, log_max, num=additional + 2)
            indices = np.floor(base ** log_indices).astype(int) - 1
            # Ensure indices are within bounds
            indices = np.clip(indices, 0, len(remaining_steps) - 1)
            # Remove potential duplicates
            indices = np.unique(indices)
            # If not enough samples due to uniqueness, sample additional
            if len(indices) < additional:
                extra = additional - len(indices)
                extra_indices = np.linspace(0, len(remaining_steps) - 1, num=extra + 2, dtype=int)[1:-1]
                indices = np.unique(np.concatenate((indices, extra_indices)))
                indices = indices[:additional]
    else:
        raise ValueError("sampling_strategy must be either 'linear' or 'log'.")

    sampled_additional = [remaining_steps[i] for i in indices]

    # Combine essential and additional steps
    sampled_steps = list(essential_steps) + sampled_additional

    # Remove duplicates and sort according to train_steps order
    sampled_steps = sorted(set(sampled_steps), key=lambda x: train_steps.index(x))

    # If we have more than T due to overlap, trim the list
    if len(sampled_steps) > T:
        # Prioritize essential steps by ensuring they are included
        essential_sorted = sorted(essential_steps, key=lambda x: train_steps.index(x))
        additional_sorted = [step for step in sampled_steps if step not in essential_steps]
        # Fill up to T
        sampled_steps = essential_sorted + additional_sorted[:T - len(essential_sorted)]

    sampled_steps = sorted(sampled_steps)
    return sampled_steps

########################################################################################
########################################################################################

def select_log_space(iterations, T, base=10):
    """
    Select T elements from the 'iterations' list, equally spaced in log scale.

    Args:
        iterations (list or array-like): List of iteration values (x-axis).
        T (int): Number of elements to select.

    Returns:
        list: T elements selected from 'iterations'.
    """
    # Convert to numpy array for easier manipulation
    iterations = np.array(iterations)

    # Ensure the iterations are sorted
    iterations = np.sort(iterations)

    # Take logarithm of the indices
    log_indices = np.logspace(0, np.emath.logn(base, len(iterations) - 1), num=T, base=base, dtype=int)

    # Remove duplicate indices (if logspace generates overlaps)
    log_indices = np.unique(log_indices)

    # Select corresponding elements
    selected_elements = iterations[log_indices]

    return selected_elements, log_indices


########################################################################################
########################################################################################

def plot_t1_t2(ax, t_1, t_2, log_x, log_y, plot_Delta=True):
    for t in [t_1, t_2] :
        if t is None : continue
        ax.axvline(x=t, ymin=0.01, ymax=1., color='b', linestyle='--', lw=1.)
        ax.plot([t, t], [0, 0], 'o', color='b')

    for t, lab in zip([t_1, t_2], ['t_1', 't_2']) :
        if t is None : continue
        a = -0.05 # bottom
        #a = 1.15 # top
        if log_y :
            a = 1.0/10000 # bottom
            #a = 1.13 # top
        ax.text(t, a, f"${lab}$", ha='center', va='top', color="b", fontsize=20)

    ## Arrow
    if (t_1 is not None) and (t_2 is not None) and plot_Delta :
        y_position = 0.5
        if log_y :
            y_position = 0.5 / 100
        # Drawing a two-sided arrow
        ax.annotate('', xy=(t_1, y_position), xytext=(t_2, y_position), arrowprops=dict(arrowstyle='<->', lw=2))
        # Adding a label above the arrow
        mid_x = (t_1 + t_2) / 2
        if log_x : mid_x = np.exp((np.log(t_1) + np.log(t_2)) / 2)
        ax.text(mid_x, y_position, r'$\Delta t$', va='bottom', ha='center', fontsize=20)

########################################################################################
########################################################################################

def find_memorization_generalization_steps(
    train_errors,
    test_errors,
    train_steps,
    train_threshold=1e-4,
    test_threshold=1e-4
):
    """
    Identifies the memorization and generalization steps based on error thresholds.

    Parameters:
    - train_errors (list of float): Training error values per step.
    - test_errors (list of float): Testing error values per step.
    - train_steps (list or iterable): Identifiers for each training step.
    - train_threshold (float): Threshold for training error to identify t_1.
    - test_threshold (float): Threshold for testing error to identify t_2.

    Returns:
    - t_1 (int or None): The first step where training error < train_threshold.
    - t_2 (int or None): The first step where testing error < test_threshold.
    """

    # Validate inputs
    if not (len(train_errors) == len(test_errors) == len(train_steps)):
        raise ValueError("All input lists must have the same length.")

    t_1 = None
    t_2 = None

    for step, train_err, test_err in zip(train_steps, train_errors, test_errors):
        if t_1 is None and train_err <= train_threshold:
            t_1 = step
        if t_2 is None and test_err <= test_threshold:
            t_2 = step
        if t_1 is not None and t_2 is not None:
            break  # Both steps found

    return t_1, t_2

########################################################################################
########################################################################################

def find_stable_step_final_value(
    steps,
    errors,
    K=10,                 # number of points to average at the end
    tolerance_fraction=0.05,  # 5% tolerance around final plateau
    M=5                  # require 5 consecutive steps inside tolerance
):
    """
    Return the first step at which the error is 'close enough' to its
    final plateau value for M consecutive steps.

    Parameters
    ----------
    steps : array-like
        1D array of step indices (same length as errors).
    errors : array-like
        1D array of error values (same length as steps).
        Must be ordered in ascending step order.
    K : int
        Number of data points from the end to define the final plateau error.
    tolerance_fraction : float
        Fraction of the final plateau error used to define an absolute tolerance.
        If the final plateau error is E_final, the tolerance is tolerance_fraction * E_final.
    M : int
        Number of consecutive points we need inside that tolerance before we declare stability.

    Returns
    -------
    stable_step : float or None
        The step value (from `steps`) where the error first becomes stable,
        or None if it never meets the criterion.
    stable_index : int or None
        The corresponding index into `steps`/`errors`.
    """
    if len(steps) != len(errors):
        raise ValueError("steps and errors must have the same length.")
    if len(errors) < K:
        raise ValueError("Not enough data points to take an average of the last K points.")

    # Convert to numpy
    steps_arr = np.array(steps)
    errors_arr = np.array(errors)

    # 1) Compute final plateau error as average of the last K points
    E_final = np.mean(errors_arr[-K:])

    # 2) Compute absolute tolerance
    #    e.g. if E_final=1e-4 and tolerance_fraction=0.05 => tol = 5e-6
    tol = tolerance_fraction * E_final

    # 3) Scan from left, looking for first time the error is within [E_final ± tol]
    #    for M consecutive steps. Typically, we only need "below (E_final + tol)",
    #    because the curve is presumably decreasing, but we use a symmetrical check
    #    if you want to handle all cases.
    consec = 0
    stable_index = None

    lower_bound = E_final - tol
    upper_bound = E_final + tol

    for i in range(len(errors_arr)):
        e = errors_arr[i]

        # If you only care about being "close but not above" E_final, you might do:
        # if e <= (E_final + tol):
        #     ...
        # here we do symmetrical, in case e < E_final.

        if (e >= lower_bound) and (e <= upper_bound):
            consec += 1
        else:
            consec = 0

        if consec >= M:
            stable_index = i - M + 1
            break

    if stable_index is not None:
        return steps_arr[stable_index], stable_index
    else:
        return None, None
   

########################################################################################
########################################################################################

if __name__ == "__main__":

    iterations = np.arange(0, 10**5+1, 100)
    T=4
    sampled_steps = sample_iterations_for_plotting(train_steps = iterations, t1=100, t2=500, T=T+2, include_first=True, include_last=True, sampling_strategy='linear', base=10)
    sampled_steps = sample_iterations_for_plotting(train_steps = iterations, t1=100, t2=500, T=T+2, include_first=True, include_last=True, sampling_strategy='log', base=10)
    print(sampled_steps)

    selected, log_indices = select_log_space(iterations, T, base=10)
    print(selected)