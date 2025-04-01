import numpy as np
import matplotlib.pyplot as plt

def plot_1d_pdf():
    pass


def plot_2d_pdf():
    pass


def plot_timeseries_1d_pdf():
    #TODO: Ghazal
    pass


def plot_2d_intermediate_samples(samples, out_dir, no_of_samples_to_save, reverse=True):
    """
    Plot the intermediate samples generated during the diffusion process in 2D dataset.
    
    Args:
        samples (numpy.ndarray): [timestep, no_of_datapoints, 2] Intermediate samples generated during the diffusion process.
    """
    #TODO: make this 0 to 1 by normalizing sample even before plotting
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
        
    no_of_samples = samples.shape[0]  # Number of timesteps
    if no_of_samples_to_save < no_of_samples:
        indices = np.round(np.linspace(0, no_of_samples-1, no_of_samples_to_save)).astype(int)
    else:
        indices = np.arange(no_of_samples)

    #! Plot Individual samples
    for i in indices:
        plt.figure(figsize=(5,5))
        plt.scatter(samples[i, :, 0], samples[i, :, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.tight_layout()
        image_name = f"sample_{no_of_samples-i-1:04}.png" if reverse else f"sample_{i:04}.png"
        plt.savefig(f"{out_dir}/{image_name}")
        plt.close()