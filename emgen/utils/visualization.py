import numpy as np
import matplotlib.pyplot as plt

def plot_1d_pdf():
    pass


def plot_2d_pdf():
    pass


def plot_timeseries_1d_pdf():
    #TODO: Ghazal
    pass


def plot_2d_intermediate_samples(samples, out_dir):
    """
    Plot the intermediate samples generated during the diffusion process in 2D dataset.
    
    Args:
        samples (numpy.ndarray): [timestep, no_of_datapoints, 2] Intermediate samples generated during the diffusion process.
    """
    #TODO: make this 0 to 1 by normalizing sample even before plotting
    xmin, xmax = -6, 6
    ymin, ymax = -6, 6
        
    no_of_samples = samples.shape[0]  # Number of timesteps
    #! Plot Individual samples
    for i in range(no_of_samples):
        plt.figure(figsize=(5,5))
        plt.scatter(samples[i, :, 0], samples[i, :, 1])
        plt.xlim(xmin, xmax)
        plt.ylim(ymin, ymax)
        plt.tight_layout()
        plt.savefig(f"{out_dir}/sample_{i:04}.png")
        plt.close()