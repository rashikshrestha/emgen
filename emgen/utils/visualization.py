import matplotlib
matplotlib.use('Agg')  # use non-interactive backend for ssh connection, else plotting is slow
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import seaborn as sns
import os
import cv2
import subprocess
from emgen.dataset.toy_dataset import get_gt_dino_dataset


def plot_1d_pdf(data, model_samples=None, bins=100, title="Probability Density Function Comparison",
                figsize=(10, 6), save_path=None):
    """
    Plot the probability density function of real data vs. generated samples for 1D data.

    Args:
        data: Real data samples from the dataset (numpy array)
        model_samples: Generated samples from the diffusion model (numpy array)
        bins: Number of histogram bins
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)

    Returns:
        fig: Matplotlib figure object
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Determine data range for consistent visualization
    if model_samples is not None:
        min_val = min(data.min(), model_samples.min())
        max_val = max(data.max(), model_samples.max())
        # Add some padding
        range_padding = (max_val - min_val) * 0.1
        plot_range = (min_val - range_padding, max_val + range_padding)
    else:
        # Add some padding
        range_padding = (data.max() - data.min()) * 0.1
        plot_range = (data.min() - range_padding, data.max() + range_padding)

    # Plot real data histogram with density=True for PDF
    ax.hist(data.flatten(), bins=bins, density=True, alpha=0.7,
            label='Real Data', color='blue', range=plot_range)

    # Plot model samples if provided
    if model_samples is not None:
        ax.hist(model_samples.flatten(), bins=bins, density=True, alpha=0.7,
                label='Generated Samples', color='red', range=plot_range)

    # Add KDE curves for smoother visualization
    if data.size > 1:  # Only if we have enough data points
        sns.kdeplot(data.flatten(), ax=ax, color='darkblue', lw=2, label='_nolegend_')
        if model_samples is not None and model_samples.size > 1:
            sns.kdeplot(model_samples.flatten(), ax=ax, color='darkred', lw=2, label='_nolegend_')

    # Add labels and title
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved 1D PDF plot to {save_path}")

    plt.tight_layout()
    return fig


def plot_2d_pdf(data, model_samples=None, bins=50, cmap='viridis',
                title="2D Probability Density Function", figsize=(15, 6), save_path=None):
    """
    Plot the 2D probability density function of real data vs. generated samples.

    Args:
        data: Real data samples from the dataset (numpy array)
        model_samples: Generated samples from the diffusion model (numpy array)
        bins: Number of histogram bins
        cmap: Colormap for the density plots
        title: Plot title
        figsize: Figure size (width, height)
        save_path: Path to save the figure (optional)

    Returns:
        fig: Matplotlib figure object
    """
    # Ensure data is 2D
    if data.shape[1] != 2:
        raise ValueError("Data must be 2-dimensional for 2D PDF visualization")

    # Create figure based on whether model samples are provided
    if model_samples is None:
        fig, ax = plt.subplots(figsize=(figsize[0] // 2, figsize[1]))
        axes = [ax]  # For consistent indexing later
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Determine common plotting range for consistency
    if model_samples is not None:
        all_data = np.vstack([data, model_samples])
    else:
        all_data = data

    x_min, y_min = all_data.min(axis=0) - 0.5
    x_max, y_max = all_data.max(axis=0) + 0.5
    plot_range = [[x_min, x_max], [y_min, y_max]]

    # Plot real data
    h_real, xedges, yedges, im_real = axes[0].hist2d(
        data[:, 0], data[:, 1],
        bins=bins,
        cmap=cmap,
        range=plot_range,
        density=True,
        norm=Normalize()  # For consistent colormap scaling
    )
    axes[0].set_title("Real Data Distribution")
    axes[0].set_xlabel("Dimension 1")
    axes[0].set_ylabel("Dimension 2")
    fig.colorbar(im_real, ax=axes[0], label="Density")

    # Plot model samples if provided
    if model_samples is not None:
        h_model, _, _, im_model = axes[1].hist2d(
            model_samples[:, 0], model_samples[:, 1],
            bins=bins,
            cmap=cmap,
            range=plot_range,
            density=True,
            norm=Normalize()  # For consistent colormap scaling
        )
        axes[1].set_title("Generated Samples Distribution")
        axes[1].set_xlabel("Dimension 1")
        axes[1].set_ylabel("Dimension 2")
        fig.colorbar(im_model, ax=axes[1], label="Density")

    # Add scatter points on top to show individual samples (with small alpha)
    axes[0].scatter(data[:, 0], data[:, 1], alpha=0.1, s=1, color='white')
    if model_samples is not None:
        axes[1].scatter(model_samples[:, 0], model_samples[:, 1], alpha=0.1, s=1, color='white')

    # Add overall title
    fig.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make room for suptitle

    # Save figure if path is provided
    if save_path:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"Saved 2D PDF plot to {save_path}")

    return fig


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

    #! Plot Particle Trajectory
    plt.figure(figsize=(10,10))
    for i in range(samples.shape[1]):
        plt.plot(samples[:,i,0], samples[:,i,1], linewidth=2, zorder=1, alpha=0.7)

    plt.scatter(samples[0, :, 0], samples[0, :, 1], c='red', marker='o', s=20, label='Initial Position', zorder=2)
    plt.scatter(samples[-1, :, 0], samples[-1, :, 1], c='blue', marker='o', s=20, label='Final Position', zorder=2)
    
    gt_dino_data = get_gt_dino_dataset()
    plt.scatter(gt_dino_data[:,0], gt_dino_data[:,1], c='green', marker='o', s=1, label='GT Distribution', zorder=1)
    
    plt.title("Diffusion Trajectories")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.xlim(-6, 6)
    plt.ylim(-6, 6)
    plt.tight_layout()
    plt.legend()
    plt.savefig(f"{out_dir}/intermediate_traj.pdf")
    plt.close()
        
    #! Make GIF out of saved Images
    command = f"convert -delay 10 -loop 0 {out_dir}/sample_*.png "
    if reverse: command += "-reverse "
    command += f"{out_dir}/intermediate_samples.gif"
    subprocess.run(command, shell=True, check=True)

  
def plot_images_intermediate_samples(samples, out_dir, no_of_samples_to_save, reverse=True):
    """
    Args:
        samples (numpy.ndarray): [timesteps, B, C, H, W] Intermediate Image samples generated during the diffusion process.
    """
    #! Indices to Plot
    no_of_samples = samples.shape[0]  # Number of timesteps
    if no_of_samples_to_save < no_of_samples:
        indices = np.round(np.linspace(0, no_of_samples-1, no_of_samples_to_save)).astype(int)
    else:
        indices = np.arange(no_of_samples)

    #! Make grid of batch images
    reshaped_samples = reshape_samples_for_grid(samples)
    
    #! Plot Individual samples
    for i in indices:
        image_name = f"sample_{no_of_samples-i-1:04}.png" if reverse else f"sample_{i:04}.png"
        cv2.imwrite(f"{out_dir}/{image_name}",  reshaped_samples[i].transpose(1, 2, 0) * 255)
        
    #! Make GIF out of saved Images
    command = f"convert -delay 10 -loop 0 {out_dir}/sample_*.png "
    if reverse: command += "-reverse "
    command += f"{out_dir}/intermediate_samples.gif"
    subprocess.run(command, shell=True, check=True)
    

def reshape_samples_for_grid(samples, aspect_ratio=(9, 16)):
    """
    Reshape samples from [timesteps, B, C, H, W] to [timesteps, C, H*nrows, W*ncolumns]
    while maintaining an approximate aspect ratio for nrows and ncolumns.

    Args:
        samples (numpy.ndarray): Input samples of shape [timesteps, B, C, H, W].
        aspect_ratio (tuple): Desired aspect ratio (nrows:ncolumns), default is (9:16).

    Returns:
        numpy.ndarray: Reshaped samples of shape [timesteps, C, H*nrows, W*ncolumns].
    """
    timesteps, B, C, H, W = samples.shape

    # Calculate nrows and ncolumns to maintain the aspect ratio
    total_cells = B
    aspect_ratio_float = aspect_ratio[0] / aspect_ratio[1]
    nrows = int(np.sqrt(total_cells * aspect_ratio_float))
    ncolumns = int(np.ceil(total_cells / nrows))

    # Ensure the grid can fit all samples
    assert nrows * ncolumns >= B, "Grid size is too small to fit all samples."

    # Create an empty array for the reshaped samples
    reshaped_samples = np.zeros((timesteps, C, H * nrows, W * ncolumns), dtype=samples.dtype)

    # Fill the grid with the samples
    for t in range(timesteps):
        for idx in range(B):
            row = idx // ncolumns
            col = idx % ncolumns
            reshaped_samples[t, :, row * H:(row + 1) * H, col * W:(col + 1) * W] = samples[t, idx]

    return reshaped_samples