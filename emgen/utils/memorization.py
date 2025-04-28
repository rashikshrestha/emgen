import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import os
from typing import Tuple, List, Dict, Optional, Union
from skimage.metrics import structural_similarity as ssim
import cv2
from tqdm import tqdm


def compute_l2_distances(generated_samples: torch.Tensor, training_samples: torch.Tensor) -> torch.Tensor:
    """
    Compute L2 distances between generated samples and training samples.

    Args:
        generated_samples: Tensor of shape [B_gen, ...] containing generated samples
        training_samples: Tensor of shape [B_train, ...] containing training samples

    Returns:
        Tensor of shape [B_gen, B_train] containing L2 distances
    """
    # Flatten spatial dimensions for each sample
    gen_flat = generated_samples.view(generated_samples.shape[0], -1)
    train_flat = training_samples.view(training_samples.shape[0], -1)

    # Compute pairwise distances
    distances = torch.cdist(gen_flat, train_flat, p=2.0)

    return distances


def find_nearest_neighbors(distances: torch.Tensor, k: int = 50) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Find k nearest neighbors for each generated sample based on distances.

    Args:
        distances: Tensor of shape [B_gen, B_train] containing pairwise distances
        k: Number of nearest neighbors to find

    Returns:
        Tuple of (nearest_indices, nearest_distances) containing:
            - nearest_indices: Tensor of shape [B_gen, k] with indices of nearest neighbors
            - nearest_distances: Tensor of shape [B_gen, k] with distances to nearest neighbors
    """
    # Ensure k is not larger than the number of training samples
    k = min(k, distances.shape[1])

    # Find k nearest neighbors
    topk_values, topk_indices = torch.topk(distances, k=k, dim=1, largest=False)

    return topk_indices, topk_values


def compute_l2_memorization_metric(
        generated_samples: torch.Tensor,
        training_samples: torch.Tensor,
        k: int = 50,
        alpha: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the L2-based memorization metric according to Equation 8 in the paper.

    Args:
        generated_samples: Tensor of shape [B_gen, ...] containing generated samples
        training_samples: Tensor of shape [B_train, ...] containing training samples
        k: Number of nearest neighbors to use for normalization
        alpha: Scaling constant

    Returns:
        Tuple of (memorization_scores, nearest_indices, nearest_distances) containing:
            - memorization_scores: Tensor of shape [B_gen] with memorization scores
            - nearest_indices: Tensor of shape [B_gen] with indices of nearest neighbors
            - nearest_distances: Tensor of shape [B_gen] with distances to nearest neighbors
    """
    # Compute pairwise L2 distances
    distances = compute_l2_distances(generated_samples, training_samples)

    # Find nearest neighbors
    nn_indices, nn_distances = find_nearest_neighbors(distances, k)

    # Get the closest neighbor for each generated sample
    closest_indices = nn_indices[:, 0]
    closest_distances = nn_distances[:, 0]

    # Compute average distance to k nearest neighbors
    avg_distances = torch.mean(nn_distances, dim=1)

    # Compute memorization scores according to Equation 8
    memorization_scores = -closest_distances / (alpha * avg_distances)

    return memorization_scores, closest_indices, closest_distances


def compute_ssim_memorization_metric(
        generated_samples: torch.Tensor,
        training_samples: torch.Tensor,
        k: int = 50,
        alpha: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute the SSIM-based memorization metric (adaptation of Equation 8).

    Args:
        generated_samples: Tensor of shape [B_gen, C, H, W] containing generated samples
        training_samples: Tensor of shape [B_train, C, H, W] containing training samples
        k: Number of nearest neighbors to use for normalization
        alpha: Scaling constant

    Returns:
        Tuple of (memorization_scores, nearest_indices, similarity_values) containing:
            - memorization_scores: Tensor of shape [B_gen] with memorization scores
            - nearest_indices: Tensor of shape [B_gen] with indices of nearest neighbors
            - similarity_values: Tensor of shape [B_gen] with similarity to nearest neighbors
    """
    device = generated_samples.device
    num_gen = generated_samples.shape[0]
    num_train = training_samples.shape[0]

    # Convert tensors to numpy for SSIM computation
    gen_np = generated_samples.cpu().numpy()
    train_np = training_samples.cpu().numpy()

    # Initialize similarity matrix
    similarity_matrix = torch.zeros((num_gen, num_train), device=device)

    # Compute SSIM between each generated sample and all training samples
    for i in tqdm(range(num_gen), desc="Computing SSIM"):
        for j in range(num_train):
            # Ensure proper shape for SSIM computation
            gen_img = gen_np[i, 0] if gen_np.shape[1] == 1 else gen_np[i].transpose(1, 2, 0)
            train_img = train_np[j, 0] if train_np.shape[1] == 1 else train_np[j].transpose(1, 2, 0)

            # Compute SSIM
            similarity = ssim(gen_img, train_img, data_range=1.0)
            similarity_matrix[i, j] = similarity

    # Find k most similar training samples (highest SSIM)
    topk_values, topk_indices = torch.topk(similarity_matrix, k=k, dim=1, largest=True)

    # Get the most similar sample
    most_similar_indices = topk_indices[:, 0]
    most_similar_values = topk_values[:, 0]

    # Compute average similarity to k most similar samples
    avg_similarity = torch.mean(topk_values, dim=1)

    # Compute memorization scores - invert for consistency with L2 metric
    # Note: We negate the ratio to maintain consistency (more negative = less memorization)
    memorization_scores = -most_similar_values / (alpha * avg_similarity)

    return memorization_scores, most_similar_indices, most_similar_values


def visualize_nearest_neighbors(
        generated_samples: torch.Tensor,
        training_samples: torch.Tensor,
        indices: torch.Tensor,
        scores: torch.Tensor,
        save_path: str,
        num_samples: int = 10,
        is_image: bool = True
):
    """
    Visualize generated samples alongside their nearest neighbors from the training set.

    Args:
        generated_samples: Tensor of shape [B_gen, ...] containing generated samples
        training_samples: Tensor of shape [B_train, ...] containing training samples
        indices: Tensor of shape [B_gen] with indices of nearest neighbors
        scores: Tensor of shape [B_gen] with memorization scores
        save_path: Path to save visualizations
        num_samples: Number of samples to visualize
        is_image: Whether the data is image data (True) or 2D data (False)
    """
    os.makedirs(save_path, exist_ok=True)

    # Sort by memorization score (most memorized first)
    sorted_indices = torch.argsort(scores, descending=True)

    # Select samples to visualize
    vis_indices = sorted_indices[:num_samples]

    if is_image:
        # Visualize image data
        fig, axes = plt.subplots(num_samples, 2, figsize=(8, 2 * num_samples))

        for i, idx in enumerate(vis_indices):
            # Get generated sample and its nearest neighbor
            gen_sample = generated_samples[idx]
            train_idx = indices[idx]
            train_sample = training_samples[train_idx]
            mem_score = scores[idx].item()

            # Convert to numpy and reshape if needed
            if gen_sample.dim() == 3:  # [C, H, W]
                gen_img = gen_sample.cpu().numpy().transpose(1, 2, 0)
                train_img = train_sample.cpu().numpy().transpose(1, 2, 0)

                # Handle grayscale images
                if gen_img.shape[2] == 1:
                    gen_img = gen_img[:, :, 0]
                    train_img = train_img[:, :, 0]
            else:  # Already in [H, W] format
                gen_img = gen_sample.cpu().numpy()
                train_img = train_sample.cpu().numpy()

            # Plot generated sample
            axes[i, 0].imshow(gen_img, cmap='gray' if gen_img.ndim == 2 else None)
            axes[i, 0].set_title(f"Generated")
            axes[i, 0].axis('off')

            # Plot nearest neighbor
            axes[i, 1].imshow(train_img, cmap='gray' if train_img.ndim == 2 else None)
            axes[i, 1].set_title(f"NN (Score: {mem_score:.4f})")
            axes[i, 1].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "nearest_neighbors.png"))
        plt.close()
    else:
        # Visualize 2D data
        fig, axes = plt.subplots(2, num_samples // 2 + num_samples % 2, figsize=(4 * num_samples // 2, 8))
        axes = axes.flatten()

        for i, idx in enumerate(vis_indices):
            # Get generated sample and its nearest neighbor
            gen_sample = generated_samples[idx].cpu().numpy()
            train_idx = indices[idx].item()
            train_sample = training_samples[train_idx].cpu().numpy()
            mem_score = scores[idx].item()

            # Plot generated sample and its nearest neighbor
            axes[i].scatter(gen_sample[0], gen_sample[1], color='blue', label='Generated', s=100, zorder=2)
            axes[i].scatter(train_sample[0], train_sample[1], color='red', label='Nearest', s=100, zorder=2)

            # Draw a line connecting them
            axes[i].plot([gen_sample[0], train_sample[0]], [gen_sample[1], train_sample[1]],
                         'k--', alpha=0.5, zorder=1)

            # Set title and limits
            axes[i].set_title(f"Score: {mem_score:.4f}")
            axes[i].legend()

            # Add grid and set equal aspect
            axes[i].grid(alpha=0.3)
            axes[i].set_aspect('equal')

        plt.tight_layout()
        plt.savefig(os.path.join(save_path, "nearest_neighbors_2d.png"))
        plt.close()


def plot_memorization_score_distribution(
        scores: torch.Tensor,
        save_path: str,
        metric_name: str = "L2"
):
    """
    Plot the distribution of memorization scores.

    Args:
        scores: Tensor of shape [B] containing memorization scores
        save_path: Path to save visualizations
        metric_name: Name of the metric used
    """
    os.makedirs(save_path, exist_ok=True)

    scores_np = scores.cpu().numpy()

    plt.figure(figsize=(10, 6))
    plt.hist(scores_np, bins=50, alpha=0.7)
    plt.axvline(x=0, color='r', linestyle='--', label='Memorization Threshold')
    plt.xlabel("Memorization Score")
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {metric_name} Memorization Scores")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f"{metric_name.lower()}_memorization_distribution.png"))
    plt.close()

    # Also log statistics
    with open(os.path.join(save_path, f"{metric_name.lower()}_memorization_stats.txt"), 'w') as f:
        f.write(f"Memorization Score Statistics ({metric_name}):\n")
        f.write(f"Mean: {np.mean(scores_np):.4f}\n")
        f.write(f"Median: {np.median(scores_np):.4f}\n")
        f.write(f"Min: {np.min(scores_np):.4f}\n")
        f.write(f"Max: {np.max(scores_np):.4f}\n")
        f.write(f"Std: {np.std(scores_np):.4f}\n")
        f.write(f"Percentage of memorized samples (score > 0): {np.mean(scores_np > 0) * 100:.2f}%\n")


def visualize_diffusion_process_for_memorized_samples(
        intermediate_samples: np.ndarray,
        memorization_scores: torch.Tensor,
        training_samples: torch.Tensor,
        nearest_indices: torch.Tensor,
        save_path: str,
        num_samples: int = 3,
        num_steps: int = 10,
        is_image: bool = True
):
    """
    Visualize the diffusion process for samples with high memorization scores.

    Args:
        intermediate_samples: Array of shape [T, B, ...] containing samples at each timestep
        memorization_scores: Tensor of shape [B] with memorization scores
        training_samples: Tensor of shape [B_train, ...] containing training samples
        nearest_indices: Tensor of shape [B] with indices of nearest neighbors
        save_path: Path to save visualizations
        num_samples: Number of samples to visualize
        num_steps: Number of diffusion steps to show
        is_image: Whether the data is image data (True) or 2D data (False)
    """
    os.makedirs(save_path, exist_ok=True)

    # Sort by memorization score (most memorized first)
    sorted_indices = torch.argsort(memorization_scores, descending=True)

    # Select samples to visualize
    vis_indices = sorted_indices[:num_samples].cpu().numpy()

    # Select timesteps to visualize
    total_steps = intermediate_samples.shape[0]
    if num_steps < total_steps:
        step_indices = np.linspace(0, total_steps - 1, num_steps, dtype=int)
    else:
        step_indices = np.arange(total_steps)

    if is_image:
        # Visualize image data
        for s_idx, sample_idx in enumerate(vis_indices):
            fig, axes = plt.subplots(1, len(step_indices) + 1, figsize=(3 * (len(step_indices) + 1), 3))

            # Plot diffusion steps
            for t_idx, timestep in enumerate(step_indices):
                img = intermediate_samples[timestep, sample_idx]

                # Handle dimensionality
                if img.ndim == 3:  # [C, H, W]
                    img = img.transpose(1, 2, 0)
                    # Handle grayscale images
                    if img.shape[2] == 1:
                        img = img[:, :, 0]

                axes[t_idx].imshow(img, cmap='gray' if img.ndim == 2 else None)
                axes[t_idx].set_title(f"Step {timestep}")
                axes[t_idx].axis('off')

            # Plot nearest training sample
            train_idx = nearest_indices[sample_idx].item()
            train_img = training_samples[train_idx].cpu().numpy()

            if train_img.ndim == 3:  # [C, H, W]
                train_img = train_img.transpose(1, 2, 0)
                # Handle grayscale images
                if train_img.shape[2] == 1:
                    train_img = train_img[:, :, 0]

            axes[-1].imshow(train_img, cmap='gray' if train_img.ndim == 2 else None)
            axes[-1].set_title(f"NN (Score: {memorization_scores[sample_idx]:.4f})")
            axes[-1].axis('off')

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"diffusion_process_sample_{s_idx}.png"))
            plt.close()
    else:
        # Visualize 2D data
        for s_idx, sample_idx in enumerate(vis_indices):
            # Create a figure with a grid of plots (timesteps) and one extra for the nearest neighbor
            fig, axes = plt.subplots(2, (len(step_indices) + 1) // 2 + ((len(step_indices) + 1) % 2 > 0),
                                     figsize=(15, 6))
            axes = axes.flatten()

            # Find appropriate axis limits by considering all timesteps
            all_points = []
            for timestep in step_indices:
                all_points.append(intermediate_samples[timestep, sample_idx])
            all_points = np.array(all_points)

            # Add the nearest neighbor to the points for determining limits
            train_idx = nearest_indices[sample_idx].item()
            train_sample = training_samples[train_idx].cpu().numpy()
            all_points = np.vstack([all_points, train_sample])

            # Calculate axis limits with some padding
            x_min, y_min = np.min(all_points, axis=0) - 0.5
            x_max, y_max = np.max(all_points, axis=0) + 0.5

            # Plot diffusion steps
            for i, timestep in enumerate(step_indices):
                if i >= len(axes):
                    break

                point = intermediate_samples[timestep, sample_idx]

                # Plot the point
                axes[i].scatter(point[0], point[1], color='blue', label=f'Step {timestep}', s=100)

                # Set consistent limits
                axes[i].set_xlim(x_min, x_max)
                axes[i].set_ylim(y_min, y_max)
                axes[i].set_title(f"Step {timestep}")
                axes[i].grid(alpha=0.3)
                axes[i].set_aspect('equal')

            # Plot the nearest neighbor in the last subplot
            axes[-1].scatter(train_sample[0], train_sample[1], color='red',
                             label=f'NN (Score: {memorization_scores[sample_idx]:.4f})', s=100)
            axes[-1].set_xlim(x_min, x_max)
            axes[-1].set_ylim(y_min, y_max)
            axes[-1].set_title(f"Nearest Neighbor")
            axes[-1].grid(alpha=0.3)
            axes[-1].set_aspect('equal')

            # Add a main title to the figure
            plt.suptitle(f"Diffusion Process for Sample {s_idx} (Score: {memorization_scores[sample_idx]:.4f})")

            plt.tight_layout()
            plt.savefig(os.path.join(save_path, f"diffusion_process_sample_{s_idx}_2d.png"))
            plt.close()


def compute_memorization_across_training(
        model_checkpoints: List[str],
        dataset_loader,
        sample_fn,
        save_path: str,
        num_samples: int = 50,
        k: int = 50,
        alpha: float = 0.5,
        is_image: bool = True
):
    """
    Compute and visualize memorization metrics across different training checkpoints.

    Args:
        model_checkpoints: List of paths to model checkpoints at different training stages
        dataset_loader: DataLoader for the training dataset
        sample_fn: Function that takes a model path and returns generated samples
        save_path: Path to save visualizations
        num_samples: Number of samples to generate for each checkpoint
        k: Number of nearest neighbors to use for normalization
        alpha: Scaling constant for memorization metrics
        is_image: Whether the data is image data (True) or 2D data (False)
    """
    os.makedirs(save_path, exist_ok=True)

    # Load training samples
    training_samples = []
    for batch in dataset_loader:
        if isinstance(batch, tuple):
            # Handle case where dataset returns (data, label)
            training_samples.append(batch[0])
        else:
            training_samples.append(batch)

        if len(torch.cat(training_samples)) >= 10000:
            break

    training_samples = torch.cat(training_samples, dim=0)[:10000]
    print(f"Loaded {training_samples.shape[0]} training samples for comparison")

    # Initialize lists to store metrics for each checkpoint
    checkpoint_ids = []
    l2_means = []
    l2_maxes = []
    l2_memorized_pcts = []
    ssim_means = []
    ssim_maxes = []
    ssim_memorized_pcts = []

    # Process each checkpoint
    for i, checkpoint_path in enumerate(model_checkpoints):
        print(f"Processing checkpoint {i + 1}/{len(model_checkpoints)}: {checkpoint_path}")
        checkpoint_id = os.path.basename(os.path.dirname(checkpoint_path))
        checkpoint_ids.append(checkpoint_id)

        # Generate samples using the provided function
        generated_samples = sample_fn(checkpoint_path, num_samples)

        # Compute L2 memorization metric
        l2_scores, _, _ = compute_l2_memorization_metric(
            generated_samples, training_samples, k=k, alpha=alpha
        )

        # Store L2 metrics
        l2_means.append(l2_scores.mean().item())
        l2_maxes.append(l2_scores.max().item())
        l2_memorized_pcts.append((l2_scores > 0).float().mean().item() * 100)

        # Compute SSIM memorization metric if using image data
        if is_image:
            ssim_scores, _, _ = compute_ssim_memorization_metric(
                generated_samples, training_samples, k=k, alpha=alpha
            )

            # Store SSIM metrics
            ssim_means.append(ssim_scores.mean().item())
            ssim_maxes.append(ssim_scores.max().item())
            ssim_memorized_pcts.append((ssim_scores > 0).float().mean().item() * 100)

    # Plot memorization metrics across training
    plt.figure(figsize=(15, 10))

    # Plot for L2 metric
    plt.subplot(2, 2, 1)
    plt.plot(checkpoint_ids, l2_means, marker='o', label='Mean Score')
    plt.plot(checkpoint_ids, l2_maxes, marker='s', label='Max Score')
    plt.axhline(y=0, color='r', linestyle='--', label='Memorization Threshold')
    plt.xlabel("Training Checkpoint")
    plt.ylabel("L2 Memorization Score")
    plt.title("L2 Memorization Scores Across Training")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 2)
    plt.plot(checkpoint_ids, l2_memorized_pcts, marker='o', color='purple')
    plt.xlabel("Training Checkpoint")
    plt.ylabel("Percentage (%)")
    plt.title("Percentage of Memorized Samples (L2)")
    plt.grid(alpha=0.3)
    plt.xticks(rotation=45)

    # Plot for SSIM metric if image data
    if is_image:
        plt.subplot(2, 2, 3)
        plt.plot(checkpoint_ids, ssim_means, marker='o', label='Mean Score')
        plt.plot(checkpoint_ids, ssim_maxes, marker='s', label='Max Score')
        plt.axhline(y=0, color='r', linestyle='--', label='Memorization Threshold')
        plt.xlabel("Training Checkpoint")
        plt.ylabel("SSIM Memorization Score")
        plt.title("SSIM Memorization Scores Across Training")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)

        plt.subplot(2, 2, 4)
        plt.plot(checkpoint_ids, ssim_memorized_pcts, marker='o', color='purple')
        plt.xlabel("Training Checkpoint")
        plt.ylabel("Percentage (%)")
        plt.title("Percentage of Memorized Samples (SSIM)")
        plt.grid(alpha=0.3)
        plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, "memorization_across_training.png"))
    plt.close()

    # Save metrics to CSV
    import csv
    with open(os.path.join(save_path, "memorization_metrics_across_training.csv"), 'w', newline='') as f:
        writer = csv.writer(f)

        if is_image:
            writer.writerow(['Checkpoint', 'L2 Mean', 'L2 Max', 'L2 Memorized (%)',
                             'SSIM Mean', 'SSIM Max', 'SSIM Memorized (%)'])
            for i in range(len(checkpoint_ids)):
                writer.writerow([checkpoint_ids[i], l2_means[i], l2_maxes[i], l2_memorized_pcts[i],
                                 ssim_means[i], ssim_maxes[i], ssim_memorized_pcts[i]])
        else:
            writer.writerow(['Checkpoint', 'L2 Mean', 'L2 Max', 'L2 Memorized (%)'])
            for i in range(len(checkpoint_ids)):
                writer.writerow([checkpoint_ids[i], l2_means[i], l2_maxes[i], l2_memorized_pcts[i]])

