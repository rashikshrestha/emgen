import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

from emgen.dataset.mnist import MNISTDataset
from emgen.generative_model.diffusion.diffusion_model import DiffusionModel
from emgen.generative_model.diffusion.noise_scheduler import NoiseScheduler
from emgen.generative_model.diffusion.diffusion_model_arch import UNetArch
from emgen.utils.memorization import (
    compute_l2_memorization_metric,
    compute_ssim_memorization_metric,
    visualize_nearest_neighbors,
    plot_memorization_score_distribution,
    visualize_diffusion_process_for_memorized_samples
)


def analyze_mnist_memorization(model_path, output_dir, num_samples=50):
    """
    Analyze memorization in a trained diffusion model on MNIST.

    Args:
        model_path: Path to the saved model weights
        output_dir: Directory to save results
        num_samples: Number of samples to generate for analysis
    """
    os.makedirs(output_dir, exist_ok=True)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    dataset = MNISTDataset(train=True)
    train_loader = DataLoader(dataset, batch_size=100, shuffle=False)

    # Initialize model architecture
    diffusion_arch = UNetArch(
        device=device,
        in_channels=1,
        base_channels=64,
        emb_size=128,
        num_down=2,
        weights=model_path
    )

    # Initialize noise scheduler
    noise_scheduler = NoiseScheduler(
        device=device,
        num_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear"
    )

    # Initialize diffusion model
    model = DiffusionModel(
        device=device,
        noise_scheduler=noise_scheduler,
        diffusion_arch=diffusion_arch,
        dataset=dataset
    )

    # Set batch size for generation
    model.train_config.eval_batch_size = 10

    # Generate samples
    print("Generating samples...")
    generated_samples = []
    intermediate_samples_list = []

    for _ in tqdm(range((num_samples + 9) // 10)):
        sample_output = model.sample(get_intermediate_samples=True)
        generated_samples.append(torch.tensor(sample_output['generated_sample'], device=device))
        intermediate_samples_list.append(sample_output['intermediate_samples'])

    # Concatenate batches
    generated_samples = torch.cat(generated_samples, dim=0)[:num_samples]
    intermediate_samples = np.concatenate([s[:, :10] for s in intermediate_samples_list], axis=1)[:, :num_samples]

    print(f"Generated {generated_samples.shape[0]} samples")

    # Load a subset of training data for comparison
    all_training_samples = []
    for batch in train_loader:
        all_training_samples.append(batch[0])
        if len(torch.cat(all_training_samples)) >= 10000:
            break

    training_samples = torch.cat(all_training_samples, dim=0)[:10000].to(device)
    print(f"Loaded {training_samples.shape[0]} training samples for comparison")

    # Compute L2 memorization metric
    print("Computing L2 memorization metric...")
    l2_scores, l2_indices, _ = compute_l2_memorization_metric(
        generated_samples, training_samples
    )

    # Plot L2 memorization scores
    plot_memorization_score_distribution(
        l2_scores, output_dir, metric_name="L2"
    )

    # Visualize nearest neighbors based on L2
    visualize_nearest_neighbors(
        generated_samples, training_samples, l2_indices, l2_scores,
        os.path.join(output_dir, "l2_nearest_neighbors"),
        num_samples=10, is_image=True
    )

    # Visualize diffusion process for top memorized samples
    visualize_diffusion_process_for_memorized_samples(
        intermediate_samples, l2_scores, training_samples, l2_indices,
        os.path.join(output_dir, "l2_diffusion_process"),
        num_samples=3, num_steps=10, is_image=True
    )

    # Compute SSIM memorization metric
    print("Computing SSIM memorization metric...")
    ssim_scores, ssim_indices, _ = compute_ssim_memorization_metric(
        generated_samples, training_samples
    )

    # Plot SSIM memorization scores
    plot_memorization_score_distribution(
        ssim_scores, output_dir, metric_name="SSIM"
    )

    # Visualize nearest neighbors based on SSIM
    visualize_nearest_neighbors(
        generated_samples, training_samples, ssim_indices, ssim_scores,
        os.path.join(output_dir, "ssim_nearest_neighbors"),
        num_samples=10, is_image=True
    )

    # Visualize diffusion process for top memorized samples by SSIM
    visualize_diffusion_process_for_memorized_samples(
        intermediate_samples, ssim_scores, training_samples, ssim_indices,
        os.path.join(output_dir, "ssim_diffusion_process"),
        num_samples=3, num_steps=10, is_image=True
    )

    # Save memorization scores and indices
    torch.save({
        'l2_scores': l2_scores,
        'l2_indices': l2_indices,
        'ssim_scores': ssim_scores,
        'ssim_indices': ssim_indices
    }, os.path.join(output_dir, 'memorization_metrics.pt'))

    print(f"Memorization analysis complete. Results saved to {output_dir}")

    # Print summary statistics
    print("\nMemorization Analysis Summary:")
    print(f"  L2 Mean Score: {l2_scores.mean().item()}")
    print(f"  L2 Max Score: {l2_scores.max().item()}")
    print(f"  L2 Memorized Percentage: {(l2_scores > 0).float().mean().item() * 100:.2f}%")
    print(f"  SSIM Mean Score: {ssim_scores.mean().item()}")
    print(f"  SSIM Max Score: {ssim_scores.max().item()}")
    print(f"  SSIM Memorized Percentage: {(ssim_scores > 0).float().mean().item() * 100:.2f}%")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze memorization in a trained MNIST diffusion model")
    parser.add_argument("model_path", help="Path to the saved model weights")
    parser.add_argument("output_dir", help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=50, help="Number of samples to generate")

    args = parser.parse_args()

    analyze_mnist_memorization(args.model_path, args.output_dir, args.num_samples)

