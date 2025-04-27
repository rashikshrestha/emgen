from emgen.utils.metrics import *


def test_compute_kl_divergence():
    # 1. Test with identical distributions
    samples1 = torch.randn(10000)
    samples2 = samples1.clone()
    kl_same = compute_kl_divergence(samples1, samples2)
    print(f"KL (same distribution): {kl_same:.6f}")  # Should be close to 0

    # 2. Test with shifted distributions
    samples3 = samples1 + 2.0  # Shifted by 2
    kl_shifted = compute_kl_divergence(samples1, samples3)
    print(f"KL (shifted distribution): {kl_shifted:.6f}")  # Should be > 0

    # 3. Test with different scale
    samples4 = samples1 * 2.0  # Scaled by 2
    kl_scaled = compute_kl_divergence(samples1, samples4)
    print(f"KL (scaled distribution): {kl_scaled:.6f}")  # Should be > 0

    # 4. Test with totally different distributions
    samples5 = torch.randn(10000) + 5  # Completely shifted
    kl_diff = compute_kl_divergence(samples1, samples5)
    print(f"KL (different distribution): {kl_diff:.6f}")  # Should be even larger
  
 
def test_compute_Nd_kl_divergence():
    d = 2
    
    # 1. Generate identical distributions
    p_samples = np.random.randn(10000, d)  # 10k samples, 3D
    q_samples = p_samples.copy()

    kl_same = compute_Nd_kl_divergence(p_samples, q_samples)
    print(f"KL divergence (identical distributions): {kl_same:.6f}")

    # 2. Generate shifted distribution
    q_samples_shifted = p_samples + 2.0
    kl_shifted = compute_Nd_kl_divergence(p_samples, q_samples_shifted)
    print(f"KL divergence (shifted distributions): {kl_shifted:.6f}")

    # 3. Generate scaled distribution
    q_samples_scaled = p_samples * 1.5
    kl_scaled = compute_Nd_kl_divergence(p_samples, q_samples_scaled)
    print(f"KL divergence (scaled distributions): {kl_scaled:.6f}")

    # 4. Generate totally different distribution
    q_samples_diff = np.random.randn(10000, d) + 5.0
    kl_diff = compute_Nd_kl_divergence(p_samples, q_samples_diff)
    print(f"KL divergence (different distributions): {kl_diff:.6f}")


test_compute_kl_divergence()
test_compute_Nd_kl_divergence()