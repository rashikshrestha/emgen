from emgen.utils.visualization import *


def test_plot_timeseries_1d_pdf():
    # Create dummy diffusion samples
    timesteps = list(range(0, 100))  # 10 timesteps
    
    diffusion_samples = []
    for t in timesteps:
        # Generate samples whose mean slowly moves from 0 to 5 and variance increases
        mean = t * 0.5
        std = 0.5 + 0.1 * t
        samples = torch.normal(mean=mean, std=std, size=(1000, 1))
        diffusion_samples.append(samples)
    
    # Call the function
    fig = plot_timeseries_1d_pdf(diffusion_samples, timesteps,
                                 title="Test Plot: Diffusion Evolution",
                                 figsize=(8, 5),
                                 save_path='this.png'
    )  # No file saving during test
    
    # Assertions (basic)
    assert fig is not None, "Returned figure is None"
    assert isinstance(fig, plt.Figure), "Returned object is not a matplotlib Figure"

    print("Test passed successfully!")
    
    
test_plot_timeseries_1d_pdf()
