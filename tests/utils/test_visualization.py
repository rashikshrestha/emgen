from emgen.utils.visualization import *


def test_plot_timeseries_1d_pdf():
    # Create dummy diffusion samples
    timesteps = torch.tensor(list(range(0, 100))[::-1])  #  timesteps
    
    diffusion_samples = []
    for t in timesteps:
        # Generate samples whose mean slowly moves from 0 to 5 and variance increases
        mean = t * 0.5
        std = 0.5 + 0.1 * t
        samples = torch.normal(mean=mean, std=std, size=(1000,))
        diffusion_samples.append(samples)
        
    diffusion_samples = torch.stack(diffusion_samples, dim=0)  # Shape: (100, 1000)
    
    # Call the function
    fig = plot_timeseries_1d_pdf(diffusion_samples, timesteps, invert_x=True,
                                 title="Test Plot: Diffusion Evolution",
                                 figsize=(8, 5),
                                 save_path='this.png'
    )  # No file saving during test
    
    # Assertions (basic)
    assert fig is not None, "Returned figure is None"
    assert isinstance(fig, plt.Figure), "Returned object is not a matplotlib Figure"

    print("Test passed successfully!")
    
    
# test_plot_timeseries_1d_pdf()
