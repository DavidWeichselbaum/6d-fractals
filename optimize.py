import sys
import numpy as np
import cma
from skimage.metrics import structural_similarity as ssim
import os
import matplotlib.pyplot as plt
from PIL import Image

from utils.parameters import sample_plane
from utils.fractal import compute_fractal

# Ensure output directory exists
output_dir = "optimization_images"
os.makedirs(output_dir, exist_ok=True)


# Load and normalize the target image
target_image = Image.open(sys.argv[1])
target_image = target_image.convert('L')

target_image = np.array(target_image, dtype=float)
target_image = - target_image
target_image_resolution = (target_image.shape[1], target_image.shape[0])  # width, height
target_image_normalized = (target_image - target_image.min()) / (target_image.max() - target_image.min())

# target_image.show()  # Display the original image
# target_image_uint8 = (target_image_normalized * 255).astype(np.uint8)
# img = Image.fromarray(target_image_uint8)
# img.show()


# Fractal generation wrapper
def generate_fractal_image(params, resolution=(500, 500)):
    """
    Generate a fractal image based on the given parameters.
    :param params: 22 parameters: [u, v, o, center_x, center_y, rotation, scale]
    :return: Fractal image as a 2D array
    """
    u = params[:6]
    v = params[6:12]
    o = params[12:18]
    center = (params[18], params[19])
    rotation = params[20]
    scale = params[21]

    # Sample the plane
    c_z_p_array = sample_plane(u, o, v, center=center, rotation=rotation, scale=scale, resolution=resolution)

    # Compute the fractal
    fractal_image = compute_fractal(c_z_p_array, max_iterations=100)
    return fractal_image


# Loss function for optimization
def loss_function(params, iteration):
    """
    Compute the loss between the generated fractal image and the target image.
    Save the generated fractal image at each iteration.
    """
    fractal_image = generate_fractal_image(params, target_image_resolution)
    # Normalize images for SSIM comparison
    if fractal_image.max() > fractal_image.min():
        fractal_image_normalized = (fractal_image - fractal_image.min()) / (fractal_image.max() - fractal_image.min())
    else:
        fractal_image_normalized = np.zeros_like(fractal_image)  # Handle constant images

    # Save the current fractal image
    plt.imsave(f"{output_dir}/fractal_iter_{iteration:04d}.png", fractal_image_normalized, cmap="inferno")

    # Compute SSIM with data_range specified
    return -ssim(fractal_image_normalized, target_image_normalized, data_range=1.0)


# CMA-ES setup
initial_params = np.random.uniform(-1, 1, 22)  # Random initial guess
sigma = 0.5  # Initial step size

# Run CMA-ES with saving at each iteration
def callback(es):
    iteration = es.countiter
    params = es.result.xbest
    loss_function(params, iteration)

es = cma.CMAEvolutionStrategy(initial_params, sigma)
es.optimize(lambda params: loss_function(params, es.countiter), iterations=100, callback=callback)

# Best solution
best_params = es.result.xbest
print("Best Parameters Found:", best_params)

# Generate and visualize the optimized fractal
optimized_fractal = generate_fractal_image(best_params)
plt.imsave(f"{output_dir}/optimized_fractal.png", optimized_fractal, cmap="viridis")
