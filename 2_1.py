from matplotlib import image
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread

# Load the image
image_path = 'Cross.pgm'
image = imread(image_path)

def apply_ideal_low_pass_filter(image, cutoff_frequency):
    # Perform FFT
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)

    # Create a low-pass filter mask with the cutoff frequency
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    x, y = np.ogrid[:rows, :cols]
    mask = np.sqrt((x - center_row)**2 + (y - center_col)**2) <= cutoff_frequency
    filtered = f_transform_shifted * mask

    # Inverse FFT
    f_ishift = ifftshift(filtered)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

# Convert image to grayscale numpy array if not already
image_np = np.array(image)

# Apply the ideal low-pass filter with different cutoff frequencies
cutoff_frequencies = [10, 30, 50, 70]
fig, axs = plt.subplots(1, len(cutoff_frequencies) + 1, figsize=(10, 5))

# Original image
axs[0].imshow(image_np, cmap='gray')
axs[0].set_title('Original')
axs[0].axis('off')

# Filtered images
for i, cutoff in enumerate(cutoff_frequencies):
    filtered_image = apply_ideal_low_pass_filter(image_np, cutoff)
    axs[i+1].imshow(filtered_image, cmap='gray')
    axs[i+1].set_title(f'Cutoff: {cutoff}')
    axs[i+1].axis('off')

plt.tight_layout()
plt.show()

def apply_gaussian_low_pass_filter(image, sigma):
    # Perform FFT
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)

    # Create a Gaussian low-pass filter
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    x, y = np.ogrid[:rows, :cols]
    distance = np.sqrt((x - center_row)**2 + (y - center_col)**2)
    gaussian_mask = np.exp(-(distance**2) / (2*(sigma**2)))

    # Apply filter
    filtered = f_transform_shifted * gaussian_mask

    # Inverse FFT
    f_ishift = ifftshift(filtered)
    img_back = ifft2(f_ishift)
    img_back = np.abs(img_back)

    return img_back

# Sigma values for Gaussian low-pass filter
sigma_values = [10, 30, 50, 70]
fig, axs = plt.subplots(1, len(sigma_values) + 1, figsize=(10, 5))

# Original image
axs[0].imshow(image_np, cmap='gray')
axs[0].set_title('Original')
axs[0].axis('off')

# Filtered images with Gaussian filter
for i, sigma in enumerate(sigma_values):
    filtered_image = apply_gaussian_low_pass_filter(image_np, sigma)
    axs[i+1].imshow(filtered_image, cmap='gray')
    axs[i+1].set_title(f'Sigma: {sigma}')
    axs[i+1].axis('off')

plt.tight_layout()
plt.show()
