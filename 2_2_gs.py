import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def apply_gaussian_filter(image, cutoff_frequency):
    # Fourier Transform
    f_transform = fft2(image)
    f_transform_shifted = fftshift(f_transform)
    
    rows, cols = image.shape
    center_row, center_col = rows // 2, cols // 2
    
    # Create a Gaussian filter mask
    X, Y = np.ogrid[:rows, :cols]
    distance = np.sqrt((X - center_row)**2 + (Y - center_col)**2)
    sigma = cutoff_frequency / np.sqrt(2*np.log(2))  # Convert cutoff frequency to sigma
    gaussian_mask = np.exp(-distance**2 / (2*sigma**2))
    
    # Apply the Gaussian mask to the shifted Fourier spectrum
    filtered = f_transform_shifted * gaussian_mask
    
    # Inverse Fourier Transform
    f_ishift = ifftshift(filtered)
    img_back = ifft2(f_ishift)
    img_filtered = np.abs(img_back)
    
    return img_filtered

def calculate_rms_error(original, filtered):
    return np.sqrt(np.mean((original - filtered) ** 2))

# Load the images (replace 'path_to_your_image' with the actual paths)
chess_noise = imread('Chess_noise.pgm')
lenna_noise = imread('Lenna_noise.pgm')
chess = imread('Chess.pgm')
lenna = imread('Lenna.pgm')

cutoff_frequencies = [10, 30, 50, 70]

# Adjustments for better inline display
plt.rcParams['figure.figsize'] = [8, 8]  # Adjust based on your display needs

# Process, calculate RMS for each cutoff frequency, and visualize
for cutoff in cutoff_frequencies:
    chess_filtered = apply_gaussian_filter(chess_noise, cutoff)
    lenna_filtered = apply_gaussian_filter(lenna_noise, cutoff)
    
    chess_rms = calculate_rms_error(chess, chess_filtered)
    lenna_rms = calculate_rms_error(lenna, lenna_filtered)
    
    # Print RMS error
    print(f'Chess RMS Error for Gaussian Filter (Cutoff: {cutoff}): {chess_rms:.4f}')
    print(f'Lenna RMS Error for Gaussian Filter (Cutoff: {cutoff}): {lenna_rms:.4f}')
    
    # Visualization
    fig, axs = plt.subplots(2, 3)
    
    axs[0, 0].imshow(chess, cmap='gray')
    axs[0, 0].set_title('Original Chess')
    axs[0, 0].axis('off')
    
    axs[0, 1].imshow(chess_noise, cmap='gray')
    axs[0, 1].set_title('Chess with Noise')
    axs[0, 1].axis('off')
    
    axs[0, 2].imshow(chess_filtered, cmap='gray')
    axs[0, 2].set_title(f'Filtered Chess (Cutoff: {cutoff})')
    axs[0, 2].axis('off')
    
    axs[1, 0].imshow(lenna, cmap='gray')
    axs[1, 0].set_title('Original Lenna')
    axs[1, 0].axis('off')
    
    axs[1, 1].imshow(lenna_noise, cmap='gray')
    axs[1, 1].set_title('Lenna with Noise')
    axs[1, 1].axis('off')
    
    axs[1, 2].imshow(lenna_filtered, cmap='gray')
    axs[1, 2].set_title(f'Filtered Lenna (Cutoff: {cutoff})')
    axs[1, 2].axis('off')
    
    plt.show()
