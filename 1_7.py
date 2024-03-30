import numpy as np
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fftshift
import cv2



# Define a function to perform convolution in the spatial domain
def spatial_domain_convolution(image, kernel):
    # Kernel needs to be flipped for convolution
    kernel_flipped = np.flipud(np.fliplr(kernel))
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2
    
    # Pad the image with zeros on all sides
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')
    
    # Initialize the output array
    output = np.zeros_like(image)
    
    # Perform the convolution using the flipped kernel
    for i in range(image_height):
        for j in range(image_width):
            # Extract the current region of interest
            region = padded_image[i:i+kernel_height, j:j+kernel_width]
            output[i, j] = np.sum(region * kernel_flipped)
    
    return output

# Define a function to perform convolution in the frequency domain
def frequency_domain_convolution(image, kernel):
    # Get the size of the image and kernel
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    
    # Pad the kernel to be the same size as the image
    padded_kernel = np.zeros((image_height, image_width))
    padded_kernel[:kernel_height, :kernel_width] = kernel
    # Shift the kernel to the center
    padded_kernel = fftshift(padded_kernel)
    
    # Compute the FFT of the image and the padded kernel
    image_fft = fft2(image)
    kernel_fft = fft2(padded_kernel)
    
    # Perform element-wise multiplication
    convolved_fft = image_fft * kernel_fft
    
    # Compute the inverse FFT to get the convolved image
    convolved_image = np.real(ifft2(convolved_fft))
    
    return convolved_image

# Define a 3x3 averaging kernel
kernel = np.ones((3, 3), np.float32) / 9.0

# Read the chess image
chess_image_path = 'Chess.pgm'
chess_image = cv2.imread(chess_image_path, cv2.IMREAD_GRAYSCALE)

# Perform spatial domain convolution
spatial_blurred = spatial_domain_convolution(chess_image, kernel)

# Perform frequency domain convolution
frequency_blurred = frequency_domain_convolution(chess_image, kernel)

# Normalize the output images
spatial_blurred_normalized = np.clip(spatial_blurred, 0, 255).astype(np.uint8)
frequency_blurred_normalized = np.clip(frequency_blurred, 0, 255).astype(np.uint8)

# Plot the results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(spatial_blurred_normalized, cmap='gray')
axes[0].set_title('Spatial Domain Blur')
axes[0].axis('off')

axes[1].imshow(frequency_blurred_normalized, cmap='gray')
axes[1].set_title('Frequency Domain Blur')
axes[1].axis('off')

plt.tight_layout()
plt.show()
