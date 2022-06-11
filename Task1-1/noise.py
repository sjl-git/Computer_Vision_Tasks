import cv2
import numpy as np

## Do not erase or modify any lines already written
## Each noise function should return image with noise

def add_gaussian_noise(image):
    # Use mean of 0, and standard deviation of image itself to generate gaussian noise
    height, width = image.shape
    sigma = np.std(image)
    for i in range(height):
        for j in range(width):
            noise = np.random.normal(scale=sigma)
            image[i][j] += noise
    
    return image

def add_uniform_noise(image):
    # Generate noise of uniform distribution in range [0, standard deviation of image)
    height, width = image.shape
    high = np.std(image)
    for i in range(height):
        for j in range(width):
            noise = np.random.uniform(0, high, 1)[0]
            image[i][j] += noise

    return image

def apply_impulse_noise(image):
    # Implement pepper noise so that 20% of the image is noisy
    height, width = image.shape
    num_pepper = np.ceil(0.2*image.size)
    for n in range(int(num_pepper)):
        i = np.random.randint(0, height-1)
        j = np.random.randint(0, width-1)
        image[i][j] = 0
    return image


def rms(img1, img2):
    # This function calculates RMS error between two grayscale images. 
    # Two images should have same sizes.
    if (img1.shape[0] != img2.shape[0]) or (img1.shape[1] != img2.shape[1]):
        raise Exception("img1 and img2 should have the same sizes.")

    diff = np.abs(img1.astype(np.int32) - img2.astype(np.int32))

    return np.sqrt(np.mean(diff ** 2))


if __name__ == '__main__':
    np.random.seed(0)
    original = cv2.imread('bird.jpg', cv2.IMREAD_GRAYSCALE)
    
    gaussian = add_gaussian_noise(original.copy())
    print("RMS for Gaussian noise:", rms(original, gaussian))
    cv2.imwrite('gaussian.jpg', gaussian)
    
    uniform = add_uniform_noise(original.copy())
    print("RMS for Uniform noise:", rms(original, uniform))
    cv2.imwrite('uniform.jpg', uniform)
    
    impulse = apply_impulse_noise(original.copy())
    print("RMS for Impulse noise:", rms(original, impulse))
    cv2.imwrite('impulse.jpg', impulse)