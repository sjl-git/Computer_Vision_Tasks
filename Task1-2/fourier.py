import cv2
import matplotlib.pyplot as plt
import numpy as np

##### To-do #####

def fftshift(img):
    '''
    This function should shift the spectrum image to the center.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    height, width = img.shape
    img = np.roll(img, height//2, axis=0)
    img = np.roll(img, width//2, axis=1)
    return img

def ifftshift(img):
    '''
    This function should do the reverse of what fftshift function does.
    You should not use any kind of built in shift function. Please implement your own.
    '''
    height, width = img.shape
    img = np.roll(img, (height + 1)//2, axis=0)
    img = np.roll(img, (width + 1)//2, axis=1)
    return img

def fm_spectrum(img):
    '''
    This function should get the frequency magnitude spectrum of the input image.
    Make sure that the spectrum image is shifted to the center using the implemented fftshift function.
    You may have to multiply the resultant spectrum by a certain magnitude in order to display it correctly.
    '''
    fmSpec = np.fft.fft2(img)
    magSpec = 20*np.log(np.abs(fftshift(fmSpec)))
    return magSpec

def low_pass_filter(img, r=30):
    '''
    This function should return an image that goes through low-pass filter.
    '''
    height, width = img.shape
    spec = fftshift(np.fft.fft2(img))
    for h in range(height):
        for w in range(width):
            if (h - height//2)**2 + (w - width//2)**2 > r**2:
                spec[h, w] = 0.1

    spec = ifftshift(spec)
    filteredImg = np.fft.ifft2(spec).real
    return filteredImg

def high_pass_filter(img, r=20):
    '''
    This function should return an image that goes through high-pass filter.
    '''
    height, width = img.shape
    spec = fftshift(np.fft.fft2(img))
    for h in range(height):
        for w in range(width):
            if (h - height//2)**2 + (w - width//2)**2 < r**2:
                spec[h, w] = 0.1

    spec = ifftshift(spec)
    filteredImg = np.fft.ifft2(spec).real
    return filteredImg

def denoise1(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    height, width = img.shape
    spec = fftshift(np.fft.fft2(img))
    band1 = [70, 80]
    band2 = [110, 120]
    for h in range(height):
        for w in range(width):
            distance = (h - height//2)**2 + (w - width//2)**2
            if (distance < band1[1]**2 and distance > band1[0]**2) or (distance < band2[1]**2 and distance > band2[0]**2):
                spec[h, w] = 0.1

    spec = ifftshift(spec)
    filteredImg = np.fft.ifft2(spec).real
    return filteredImg

def denoise2(img):
    '''
    Use adequate technique(s) to denoise the image.
    Hint: Use fourier transform
    '''
    height, width = img.shape
    spec = fftshift(np.fft.fft2(img))
    band = [30, 40]
    for h in range(height):
        for w in range(width):
            distance = (h - height//2)**2 + (w - width//2)**2
            if (distance < 800 and distance > 600):
                spec[h, w] = 0.1

    spec = ifftshift(spec)
    filteredImg = np.fft.ifft2(spec).real
    return filteredImg

#################

# Extra Credit
def dft2(img):
    '''
    Extra Credit. 
    Implement 2D Discrete Fourier Transform.
    Naive implementation runs in O(N^4).
    '''
    return img

def idft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Discrete Fourier Transform.
    Naive implementation runs in O(N^4). 
    '''
    return img

def fft2(img):
    '''
    Extra Credit. 
    Implement 2D Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    return img

def ifft2(img):
    '''
    Extra Credit. 
    Implement 2D Inverse Fast Fourier Transform.
    Correct implementation runs in O(N^2*log(N)).
    '''
    return img

if __name__ == '__main__':
    img = cv2.imread('task2_filtering.png', cv2.IMREAD_GRAYSCALE)
    noised1 = cv2.imread('task2_noised1.png', cv2.IMREAD_GRAYSCALE)
    noised2 = cv2.imread('task2_noised2.png', cv2.IMREAD_GRAYSCALE)

    low_passed = low_pass_filter(img)
    high_passed = high_pass_filter(img)
    denoised1 = denoise1(noised1)
    denoised2 = denoise2(noised2)

    # save the filtered/denoised images
    cv2.imwrite('low_passed.png', low_passed)
    cv2.imwrite('high_passed.png', high_passed)
    cv2.imwrite('denoised1.png', denoised1)
    cv2.imwrite('denoised2.png', denoised2)

    # draw the filtered/denoised images
    def drawFigure(loc, img, label):
        plt.subplot(*loc), plt.imshow(img, cmap='gray')
        plt.title(label), plt.xticks([]), plt.yticks([])

    drawFigure((2,7,1), img, 'Original')
    drawFigure((2,7,2), low_passed, 'Low-pass')
    drawFigure((2,7,3), high_passed, 'High-pass')
    drawFigure((2,7,4), noised1, 'Noised')
    drawFigure((2,7,5), denoised1, 'Denoised')
    drawFigure((2,7,6), noised2, 'Noised')
    drawFigure((2,7,7), denoised2, 'Denoised')

    drawFigure((2,7,8), fm_spectrum(img), 'Spectrum')
    drawFigure((2,7,9), fm_spectrum(low_passed), 'Spectrum')
    drawFigure((2,7,10), fm_spectrum(high_passed), 'Spectrum')
    drawFigure((2,7,11), fm_spectrum(noised1), 'Spectrum')
    drawFigure((2,7,12), fm_spectrum(denoised1), 'Spectrum')
    drawFigure((2,7,13), fm_spectrum(noised2), 'Spectrum')
    drawFigure((2,7,14), fm_spectrum(denoised2), 'Spectrum')

    plt.show()