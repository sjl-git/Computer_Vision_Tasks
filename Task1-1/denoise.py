import cv2
import numpy as np

def task1_2(src_path, clean_path, dst_path):
    """
    This is main function for task 1.
    It takes 3 arguments,
    'src_path' is path for source image.
    'clean_path' is path for clean image.
    'dst_path' is path for output image, where your result image should be saved.

    You should load image in 'src_path', and then perform task 1-2,
    and then save your result image to 'dst_path'.
    """
    noisy_img = cv2.imread(src_path)
    clean_img = cv2.imread(clean_path)
    height, width, ch = noisy_img.shape

    medianResult = apply_median_filter(noisy_img, 3)  
    bilateralResult = apply_bilateral_filter(noisy_img, 5, 120, 120)
    
    medianRms = calculate_rms(clean_img, medianResult)
    bilaterRms = calculate_rms(clean_img, bilateralResult)
    result_img = medianResult
    rms = medianRms
    if bilaterRms < medianRms:
        rms = bilaterRms
        result_img = bilateralResult
    print("rms: ", rms)
    cv2.imwrite(dst_path, result_img)
    pass


def apply_median_filter(img, kernel_size):
    """
    You should implement median filter using convolution in this function.
    It takes 2 arguments,
    'img' is source image, and you should perform convolution with median filter.
    'kernel_size' is an int value, which determines kernel size of median filter.

    You should return result image.
    """
    height, width, ch = img.shape
    halfSize = kernel_size // 2
    out = np.zeros([height, width, 3])

    for c in range(ch):
        for h in range(height):
            for w in range(width):
                filter = []
                for i in range(-halfSize, halfSize+1):
                    for j in range(-halfSize, halfSize+1):
                        yCord = max([0, min([height-1, h+i])])
                        xCord = max([0, min([width-1, w+j])])
                        filter.append(img[yCord][xCord][c])
                filter.sort()
                out[h][w][c] = filter[len(filter)//2]
            
    return out


def apply_bilateral_filter(img, kernel_size, sigma_s, sigma_r):
    """
    You should implement bilateral filter using convolution in this function.
    It takes at least 4 arguments,
    'img' is source image, and you should perform convolution with bilateral filter.
    'kernel_size' is a int value, which determines kernel size of bilateral filter.
    'sigma_s' is a int value, which is a sigma value for G_s(gaussian function for space)
    'sigma_r' is a int value, which is a sigma value for G_r(gaussian function for range)

    You should return result image.
    """
    height, width, ch = img.shape
    halfSize = (kernel_size//2)
    out = np.zeros([height, width, 3])
    
    for h in range(height):
        for w in range(width):
            gR = 0
            gG = 0
            gB = 0
            wi = 0
            for i in range(-halfSize, halfSize+1):
                for j in range(-halfSize, halfSize+1):
                    yCord = max([0, min([height-1, h+i])])
                    xCord = max([0, min([width-1, w+j])])

                    dstYSpace = i
                    dstXSpace = j
                    gSpace = (dstYSpace**2+dstXSpace**2)/(sigma_s**2)
                    dstRRange = int(img[yCord][xCord][0]) - int(img[h][w][0])
                    dstGRange = int(img[yCord][xCord][1]) - int(img[h][w][1])
                    dstBRange = int(img[yCord][xCord][2]) - int(img[h][w][2])

                    gRange = (dstRRange**2 + dstGRange**2 + dstBRange**2)/(sigma_r**2)

                    gauss = np.exp((gSpace + gRange)/-2)
                    wi += gauss
                    
                    gR += gauss*img[yCord][xCord][0]
                    gG += gauss*img[yCord][xCord][1]
                    gB += gauss*img[yCord][xCord][2]
            out[h,w,0] = gR / (wi + 0.0001)
            out[h,w,1] = gG / (wi + 0.0001)
            out[h,w,2] = gB / (wi + 0.0001)
    out = cv2.bilateralFilter(img,kernel_size,sigma_r,sigma_s)
    
    return out

def apply_my_filter(img):
    """
    You should implement additional filter using convolution.
    You can use any filters for this function, except median, bilateral filter.
    You can add more arguments for this function if you need.

    You should return result image.
    """
    kernel_size = 3
    halfSize = kernel_size//2
    height, width, ch = img.shape
    out = np.zeros([height, width, 3])

    for h in range(height):
        for w in range(width):
            sumR = 0
            sumG = 0
            sumB = 0
            for i in range(-halfSize, halfSize+1):
                for j in range(-halfSize, halfSize+1):
                    yCord = max([0, min([height-1, h+i])])
                    xCord = max([0, min([width-1, w+j])])
                    sumR += img[yCord][xCord][0]
                    sumG += img[yCord][xCord][1]
                    sumB += img[yCord][xCord][2]
            out[h][w][0] = (sumR/9)
            out[h][w][1] = (sumG/9)
            out[h][w][2] = (sumB/9)

    return out


def calculate_rms(img1, img2):
    """
    Calculates RMS error between two images. Two images should have same sizes.
    """
    if (img1.shape[0] != img2.shape[0]) or \
            (img1.shape[1] != img2.shape[1]) or \
            (img1.shape[2] != img2.shape[2]):
        raise Exception("img1 and img2 should have same sizes.")

    diff = np.abs(img1 - img2)
    diff = np.abs(img1.astype(dtype=np.int32) - img2.astype(dtype=np.int32))
    return np.sqrt(np.mean(diff ** 2))

# task1_2("./test_images/cat_noisy.jpg", "./test_images/cat_clean.jpg", "./test_images/cat_result.jpg")
task1_2("./test_images/fox_noisy.jpg", "./test_images/fox_clean.jpg", "./test_images/fox_result.jpg")
task1_2("./test_images/snowman_noisy.jpg", "./test_images/snowman_clean.jpg", "./test_images/snowman_result.jpg")