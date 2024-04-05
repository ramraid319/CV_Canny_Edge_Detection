from PIL import Image
import math
import numpy as np

def gauss1d(sigma):             # generate odd(sigma) size array, gaussian filter using given density function
    
    odd = np.ceil(sigma * 6)    # calculate filter size using given formula
    if odd % 2 == 1:
        size = odd
    else:
        size = odd + 1
        
    gauss1d_filter = np.arange(-(size-1)/2, (size-1)/2 + 1, 1)              # generate 1D zero-centered array
    gauss1d_filter = (np.exp(-((gauss1d_filter**2) / (2 * (sigma**2)))))    # calculate gaussian function
    gauss1d_filter /= sum(gauss1d_filter)                                   # normalize the values
    
    return gauss1d_filter

def gauss2d(sigma):             # generate 1D Gaussian filter and outer product
    gauss2d_filter = gauss1d(sigma)
    gauss2d_filter = np.outer(gauss2d_filter, gauss2d_filter)   # outer product for 2D filter
    return gauss2d_filter

def convolve2d(array,filter):
    array_shape = np.shape(array)               # calculate image's shape to generate padding array
    filter_size = np.size(filter)               # calculate filter's size to get "m space" value
    m = int((np.sqrt(filter_size) - 1) / 2)     # calculate "m space" value
    
    padding = np.pad(array, (m,m))              # generate zero padding array
        
    result = np.zeros_like(padding, dtype=np.float32)                       # generate padding_size zeros array
    
    for i in range(m, array_shape[0] + m):                                  # convolution (padded array) * (flipped filter)
        for j in range(m, array_shape[1] + m):
            result[i][j] = np.sum(padding[i-m:i+m+1, j-m:j+m+1] * np.flip(filter))
        
    result = result[m:array_shape[0] + m, m:array_shape[1] + m]             # extract sub-matrix of original image size
    result = result.astype(np.float32)
    return result

def gaussconvolve2d(array,sigma):
    return convolve2d(array, gauss2d(sigma))

def reduce_noise(img):
    img = img.convert('L')                                          # get grayscale image
    img_array = np.asarray(img)
    res = gaussconvolve2d(img_array.astype(np.float32), 1.6)        # get gaussian blured image with (sigma = 1.6)
    return res

def sobel_filters(img):
    x_filter = (1/8)*np.array([[1.0,0.0,-1.0],[2.0,0.0,-2.0],[1.0,0.0,-1.0]], dtype=np.float32)             # make x, y filter and multiply 1/8
    y_filter = (1/8)*np.array([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=np.float32)
    
    dx = convolve2d(img, x_filter)                  # get dx dy images
    dy = convolve2d(img, y_filter)
            
    G = np.zeros_like(dx, dtype=np.float32)         # make dx(dy) size array
    theta = np.zeros_like(dx, dtype=np.float32)
    
    G = np.hypot(dx, dy)                            # sqrt(square(dx) + sqare(dy)) --> get Gradient magnitude image
    theta = np.arctan2(dy, dx)                      # tan^-1 (dy / dx) --> get Direction of gradient image
    
    return (G, theta)

def non_max_suppression(G, theta):
    height, width = G.shape             # get G's shape for operate theta with G
    res = np.zeros_like(G)              # make G' size result array
    
    for i in range(1, height-1):
        for j in range(1, width-1):
            rad = theta[i, j]
            if (np.pi/8 < rad <= 3*np.pi/8) or (-7*np.pi/8 < rad <= -5*np.pi/8):            # 22.5 ~ 67.5 and -157.5 ~ -112.5
                t1 = G[i-1, j+1]                                                            # t1 = Top Right, t2 = Bot Left
                t2 = G[i+1, j-1]
            elif (3*np.pi/8 < rad <= 5*np.pi/8) or (-5*np.pi/8 < rad <= -3*np.pi/8):        # 67.5 ~ 112.5 and -112.5 ~ -67.5
                t1 = G[i-1, j]                                                              # t1 = Top, t2 = Bot
                t2 = G[i+1, j]
            elif (5*np.pi/8 < rad <= 7*np.pi/8) or (-3*np.pi/8 < rad <= -np.pi/8):          # 112.5 ~ 157.5 and -67.5 ~ -22.5
                t1 = G[i-1, j-1]                                                            # t1 = Top Left, t2 = Bot Right
                t2 = G[i+1, j+1]
            else:                                                                           # 157.5 ~ -157.5 and -22.5 ~ 22.5
                t1 = G[i, j-1]                                                              # t1 = Left, t2 = Right
                t2 = G[i, j+1]

            if G[i, j] >= t1 and G[i, j] >= t2:                                 # sharpening edge
                res[i, j] = G[i, j]
            else:
                res[i, j] = 0

    return res

def double_thresholding(img):
    max_val = np.max(img)           # get max value
    min_val = np.min(img)           # get min value
    
    diff = max_val - min_val        # get diff
    T_high = min_val + diff * 0.15  # get threshold (high) value
    T_low = min_val + diff * 0.03   # get threshold (low) value
    
    height, width = img.shape
    res = np.zeros_like(img)
    
    for i in range(0, height):      # convert images with only 3 values, 0, 80, 255 / using threshold values
        for j in range(0, width):
            if(img[i,j] < T_low):
                res[i,j] = 0
            elif(img[i,j] < T_high):
                res[i,j] = 80
            else:
                res[i,j] = 255
            
    return res

def dfs(img, res, i, j, visited=[]):
    # calling dfs on (i, j) coordinate imply that
    #   1. the (i, j) is strong edge
    #   2. the (i, j) is weak edge connected to a strong edge
    # In case 2, it meets the condition to be a strong edge
    # therefore, change the value of the (i, j) which is weak edge to 255 which is strong edge
    res[i, j] = 255

    # mark the visitation
    visited.append((i, j))

    # examine (i, j)'s 8 neighbors
    # call dfs recursively if there is a weak edge
    for ii in range(i-1, i+2) :
        for jj in range(j-1, j+2) :
            if (img[ii, jj] == 80) and ((ii, jj) not in visited) :
                dfs(img, res, ii, jj, visited)

def hysteresis(img):
    res = np.zeros_like(img)
    strong = np.where(img == 255)               # search strong_edge

    for i, j in zip(strong[0], strong[1]):      # apply dfs to all strong_edge
        dfs(img, res, i, j)
    return res

def main():
    RGB_img = Image.open('./iguana.bmp')

    noise_reduced_img = reduce_noise(RGB_img)
    Image.fromarray(noise_reduced_img.astype('uint8')).save('./iguana_blurred.bmp', 'BMP')
    
    g, theta = sobel_filters(noise_reduced_img)
    Image.fromarray(g.astype('uint8')).save('./iguana_sobel_gradient.bmp', 'BMP')
    Image.fromarray(theta.astype('uint8')).save('./iguana_sobel_theta.bmp', 'BMP')

    non_max_suppression_img = non_max_suppression(g, theta)
    Image.fromarray(non_max_suppression_img.astype('uint8')).save('./iguana_non_max_suppression.bmp', 'BMP')

    double_threshold_img = double_thresholding(non_max_suppression_img)
    Image.fromarray(double_threshold_img.astype('uint8')).save('./iguana_double_thresholding.bmp', 'BMP')

    hysteresis_img = hysteresis(double_threshold_img)
    Image.fromarray(hysteresis_img.astype('uint8')).save('./iguana_hysteresis.bmp', 'BMP')
    
main()