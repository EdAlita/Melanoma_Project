import cv2 as cv
import numpy as np
from scipy.stats import skew

def gloh_data(image,color_space='HSV'):
    fos = {}
        
    if color_space == 'RGB':
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)
        pass
    elif color_space == 'HSV':
        tmp = cv.cvtColor(image, cv.COLOR_HSV2RGB)
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    elif color_space == 'GRAY':
        pass
    elif color_space == 'LAB':
        tmp = cv.cvtColor(image, cv.COLOR_LAB2RGB)
        gray = cv.cvtColor(image, cv.COLOR_RGB2GRAY)

    else:
        raise NotImplementedError()
    
    if image is None:
        raise ValueError("Image not found or cannot be read.")

    # Convert the image to grayscale
    gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Compute the histogram of the grayscale image
    hist = cv.calcHist([gray_image], [0], None, [256], [0, 256])

    # Normalize the histogram to make it a probability distribution
    hist /= hist.sum()

    # Compute various statistical measures
    mean_value = np.mean(hist)
    variance_value = np.var(hist)
    median_value = np.median(hist)

    fos["mean_value"]=mean_value
    fos["variance_value"]=variance_value
    fos["median_value"]=median_value
    
    return fos
    