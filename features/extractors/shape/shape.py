import numpy as np
import cv2 as cv

def shape_measurements(image,color_space='HSV'):
    fos = {}
    
    Area,Perimeter = 0,0
    
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
    
    ch1, ch2, ch3 = cv.split(image)
    
    ret, thresh = cv.threshold(ch2,0,127,0)
    countours, hierarchy = cv.findContours(thresh, 1, 2)
    
    for cnt in countours:    
        Area =+ cv.contourArea(cnt)
        Perimeter =+ cv.arcLength(cnt,True)
    
    if Area == 0:
        fos['Dispersity'] = 0
        fos['Saturation'] = 0
        fos['Roundness']  = 0 
    else:    
        fos['Dispersity'] = Perimeter**2/Area
        fos['Saturation'] = Area/Perimeter
        fos['Roundness']  = 4*np.pi*Area/Perimeter**2

    return fos