import numpy as np
import cv2 as cv

def shaṕe_measurements(image):
    fos = {}
    Area,Perimeter = 0,0
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    ret, thresh = cv.thresholding(gray,127,255,0)
    countours, hierarchy = cv.findCountours(cv.findContours(thresh, 1, 2))
    
    for cnt in countours:    
        Area =+ cv.contourArea(cnt)
        Perimeter =+ cv.arcLength(cnt,True)
        
    fos['Dispersity'] = Perimeter**2/Area
    fos['Saturation'] = Area/Perimeter
    fos['Roundness']  = 4*np.pi*Area/Perimeter**2

    return fos