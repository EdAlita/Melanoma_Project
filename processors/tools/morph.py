import cv2 
import numpy as np

def inbounds(shape, indices):
    '''
    Test if the given coordinates inside the given image. 

    The first input parameter is the shape of image (height, weight) and the 
    second parameter is the coordinates to be tested (y, x)

    The function returns True if the coordinates inside the image and vice versa.

    '''
    assert len(shape) == len(indices)
    for i, ind in enumerate(indices):
        if ind < 0 or ind >= shape[i]:
            return False
    return True


def setlable(img, labimg, x, y, label, size):
    '''
    This function is used for label image. 

    The first two input images are the image to be labeled and an output image with 
    labeled region. "x", "y" are the coordinate to be tested, "label" is the ID
    of a region and size is used to limit maximum size of a region. 

    '''
    if img[y][x] and not labimg[y][x]:
        labimg[y][x] = label
        size += 1
        if size > 500:
                return False
        if inbounds(img.shape, (y, x+1)):
            setlable(img, labimg, x+1, y,label, size)
        if inbounds(img.shape, (y+1, x)):
            setlable(img, labimg, x, y+1,label, size)
        if inbounds(img.shape, (y, x-1)):
            setlable(img, labimg, x-1, y,label, size)
        if inbounds(img.shape, (y-1, x)):
            setlable(img, labimg, x, y-1,label, size)
        if inbounds(img.shape, (y+1, x+1)):
            setlable(img, labimg, x+1, y+1,label, size)
        if inbounds(img.shape, (y+1, x-1)):
            setlable(img, labimg, x-1, y+1,label, size)
        if inbounds(img.shape, (y-1, x+1)):
            setlable(img, labimg, x+1, y-1,label, size)
        if inbounds(img.shape, (y-1, x-1)):
            setlable(img, labimg, x-1, y-1,label, size)


def cls(src, r = 3, eps = 0.01, max_iter = 10, filter_out = 130):

    # initialize change var
    change = 1
    # copy target matrix
    out = src.copy()
    # initialize a prev
    prev_dst = out

    while change >= eps:
        
        prev_dst = out

        se = cv2.getStructuringElement(ksize=[r, r], shape=cv2.MORPH_RECT)
        out = cv2.morphologyEx(out, 
                        op= cv2.MORPH_CLOSE, 
                        kernel= se)
        
        
        change = np.sum(np.abs(prev_dst - out)) / np.sum(out)
        # print("We're at", change* 100, "% .")

    # print("Reached eps. percentage at ", change * 100, "% .")

    h, w = out.shape[:2]

    dsc = out.copy()

    if filter_out != -1:
        lab = 1
        label = np.zeros(out.shape)
        for y in range(h):
            for x in range(w):
                if not label[y][x] and out[y][x]:
                    size = 0
                    setlable(out, label, x, y, lab, size)
                    lab += 1
        num = np.zeros(lab)
        for y in range(h):
            for x in range(w):
                num[int(label[y][x]-1)] += 1
        for y in range(h):
            for x in range(w):
                if num[int(label[y][x]-1)] <= filter_out:
                    out[y][x] = 0

    return dsc, out




# apply morphological area opening to filter small objects.
min_area_threshold = 130
# filtered_labels = np.where(stats[:, cv2.CC_STAT_AREA] >= min_area_threshold)[0]
# hair_labels = np.isin(labels, filtered_labels).astype(np.uint8) * 255


