
import cv2
import numpy as np
import matplotlib.pyplot as plt

def haralick(mask):


    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=4)

    def calc_rdst(x, y, center):
        # calculate the distance metric (rdst) based on (x, y) and center
        return np.sqrt((x - center[0])**2 + (y - center[1])**2)

    # min_area_threshold = 1
    # fout_labs = np.where(stats[:, cv2.CC_STAT_AREA] >= min_area_threshold)[0]
    # labelsf = np.isin(labels, fout_labs).astype(np.uint8)


    # loop through each labeled region
    hair_labs = []

    for label in range(1, num_labels):  # skip the background label (0)
        # create a mask for the current
        label_mask = (labels == label).astype(np.uint8)

        # extract the statistics
        stats_label = stats[label]

        # get the bbox coordinates (x, y, width, height)
        x, y, w, h = stats_label[0], stats_label[1], stats_label[2], stats_label[3]

        # crop the region from the original image
        cropped_region = label_mask[y:y+h, x:x+w]

        center = cropped_region.shape[0] / 2, cropped_region.shape[1] / 2

        y_coords, x_coords = np.where(cropped_region == 1)

        rdst = [calc_rdst(x, y, center = center) for x, y in  zip(x_coords, y_coords)]
        
        hcirc = np.mean(rdst) / np.std(rdst) if np.std(rdst) !=0 else 1 

        # plt.imshow(cropped_region)

        meanc =  hcirc * np.mean(cropped_region)

        print(hcirc, meanc)

        if meanc <= 0.4:
            # assume it is a hair.
            hair_labs.append(label)

        hair_mask = np.isin(labels, hair_labs).astype(np.uint8) * 255

        return hair_mask