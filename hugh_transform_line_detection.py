import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm

def get_array(filename, size):
    f = open(str(filename), "r")
    array = np.ones((size,size))
    m =0
    for line in f:
        n=0
        line_cleaned = line.strip("   \n")
        line_list = line_cleaned.split("   ")
        for i in line_list:
            array[m][n] = int(float(i))
            n+=1
        m +=1
    return array

def detect_edge_points(array, threshold):
    size_i = len(array)-1
    size_j = len(array[0])-1
    dark_value = array[0][0]
    light_value = array[0][len(array[0])-1]
    if dark_value>light_value:
        (dark_value,light_value) = (light_value,dark_value)
    average_value = (dark_value+light_value)/2
    output_image = np.zeros((size_i,size_i), np.int8)
    for i in range(0,size_i):
        for j in range(0,size_j):
            if (1-threshold)*average_value<array[i][j] and array[i][j]< (1+threshold)*average_value:
                output_image[i][j]= 1




    return output_image




def main():
    array1 =get_array("csv.txt",100)
    array  = detect_edge_points(array1, threshold =0.5)



    angles = np.linspace(-np.pi/2, np.pi/2,720, endpoint=False)
    h,theta, d = hough_line(array, theta= angles)
    fig, axes = plt.subplots(1,3,figsize = (15,6))
    ax = axes.ravel()
    ax[0].imshow(array, cmap=cm.gray)
    ax[0].set_title('Input image')
    ax[0].set_axis_off()

    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]
    ax[1].imshow(np.log(1 + h), extent=bounds, cmap=cm.gray, aspect=1 / 1.5)
    ax[1].set_title('Hough transform')
    ax[1].set_xlabel('Angles (degrees)')
    ax[1].set_ylabel('Distance (pixels)')
    ax[1].axis('image')


    ax[2].imshow(array, cmap=cm.gray)
    ax[2].set_ylim((array.shape[0], 0))
    ax[2].set_axis_off()
    ax[2].set_title('Detected lines')

    for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))
    plt.tight_layout()
    plt.show()



main()