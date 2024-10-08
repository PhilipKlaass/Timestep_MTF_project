import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
import os
import os.path
import matplotlib.pyplot as plt
from matplotlib import cm

script_dir= os.path.dirname(__file__)

def get_array(filename, size):
    rel_path = "images_csv/" + filename
    abs_file_path = os.path.join(script_dir,rel_path)
    f = open(abs_file_path, "r")
    array = np.ones((size,size))
    m =0
    for line in f:
        n=0
        line_cleaned = line.strip("   \n")
        line_list = line_cleaned.split("   ")
        for i in line_list:
            array[m][n] = float(i)
            n+=1
        m +=1
    f.close()
    return array

def detect_edge_points(array, threshold):
    m = len(array[0])
    light_value = 0.9 #array[0][m-1]
    output_image = np.zeros((len(array),len(array[0])))
    for j in range(len(array)):
        for i in range(len(array)):
            if (0.5-threshold)*light_value <=  array[j][i] and array[j][i] <= (0.5+threshold)*light_value:
                output_image[j][i] =1




    return output_image




def main():
    array1 =get_array("razor.csv",200)
    array  = detect_edge_points(array1, threshold =0.25)



    angles = np.linspace(-np.pi/2, np.pi/2,720, endpoint=False)
    h,theta, d = hough_line(array, theta= angles)
    fig, ax = plt.subplots(1,4,figsize = (10,5))


    ax[0].imshow(array1, cmap= cm.gray, interpolation= 'none')
    ax[0].set_title("Input Image")
    ax[0].set_xlim(0,len(array1))
    ax[0].set_ylim(0,len(array1))
    #ax[0].set_axis_off()

    ax[3].imshow(array, cmap=cm.gray)
    ax[3].set_title('Input image with edge points identified')
    ax[3].set_axis_off()

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

    lines = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold = 100)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ax[2].axline((x0, y0), slope=np.tan(angle + np.pi/2))
        lines.append((_,angle,dist))
    print(lines)
    pltlines=[]
    for i in lines:
        if 110>i[2] >100:
            x = np.linspace(0,len(array), 250)
            y= (i[2]-x*np.cos(i[1])) / np.sin(i[1])
            ax[0].plot(x,y, color = "r", lw= 1)
            pltlines.append((x,y))
            print(i)
    plt.tight_layout()
    plt.show()



main()