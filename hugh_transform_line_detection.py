import numpy as np

from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data

import matplotlib.pyplot as plt
from matplotlib import cm

def get_array(filename, size):
    f = open(str(filename), "r")
    array = np.ones((size,size), np.int8)
    m =0
    for line in f:
        n=0
        line_cleaned = line.strip("\n")
        line_list = line_cleaned.split("  ")
        for i in line_list:
            array[m][n] = int(i)
            n+=1
        m +=1
    return array

def detect_edge_points(array, pixel_pitch):
    size_i = len(array)-1
    size_j = len(array[0])-1
    dark_value = array[0][0]
    light_value = array[0][len(array[0])-1]
    if dark_value>light_value:
        (dark_value,light_value) = (light_value,dark_value)
    average_value = (dark_value+light_value)/2
    coords_for_edge = []
    for i in range(0,size_i):
        for j in range(0,size_j):
            if 0.9*average_value<array[i][j] and array[i][j]< 1.1*average_value:
                (x,y) = ((j+0.5)*pixel_pitch,(size_i-(i)+0.5)*pixel_pitch)
                coords_for_edge.append((x,y))
    return coords_for_edge


def find_line(point_coords):
    possible_lines = []
    h= 0
    theta_low = -70
    theta_high = -50
    while h <500:
        while len(possible_lines)<1000-h*.05:
            for element in point_coords:
                r_max = (element[0]**2 + element[1]**2)**0.5
                possible_lines.append((random.randint(1,100)*0.01*r_max,random.randint(theta_low,theta_high)))
        for i in possible_lines:
            weight = 0 
            for j in point_coords:
                x= j[0]
                y= j[1]
                theta = (np.pi/180)*np.arctan(y/x)
                r_prime = x*np.cos(theta)+y*np.sin(theta)
                if r_prime>0.90*i[0] and r_prime<1.1*i[1]:
                    weight += 1
            if weight<5:
                possible_lines.remove(i)
        h+=1
    return possible_lines

def display_accumulator_space(list):
    x =[]
    y =[]
    for i in list:
        x.append(i[0])
        y.append(i[1])
    plt.scatter(x,y)
    plt.show()




def main():
    array =get_array("csv.txt",6)
    #edge_coords  = detect_edge_points(array, 1)
    #lines = find_line(edge_coords)
    #display_accumulator_space(lines)


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