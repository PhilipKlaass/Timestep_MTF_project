import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from math import floor

def get_array(filename, size):
    f = open(str(filename), "r")
    array = np.ones((size,size))
    m =0
    for line in f:
        n=0
        line_cleaned = line.strip("   \n")
        line_list = line_cleaned.split("   ")
        for i in line_list:
            array[m][n] = int(i)
            n+=1
        m +=1
    return array

def get_esf(array, theta, r, sampling_frequency, sample_number):
    theta = -theta #change sign because the array indexes downwards and the math
                   #is easier when we consider our edge in the 1st quadrant
    x_intercept = r/np.cos(theta)
    esf = []
    for y_edge in range(len(array)):
        y_edge += 0.5
        x_edge = -y_edge*np.tan(theta)+r/np.cos(theta)
        for i in range(sample_number):
            x_sample1 = x_intercept + i/sampling_frequency 
            x_sample2 = x_intercept - i/sampling_frequency
            y_sample1 = np.tan(theta)*(x_sample1-x_edge)+y_edge
            y_sample2 = np.tan(theta)*(x_sample2-x_edge)+y_edge
            if 0<= y_sample1<= len(array) and 0<= x_sample1<= len(array[0]): 
                intensity1 = array[floor(y_sample1)][floor(x_sample1)]
                dist1 = x_sample1-x_edge#((y_edge-y_sample1)**2+(x_edge-x_sample1)**2)**0.5
                esf.append((dist1,intensity1))
            if 0<= y_sample2<= len(array) and 0<= x_sample2<= len(array[0]):
                dist2 = x_sample2-x_edge#((y_edge-y_sample2)**2+(x_edge-x_sample2)**2)**0.5
                intensity2 =  array[floor(y_sample2)][floor(x_sample2)]
                esf.append((dist2,intensity2))
    return esf

def make_scatter(array):
    X = []
    Y = []
    for i in array:
        X.append(i[0])
        Y.append(i[1])
    return (X,Y)



def main():
    array = get_array("image0001.csv", 200)
    #sampling_frequency in samples per pixel pitch
    esf = get_esf(array, -0.04799655,89, 0.4,6)
    X,Y = make_scatter(esf)
    plt.plot(X,Y,".")
    plt.xlabel("Distance")
    plt.ylabel("Intensity")
    plt.show()

main()