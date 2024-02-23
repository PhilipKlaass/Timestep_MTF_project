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
    slope = -(np.tan(theta))
    esf = []
    
    for i in range(0,len(array),2):#Step of this range function determnies samples per row
        y0 = i+0.5
        x0 = (r-np.sin(theta)*y0)/(np.cos(theta))
        #print(str(y0)+str(  x0))
        for j in range(sample_number):
            
            x1= x0+j/sampling_frequency
            x2 = x0-j/sampling_frequency
            y1 = slope*(x1-x0)+y0
            y2 = slope*(x2-x0)+y0
            if 0<y1<len(array)  and 0<x1<len(array[0]) :
                intensity1 = array[floor(y1)][floor(x1)]
                dist1 = round(((x1-x0)**2+(y1-y0)**2)**(0.5),6)
                esf.append((dist1,intensity1))
            if 0<y2<len(array)and 0<x2<len(array[0]):
                intensity2 = array[floor(y2)][floor(x2)]
                dist2 = round(((x2-x0)**2+(y2-y0)**2)**(0.5),6)
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
    array = get_array("razor0001.csv", 100)
    #sampling_frequency in samples per pixel pitch
    esf = get_esf(array, 0.017,8, 1,4)
    X,Y = make_scatter(esf)
    plt.plot(X,Y,"o")
    plt.xlabel("Distance")
    plt.ylabel("Intensity")
    plt.show()
    print(esf)

main()