import numpy as np
import random as rand
from PIL import Image
from scipy import integrate
from scipy.integrate import dblquad
from matplotlib import pyplot as plt

''''
loop to iterate over roi; rows first then columns.
if the column of a particular index is less then the edge start or 
the edge start minus the height of patern then the index value is set to 
1, this represents the pixels of the roi covered by the edge. Simlarly, the 
columns greater than are the pixels not coveered and have values set too 12.


!!change once ESF is made!!
'''
def make_object_plane(theta, roi_height, roi_width,dark,bright):
    #size of the Region of Interest or ROI/roi
    row= roi_height
    column = roi_width

    #array that will hold our simulated pixel values
    roi = np.zeros((row,column))

    #angle that the edge is tilted w.r.t. to vertical is theta
    # x-postion which the edge starts on the top of the roi
    edge_start = int(column/2)
    for y in range(0,row):
        for x in range(0, column):
            if x < edge_start or x < edge_start-np.tan((np.pi/180)*theta)*y:
                roi[y,x] = dark
            if x>=edge_start or x >= edge_start-np.tan((np.pi/180)*theta)*y:
                roi[y,x]  = bright

    return roi




def make_image_plane(object_plane, size):
    image_plane = np.zeros((size,size))
    for x in range(0,size):
        for y in range(0,size):
            value = 0
            for n in range(0,int(len(object_plane)/size)):
                for m in range(0,int(len(object_plane)/size)):
                    value =value + object_plane[n+x*int(len(object_plane)/size)][m+y*int(len(object_plane)/size)]
            image_plane[x][y] = value/((len(object_plane)/size)**2)
    return image_plane
            
def make_point_spread(edge,dark):
    edge_size = int(len(edge))
    random_scaling_factor = 0.25 #rand.randint(1,10)*0.01
    point_spread_edge = np.zeros((edge_size,edge_size))
    psf = lambda x: np.exp((-((x)**2)**(0.5))/random_scaling_factor)
    total_intensity = integrate.quad(psf,-np.inf,np.inf)
    for x in range(0,edge_size):
        for y in range(0,edge_size):
            x1=0     
            while x1 <edge_size:   
                dist_x = x1-y
                if -1<y+dist_x<edge_size:
                    intensity_for_current_pixel= integrate.quad(psf, -.1+dist_x*0.2, 0.1+dist_x*0.2)
                    intensity_percentage = intensity_for_current_pixel[0]/total_intensity[0]
                    new_intensity = intensity_percentage*edge[x][y]
                    point_spread_edge[x][y+dist_x] += new_intensity
                x1+= 1
    print(random_scaling_factor)
    return point_spread_edge




def main():
    object_edge = make_object_plane(10,100,100,50,200)
    image_edge = make_point_spread(object_edge,100)
    plt.imshow(image_edge, interpolation='nearest')
    plt.show()

main()