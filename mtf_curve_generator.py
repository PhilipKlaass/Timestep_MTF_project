import numpy as np
import matplotlib.pyplot as plt
import random as rand
from PIL import Image
import scipy as sci


''''
loop to iterate over roi; rows first then columns.
if the column of a particular index is less then the edge start or 
the edge start minus the height of patern then the index value is set to 
1, this represents the pixels of the roi covered by the edge. Simlarly, the 
columns greater than are the pixels not coveered and have values set too 12.


!!change once ESF is made!!
'''
def object_plane(theta, roi_height, roi_width):
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
                roi[y,x] = 0
            if x>edge_start or x > edge_start-np.tan((np.pi/180)*theta)*y:
                roi[y,x]  = 255

    return roi

'''
Estimating MTF as a 6th degree polynomial
'''
#def simulated_PSF(wavelength, f_number, aperature_diameter):
#    normalized_radius = 
'''
    c_1 = round(rand.uniform(0.5,1)*(10**(-1)),4)
    c_2 = round(rand.uniform(-1,0.5)*(10**(-2)),4)
    c_3 = round(rand.uniform(-1,0.5)*(10**(-3)),4)
    c_4 = round(rand.uniform(-1,0.5)*(10**(-5)),4)
    c_5 = round(rand.uniform(-1,0.5)*(10**(-7)),4)
    c_6 = round(rand.uniform(-1,0.5)*(10**(-9)),4)
    sum_coefficients = c_1+c_2+c_3+c_4+c_5+c_6
    if sum_coefficients > 1:
        c_0 = sum_coefficients-1
    else:
        c_0 =1- sum_coefficients
    fx  = c_0 + c_1*x+c_2*x**2+c_3*x**3+c_4*x**4+c_5*x**5+c_6*x**6
    return fx

    
'''
'''
plot mtf curve
generate array of x-values
find xvalue where mtf(x) is zero 
yaxis == mtf(x)
and plot
'''
def mtf_plt(step):
    zero = 10
    x_axis = []
    y_axis = []
    x=0
    while x < zero:
        x_axis.append(x)
        y_axis.append(simulated_MTF(x))
        x+=step
    plt.scatter(x_axis,y_axis)
    plt.show()


def main():
    edge_object = object_plane(10,1000,1000)
    img = Image.fromarray(edge_object)
    img.show()
    


    return

main()