import numpy as np
import matplotlib as plt
import random as rand

#size of the Region of Interest or ROI/roi
row= 15
column = 15

#array that will hold our simulated pixel values
roi = np.zeros((row,column))

#angle that the edge is tilted w.r.t. to vertical
theta = [16]


#number of pixels for a repition
height_of_pattern = (np.tan(theta))**(-1)

# x-postion which the edge starts on the top of the roi
edge_start = 15


''''
loop to iterate over roi; rows first then columns.
if the column of a particular index is less then the edge start or 
the edge start minus the height of patern then the index value is set to 
1, this represents the pixels of the roi covered by the edge. Simlarly, the 
columns greater than are the pixels not coveered and have values set too 12.


!!change once ESF is made!!
'''
for y in range(0,row):
    
    for x in range(0, column):
        if x < edge_start or x < edge_start-y/height_of_pattern:
            roi[y,x] = 1
        if x>edge_start or x > edge_start-y/height_of_pattern:
            roi[y,x]  = 12


'''
Estimating MTF as a 6th degree polynomial
'''
def MTF(x):
    c_1 = round(rand.uniform(-1,1)*(10**(-1)),4)
    c_2 = round(rand.uniform(-1,1)*(10**(-2)),4)
    c_3 = round(rand.uniform(-1,1)*(10**(-3)),4)
    c_4 = round(rand.uniform(-1,1)*(10**(-5)),4)
    c_5 = round(rand.uniform(-1,1)*(10**(-7)),4)
    c_6 = round(rand.uniform(-1,1)*(10**(-9)),4)
    sum_coefficients = c_1+((c_2)**2)**(1/2)+c_3+((c_4)**2)**(1/2)+c_5+((c_6)**2)**(1/2)
    if sum_coefficients > 1:
        c_0 = sum_coefficients-1
    else:
        c_0 =1- sum_coefficients


'''
plot mtf curve
generate array of x-values
find xvalue where mtf(x) is zero 
yaxis == mtf(x)
and plot
'''
def mtf_plt():
