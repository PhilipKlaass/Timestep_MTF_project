import numpy as np
import matplotlib as plt

#array that will hold our simulated pixel values

row= 15
column = 15
roi = np.zeros((row,column))

#angle that the edge is tilted w.r.t. to vertical
theta = [16]


#number of pixels for a repition
height_of_pattern = (np.tan(theta))**(-1)

# x-postion which the edge starts on the top of the roi
edge_start = 15

for y in range(0,row):
    
    for x in range(0, column):
        if x < edge_start or x < edge_start-y/height_of_pattern:
            roi[y,x] = 1
        if x>edge_start or x > edge_start-y/height_of_pattern:
            roi[y,x]  = 12

print(roi)


