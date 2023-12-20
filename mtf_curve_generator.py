import numpy as np
import matplotlib as plt

#array that will hold our simulated pixel values

x= 9
y = 10
roi = np.zeros((x,y))



#angle that the edge is tilted w.r.t. to vertical
theta = [5]


#number of pixels for a repition
height_of_pattern = (np.tan(theta))**(-1)


for row in range(1,x+1):
    
    for column in range(1, y+1):
        if column< 5 or column < 5-column/height_of_pattern:
            np.put(roi,[row,column], 12)
        if column >5 or column> 5 -column/height_of_pattern:
            np.put(roi,[row,column],1)
        print(roi[row][column])
print(roi)