import numpy as np
import matplotlib.pyplot as plt
import random as rand
from PIL import Image
from scipy import integrate
from scipy.integrate import dblquad


#edge = np.ones((100,100))

#edge_size = int(len(edge))
random_scaling_factor = rand.randint(1,10)*0.01
#point_spread_edge = np.zeros(edge_size,edge_size)
psf = lambda y,x: np.exp((-((x)**2+y**2)**(0.5))/random_scaling_factor)
origin = integrate.dblquad(psf,-.2,0.2,-.2,.2)
total_intensity = integrate.dblquad(psf,-np.inf,np.inf,-np.inf,np.inf)
print(origin)
print(total_intensity)
print(origin[0]/total_intensity[0])
print(random_scaling_factor)


'''for x in range(0,edge_size):
    for y in range(0,edge_size):
        point_spread_edge += edge[x][y]
        for x1 in range(0,edge_size):   
            for x2 in range(0,edge_size):'''






def make_point_spread(edge):
    edge_size = int(len(edge))
    random_scaling_factor = rand.randint(1,10)*0.01
    point_spread_edge = np.zeros((edge_size,edge_size))
    psf = lambda y,x: np.exp((-((x)**2+y**2)**(0.5))/random_scaling_factor)
    total_intensity = integrate.dblquad(psf,-np.inf,np.inf,-np.inf,np.inf)
    for x in range(int(edge_size*0.35),int(edge_size*0.65)):
        for y in range(0,edge_size):
            for x1 in range(-6,6):   
                for y1 in range(-6,6):
                    if x+x1<100 and x+x1>0 and y+y1<100 and y+y1>0:
                        dist_x = x-x1
                        dist_y = y-y1
                        intensity_for_current_pixel= integrate.dblquad(psf, -.2+dist_x*0.4, 0.2+dist_x*0.4,-.2+dist_y*0.4, 0.2+dist_y*0.4)
                        intensity_percentage = intensity_for_current_pixel[0]/total_intensity[0]
                        point_spread_edge[x+x1][y+y1]+= intensity_percentage*edge[x][y]
    print(random_scaling_factor)
    return point_spread_edge
