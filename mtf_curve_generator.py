import numpy as np
import random as rand
from scipy import integrate
from scipy.integrate import dblquad
from matplotlib import pyplot as plt
from scipy.fft import rfft, fftfreq

''''
loop to iterate over roi; rows first then columns.
if the column of a particular index is less then the edge start or 
the edge start minus the height of patern then the index value is set to 
1, this represents the pixels of the roi covered by the edge. Simlarly, the 
columns greater than are the pixels not coveered and have values set too 12.



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


def make_line_spread(edge):
    edge_size_x = int(len(edge))
    edge_size_y = int(len(edge[0]))
    #random_scaling_factor 
    K = rand.randint(2,5)
    point_spread_edge = np.zeros((edge_size_x,edge_size_y))
    lsf = lambda z: (1/(2*random_scaling_factor))*np.exp((-((z)**2)**(0.5))/random_scaling_factor)#+ np.abs(z)/(np.abs(z)+10000)) tried changing the lsf but messed up the mtf
    total_intensity = integrate.quad(lsf,-np.inf,np.inf)
    for x in range(0,edge_size_x):
        for y in range(0,edge_size_y):
            x1=0     
            while x1 <edge_size_y:
                #if y<int(0.3*edge_size_y) or y>int(0.7*edge_size_y):
                #    point_spread_edge[x][y] = edge[x][y]
                #else:
                    dist_x = x1-y
                    if -1<y+dist_x<edge_size_y:
                        intensity_for_current_pixel= integrate.quad(lsf, -.1+dist_x*0.2, 0.1+dist_x*0.2)
                        intensity_percentage = intensity_for_current_pixel[0]/total_intensity[0]
                        new_intensity = intensity_percentage*edge[x][y]
                        point_spread_edge[x][y+dist_x] += new_intensity
                #x1+= 1
                    x1+=1
    print(random_scaling_factor)
    make_mtf(lsf)
    return (point_spread_edge,lsf)

def add_poisson(edge,density):
    edge_size_x = int(len(edge))
    edge_size_y = int(len(edge[0]))
    poisson_noise= np.random.poisson(density,(edge_size_x,edge_size_y))
    return edge+poisson_noise



def make_mtf(lsf):
    # Number of sample points
    N = 1000
    # sample spacing
    T = 0.4
    x = np.linspace(0.0, N*T, N, endpoint=False)
    y = lsf(z=x)
    yf = rfft(y)
    xf = fftfreq(N, T)[:N//2]
    plt.plot(xf, 2.0/N * np.abs(yf[0:N//2]))
    plt.grid()
    plt.show()


def save_as_csv(array,filename):
    f = open(filename,'a')
    for element in array:
        for value in element:
            f.write(str(value)+"   ")
        f.write('\n')
    f.close()

def main():
    object_edge = make_object_plane(5,1000,1000,20,200)
    #save_as_csv(object_edge)
    image = make_image_plane(object_edge,100)
    save_as_csv(image, "perfect_lsf.csv")
    #image_with_lsf,lsf = make_line_spread(image)
    #noisy_image = add_poisson(image_with_lsf,0.3)
    plt.imshow(image, interpolation='nearest')
    plt.show()
    #make_mtf(lsf)
main()