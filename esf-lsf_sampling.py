import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
from math import floor
from scipy.optimize import curve_fit
import scipy
import scienceplots

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
    x_intercept = r/np.cos(theta) - len(array)*0.5*np.tan(theta)
    esf = []
    for y_edge in range(0,5*len(array)):
        y_edge = y_edge/5 + 0.5 #change for spacing of perp lines, aka nbr of data points
        x_edge = -y_edge*np.tan(theta)+r/np.cos(theta)
        for i in range(sample_number):
            x_sample1 = x_intercept + i/sampling_frequency 
            x_sample2 = x_intercept - i/sampling_frequency
            y_sample1 = np.tan(theta)*(x_sample1-x_edge)+y_edge
            y_sample2 = np.tan(theta)*(x_sample2-x_edge)+y_edge
            if 0<= y_sample1<= len(array) and 0<= x_sample1<= len(array[0]): 
                intensity1 = array[floor(y_sample1)][floor(x_sample1)]
                dist1 = ((x_sample1-x_edge)/(np.abs(x_sample1-x_edge)))*((y_edge-y_sample1)**2+(x_edge-x_sample1)**2)**0.5
                esf.append((dist1,intensity1))
            if 0<= y_sample2<= len(array) and 0<= x_sample2<= len(array[0]):
                dist2 = ((x_sample2-x_edge)/(np.abs(x_sample2-x_edge)))*((y_edge-y_sample2)**2+(x_edge-x_sample2)**2)**0.5
                intensity2 =  array[floor(y_sample2)][floor(x_sample2)]
                esf.append((dist2,intensity2))
    return esf

def average_esf(array):
    values = [[],[],[]]
    out = []
    for i in range(len(array)):
        if array[i][0] not in values[0]:
            values[0].append(array[i][0])
            values[1].append(array[i][1])
            values[2].append(1)
        else:
            values[1][values[0].index(array[i][0])]+= array[i][1]
            values[2][values[0].index(array[i][0])]+=1
    for i in range(len(values[0])):
        out.append((values[0][i],values[1][i]/values[2][i]))
    return out

def make_scatter(array):
    X = []
    Y = []
    for i in array:
        X.append(i[0])
        Y.append(i[1])
    return (X,Y)


def esf_function(x,a,b,c,D,x0):

    #return (x/np.abs(x))*(a*(np.abs(x)-x0)**(1/3)+b*(np.abs(x)-x0)**(1/5)+c*(np.abs(x)-x0)**(1/7))+D
    return (x/np.abs(x))*(a+b*(np.abs(x)-x0)**(1/5)+c*(np.abs(x)-x0)**(1/7))+D




def main():
    array = get_array("image0001.csv", 200)
    #sampling_frequency in samples per pixel pitch
    esf = get_esf(array, -0.04799655,89, 0.2,3)
    X,Y = make_scatter(sorted(esf))





    
    averaged_esf = average_esf(sorted(esf))
    X_avg,Y_avg = make_scatter(averaged_esf)
    plt.style.use(["science", "notebook", "grid"])
    fig , ax = plt.subplots(1,2, figsize = (10,3.5))
    ax[0].plot(X,Y,".", ms= 1.5)
    #ax[0].set_xlabel("Distance")
    ax[0].set_ylabel("Pixel Intensity Values")
    ax[0].set_title("Oversampled ESF")
    ax[1].plot(X_avg,Y_avg,".", ms= 1.5)
    #ax[1].set_xlabel("Distance")
    ax[1].set_title("Oversampled and Averaged ESF")

    X_interp = np.linspace(-15,15,200)
    Y_interp = scipy.interpolate.pchip_interpolate(X_avg, Y_avg, X_interp)
    ax[1].plot(X_interp,Y_interp,'-', color = 'r', label= "PCHIP Interpolation", lw = 1.5)
    
    popt, pcov = curve_fit(esf_function, X,Y, p0=[10,10,10,120,0])
    a_opt,b_opt,c_opt,D_opt, x0_opt = popt
    x_model = np.linspace(min(X), max(X), len(X))
    y_model = esf_function(x_model,a_opt,b_opt,c_opt,D_opt,x0_opt)
    ax[1].plot(x_model,y_model,"--", color= "green", label = "Cubic curve fit", lw= 1.5)
    ax[1].legend(loc ="upper left", fontsize = 10)
    plt.figtext(0.5,0.02, "Distance to Edge [Pixel Pitch]", ha= "center", fontsize = 16 )
    plt.show()
    print(len(X))
main()
#print(esf_function(-15,10,10,10,120,0))