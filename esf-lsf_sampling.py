import matplotlib.pyplot as plt
import numpy as np
from math import floor
import scipy
import scienceplots


"""
Convert from csv to numpy array
"""
def get_array(filename, size):
    f = open(str(filename), "r")
    array = np.ones((size,size))
    m =0
    for line in f:
        n=0
        line_cleaned = line.strip("   \n")
        line_list = line_cleaned.split("   ")
        for i in line_list:
            array[m][n] = float(i)
            n+=1
        m +=1
    return array

"""
Summary:
        Consider the array in R2, create a number lines perpendicular to the 
        edge and a number(sampling_number) of vertical lines whose spacing is 
        determined by the sampling_frequency. For each intersection of the 
        vertical and perpendicular lines record the distance from the edge and
        the intensity of the pixel in which the intersection ocurred.

Variables:
    - array: image
    - theta and r: definition of our edge r= xcos(theta)+ysin(theta)
    - smapling_frequency: samples per pixel width
    - sample_number: number of samples taken on each side of edge 
"""

def get_esf(array, theta, r, sampling_frequency, sample_number):
    theta = -theta #change sign because the array indexes downwards and the math
                   #is easier when we consider our edge in the 1st quadrant
    x_intercept = r/np.cos(theta) - len(array)*0.5*np.tan(theta)
    esf = []
    for y_edge in range(0,len(array)):
        y_edge = y_edge + 0.5 #change for spacing of perp lines, aka nbr of data points
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


"""
Summary:
    Bins data points.

Variable:
    - esf_dist: sorted from min to max
    - esf_intensity: list of values, identical indexes correspond to single
                    points, aka esf_0 is (x,y) = (esf_dist[0],esf_intensity[0])

    - binsize: measured as percentage of pixel width, determines the range of
               distances over which intensity values will be averaged
"""
def esf_bin_smooth(esf_dist,esf_intensity,binsize):
    minimum_dist = esf_dist[0]
    binned_esf= []
    number_of_bins = np.abs(int((2*minimum_dist)/binsize))
    for i in range(number_of_bins):
        tot_intensity = 0
        k = 0
        for j in range(len(esf_dist)):
            if minimum_dist+(i-0.5)*binsize<esf_dist[j]<=minimum_dist+(i+0.5)*binsize:
                tot_intensity+= esf_intensity[j]
                k+= 1
        binned_esf.append((minimum_dist+i*binsize, tot_intensity/k))

    return binned_esf


"""
Summary: 
    Averages distances and intensities over a square of some number of pixels given by the window size.
    The square is centered on a single data point and the new average replaces that data point.
Variables:
    - esf_dist: sorted from min to max
    - esf_intensity: list of values, identical indexes correspond to single
                    points, aka esf_0 is (x,y) = (esf_dist[0],esf_intensity[0])
    - window_size: odd integer expected, determines the size of the window over which the data points 
                   are averaged 
"""
def average_filter(esf,window_size):
    out_dist= []
    out_intensity = []
    out = []
    for i in esf:
        to_average = []
        avg_dist = 0
        avg_inten = 0
        for j in esf:
            if i[0]-window_size/2<j[0]<i[0]+window_size/2:
                to_average.append(j)
        for k in to_average:
            avg_dist+=k[0]
            avg_inten +=k[1]
        out_dist.append(avg_dist/len(to_average))
        out_intensity.append(avg_inten/len(to_average))
        out.append((avg_dist/len(to_average),avg_inten/len(to_average)))
    return out_dist,out_intensity, out

def median_filter(esf, window_size):
    out_dist= []
    out_intensity = []
    out= []
    for i in range(floor(window_size/2),len(esf)-floor(window_size/2)):
        temp = []
        for j in range(-floor(window_size/2), floor(window_size/2)):
            temp.append(esf[j+i])
        temp_dist,temp_inten = make_scatter(temp)
        median_dist = sorted(temp_dist)[floor(window_size/2)+1]
        median_inten = sorted(temp_inten)[floor(window_size/2)+1]
        out_dist.append(median_dist)
        out_intensity.append(median_inten)
        out.append((median_dist,median_inten))
    return out_dist,out_intensity,out

"""
Summary:
    Uses a 2-point kernel to approximate the derivative of a 1-D function via convlution.
    Kernel is (-1,1)
"""
def get_derivative(x,y):
    x_out=x
    for i in range(1,len(x)-1):
        y[i] = y[i+1]-y[i]
    y_out = y
    y_out = np.delete(y_out,0)
    x_out = np.delete(x_out,0)
    y_out = np.delete(y_out,len(y_out)-1)
    x_out = np.delete(x_out,len(x_out)-1)
    return x_out,y_out



def get_esf2(array, theta, r, sampling_frequency, sample_number):
    theta = -theta
    esf = []
    for row in range(0,len(array),6):
        y1 = row+0.5
        x1 = r/np.cos(theta) -y1*np.tan(theta)
        col = floor(x1)
        if col in range(len(array)):
            intensity = array[row][col]
            dist = col-x1
            #(np.cos(theta)*col+np.sin(theta)*y1-r)
            esf.append((dist,intensity))
        col = col-sample_number
        for i in range(2*sample_number):
            col += i
            if 0<col<len(array):
                intensity = array[row][floor(col)]
                dist = col-x1
                #(np.cos(theta)*(col+0.5)+np.sin(theta)*y1-r)
                esf.append((dist, intensity))
    return esf



'''
Split a list of tuples into x and y arrays.
'''
def make_scatter(array):
    X = []
    Y = []
    for i in array:
        X.append(i[0])
        Y.append(i[1])
    return (X,Y)


def main():
    array = get_array("image0006_corrected_(300,700)-(800,1200).csv", 400)
    #sampling_frequency in samples per pixel pitch
    esf = get_esf(array, -0.0479966,144,.95,4)
    X,Y =  make_scatter(sorted(esf))
    binned_esf = esf_bin_smooth(X,Y, 0.1)

    X_binned , Y_binned = make_scatter(binned_esf)
    
    X_avgfilter,Y_avgfilter, average= average_filter(binned_esf, 0.75)

    X_median, Y_median, median = median_filter(average,13)
    

    plt.style.use(["science", "notebook", "grid"])
    fig , ax = plt.subplots(2,3, figsize = (12,8))
    ax[0][0].plot(X,Y,".", ms= 2)
    ax[0][0].set_title("Oversampled ESF/ERF")

    ax[0][1].plot(X_avgfilter,Y_avgfilter, "-", ms= 2, color = "g")
    ax[0][1].plot(X_median,Y_median, "o", ms= 2, color = "red")
    ax[0][1].plot(X_binned,Y_binned, ".", ms= 2)
    ax[0][1].set_title("Binned into 0.1 pixel width")


    N = 242 #number of samples
    R = max(X_median)-min(X_median) #range of samples in pixel pitch
    delta_x = (R)/(N) #spacing of samples in mm, one pixel = 2.2 microns
    fs = N/R #sampling frequency
    k = np.linspace(0,N/2,int(N/2)+1)


    X_interp = np.linspace(-R/2,R/2,N)
    Y_interp = scipy.interpolate.pchip_interpolate(X_median, Y_median, X_interp)
    Yhat = scipy.signal.savgol_filter(Y_interp,51,2)
    ax[0][2].plot(X_interp,Y_interp,"--",label= "PCHIP Interpolation", lw= 0.75)
    ax[0][2].set_title("Peicewise Cubic Interpolation")

    ax[1][0].plot(X_interp,Yhat,'-', color = 'r',  lw = 1.5)
    ax[1][0].set_title("Savitsky-Golay filter applied")


    
    dy = np.gradient(Yhat,X_interp)
    mdy = max(dy)
    for i in range(len(dy)):
        dy[i] = dy[i]/mdy

    dx2,dy2 = get_derivative(X_interp,Y_interp)
    mdy2 = max(dy2)
    for i in range(len(dy2)):
        dy2[i] = dy2[i]/mdy2
    ax[1][1].plot(dx2,dy2, ".", color ='green')
    ax[1][1].plot(X_interp,dy,"b")
    ax[1][1].set_title("Derivative using(np.gradient)")



    
    xf = scipy.fft.fftfreq(N,delta_x)*(1/R)*(2.2/0.001)
    xf2 = k/(N*delta_x)
    yf2 = scipy.fft.rfft(dy)
    yf2 = np.abs(yf2)
    yf = scipy.fft.fft(dy)
    yf = np.abs(yf)
    maxyf = max(yf)
    maxyf2 = max(yf2)
    for i in range(len(yf)):
        yf[i] = yf[i]/maxyf
    for i in range(len(yf2)):
        yf2[i] = yf2[i]/maxyf2
    ax[1][2].plot(xf2,yf2,'.',color="green")
    ax[1][2].plot(xf,yf,'-', color = "blue", lw = 0.5)
    ax[1][2].set_title("|FFT| of LSF")
    ax[1][2].set_xlim([0,75])
    ax[1][2].set_ylim([0,1])
    ax[1][2].set_xlabel("Cycles per mm")

    print(xf)
    print(delta_x)
    print(len(X_median))

    plt.tight_layout(pad = 1.25)
    plt.show()
main()