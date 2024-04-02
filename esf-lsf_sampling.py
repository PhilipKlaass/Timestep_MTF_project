import matplotlib.pyplot as plt
import numpy as np
from math import floor
import scipy
import scienceplots
import os
import os.path



script_dir= os.path.dirname(__file__)
"""
Convert from csv to numpy array
"""
def get_array(filename, size):
    rel_path = "images_csv/" + filename
    abs_file_path = os.path.join(script_dir,rel_path)
    f = open(abs_file_path, "r")
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
    f.close()
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


"""
Summary:
    Outputs the modulus of the fft of an array, and the associated real 
    frequencies in cycles/mm
Variables:
    - lsf_inten: a 1-D array of pixel intensities
    - lsf_dist: a 1-D array of distances in pixel pitch
    - N: number of samples expected to be equal to length of lsf_inten
    - pixel_size: in microns, needed to convert to real frequencies
"""
def FFT(lsf_dist,lsf_inten,pixel_size):
    N= len(lsf_inten)
    n = np.arange(N)
    k = n.reshape((N,1))
    e = np.exp(-2j*2*np.pi*k*(n/N)) #NxN array with columns of exp(-2j*pi*kn/N)
    X= np.dot(e,lsf_inten)  #matrix multiplication, FFT
    X1 = scipy.fft.fft(lsf_inten)
    X =np.abs(X1) #modulus of the FFT

    m = max(X)
    for i in range(len(X)):
        X[i]= X[i]/m

    R = (max(lsf_dist)-min(lsf_dist))#range in units of pixels
    sr = N/(R) #sampling rate
    freq = n*sr/N #spatial frequency in cycles/mm


    return freq ,X





    """
    R = np.abs(max(lsf_dist))+np.abs(min(lsf_dist)) #range of samples in pixel pitch
    delta_x = R/N #spacing of samples
    fs = N/R #sampling frequency
    k = np.linspace(0,N,N,endpoint=False) #indexes for the fourier frequencies
    X = (k/N)*(1/R)*(pixel_size/0.001) #converts to fourier freq and normalize to cycles/mm
    Y = scipy.fft.fft(lsf_inten,N)
    Y = np.abs(Y)
    mY = max(Y)
    for i in range(len(Y)):
        Y[i]= Y[i]/mY
    return X,Y
"""


def main():
    array = get_array("image0008_corrected_(100,300)-(50,250).csv", 200)
    #sampling_frequency in samples per pixel pitch
    esf = get_esf(array, 0.10035643198967392,119,0.95,3)
    X2,Y2 =  make_scatter(sorted(esf))
    binned_esf = esf_bin_smooth(X2,Y2, .1)

    X_binned , Y_binned = make_scatter(binned_esf)
    
    X_avgfilter,Y_avgfilter, average= average_filter(binned_esf, 0.75)

    X_median, Y_median, median = median_filter(average,13)
    
    smoothing = [("binned",X_binned,Y_binned), ("averaaged",X_avgfilter,Y_avgfilter), ("median",X_median,Y_median)]

    plt.style.use(["science", "notebook"])
    fig , ax = plt.subplots(2,3, figsize = (12,8))
    ax[0][0].plot(X2,Y2,".", ms= 2)
    ax[0][0].set_title("Oversampled ESF/ERF")

    ax[0][1].plot(X_avgfilter,Y_avgfilter, ".-", lw= 1, color = "g", label ="average")
    ax[0][1].plot(X_median,Y_median, ".-", lw= .5, color = "red", label = "median")
    ax[0][1].plot(X_binned,Y_binned, ".-", lw= .5, label = 'binned')
    ax[0][1].legend(fontsize = 12)
    ax[0][1].set_title("Binned into 0.1 pixel width")



    X_interp = np.linspace(min(X_median), max(X_median),1000)
    Y_interp = scipy.interpolate.pchip_interpolate(X_median, Y_median, X_interp)
    Yhat = scipy.signal.savgol_filter(Y_interp,51,2,0)
    ax[0][2].plot(X_interp,Y_interp,".",label= "PCHIP Interpolation", lw= 0.75)
    ax[0][2].set_title("Peicewise Cubic Interpolation")

    ax[1][0].plot(X_interp,Yhat,'.', color = 'r',  lw = 1.5)
    ax[1][0].set_title("Savitsky-Golay filter applied")
    smoothing.append(("interpolated+savgol", X_interp,Yhat))

    

    for label1,x,y in smoothing:
        dx1,dy1 = get_derivative(x,y)

        mdy1 = max(dy1)
        for i in range(len(dy1)):
            dy1[i] = dy1[i]/mdy1
        ax[1][1].plot(dx1,dy1, ".-", label = label1)
        #ax[1][1].plot(X_interp,dy,"b")



        ax[1][1].set_title("Derivative, LSF")

        xf1,yf1 = FFT(X_interp,dy1,2.2)
        ax[1][2].plot(xf1,yf1,'.-', lw= 0.5, label = label1)
    ax[1][1].legend(fontsize = 12)
    ax[1][2].legend(fontsize = 12)

  
    



    Yhat_smoothed= scipy.signal.savgol_filter(Yhat, 500,2, 0)
    mdy2 = Yhat_smoothed[int(len(Yhat_smoothed)/2)]
    for i in range(len(Yhat_smoothed)):
        Yhat_smoothed[i] = Yhat_smoothed[i]/mdy2
    for i in range(len(Yhat_smoothed)):
        if Yhat_smoothed[i]>1:
            Yhat_smoothed[i]= 0
    #ax[1][1].plot(X_interp, Yhat_smoothed)

    




    #ax[1][2].plot(xf,yf,'-', color = "blue", lw = 0.5)
    ax[1][2].set_title("|FFT| of LSF")
    ax[1][2].set_xlim([0,2])
    ax[1][2].set_ylim([-0.01,1])
    #ax[1][2].set_ylim([0,1])
    ax[1][2].set_xlabel("Cycles per pixel")


    plt.tight_layout(pad = 1.25)
    plt.show()
main()
















''' 
ignore this bit, I manually calculated the fft

    xf = scipy.fft.fftfreq(N,delta_x)*(2*np.pi/R)*(2.2/0.001)
    xf2 = k/(N*delta_x)*(2*np.pi/R)*(2.2/0.001)
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
'''