import matplotlib.pyplot as plt
import numpy as np
from math import floor
import scipy
import scienceplots
import os
import os.path
from skimage.transform import hough_line, hough_line_peaks
from skimage.feature import canny
from skimage.draw import line as draw_line
from skimage import data
import skimage as ski
from scipy.integrate import dblquad
from scipy.integrate import quad
from matplotlib import cm
import cv2


"""
Utilities-----------------------------------------------------------------------------------------

"""
script_dir= os.path.dirname(__file__)

def open_images(*filename):
    out = ()
    for i in filename:
        rel_path = "images/" + i
        abs_file_path = os.path.join(script_dir,rel_path)
        img = ski.io.imread(abs_file_path)
        img_1 =img[100:300]
        img_roi = np.transpose(img_1)
        img_roi = img_roi[50:250]
        out = out + (img_roi,)
    return out

def save_as_csv(array,filename):
    f = open(filename,'a')
    for element in array:
        for value in element:
            f.write(str(value)+"   ")
        f.write('\n')
    f.close()

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

Simulations --------------------------------------------------------------------------------------

"""
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
        for x in range(column):
            if x < edge_start or x < edge_start-np.tan((np.pi/180)*theta)*y:
                roi[y,x] = dark
            if x>=edge_start or x >= edge_start-np.tan((np.pi/180)*theta)*y:
                roi[y,x]  = bright
    return roi


def make_PSFimage(size,dark,light):
    out = np.zeros((size,size))
    for i in range(len(out)):
        for j in range(i):
            out[i][j] = dark
    center = int(np.floor(size/2))
    if center< size/2:
        out[center][center] = light
        return out
    else:
        out[center][center] = light
        out[center+1][center] = light
        out[center][center+1] = light
        out[center+1][center+1] = light
        return out

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


def make_kernal(xscaling_factor, yscaling_factor, kernel_size):
    f = lambda x,y: np.exp(-xscaling_factor*x**2-yscaling_factor*y**2)
    kernal = np.zeros((kernel_size,kernel_size))
    total = dblquad(f, kernel_size/2,-kernel_size/2,kernel_size/2,-kernel_size/2)[0]
    #in general total is more accurate, however, total and total1 are often equivalent
    total1 = dblquad(f, np.Infinity,-np.Infinity,np.Infinity,-np.Infinity)[0]
    for j in range(kernel_size):
        for i in range(kernel_size):
            xllim= -kernel_size/2+i
            xulim= -kernel_size/2+i+1
            yllim= -kernel_size/2+j
            yulim= -kernel_size/2+j+1
            kernal[j][i] = (dblquad(f,xllim,xulim,yllim,yulim )[0])/total
    return kernal

def convolve(kernel, image):
    (iH, iW) = image.shape[:2]
    (kH, kW) = kernel.shape[:2]

    pad = (kW-1)//2
    image = cv2.copyMakeBorder(image, pad,pad,pad,pad,cv2.BORDER_REPLICATE)
    output = np.zeros((iH,iW))
    for j in np.arange(pad,iH+pad):
        for i in np.arange(pad,iW+pad):
            roi = image[j - pad:j + pad + 1, i - pad:i + pad + 1]
            k = (roi * kernel).sum()
            output[j-pad,i-pad]= k
    return output


def add_poisson(edge,density):
    edge_size_x = int(len(edge))
    edge_size_y = int(len(edge[0]))
    poisson_noise= np.random.poisson(density,(edge_size_x,edge_size_y))
    return edge+poisson_noise



def make_lsf(theta,xscaling_factor, yscaling_factor):
    dist =  np.linspace(-5,5, 100)
    intensity = []
    for i in dist:
        intensity.append( np.exp(i**2*(-yscaling_factor)-xscaling_factor*(np.tan(np.pi/180 *theta))**(2)))
    m_inten = max(intensity)
    for i in range(len(intensity)):
        intensity[i]= intensity[i]/m_inten
    return dist, intensity

def make_erf(theta,xscaling_factor, yscaling_factor):
    dist =  np.linspace(-5,5, 100)
    deltax= 10/100
    f = lambda x: np.exp(x**2*(-yscaling_factor)-xscaling_factor*(np.tan(np.pi/180 *theta))**(2))
    intensity = []
    for i in dist:
        intensity.append(quad(f, i+deltax/2, i-deltax/2)[0])
    m_inten = max(intensity)
    for i in range(len(intensity)):
        intensity[i]= intensity[i]/m_inten
    return dist, intensity


def fft(a,b,theta):
    freqs = np.linspace(0,2,20)
    mtf = []
    for  i in freqs:
        mtf.append(np.abs(function(a,i)))
    m_inten = max(mtf)
    for i in range(len(mtf)):
        mtf[i]= mtf[i]/m_inten
    """
    freqs = np.linspace(0,2,100)
    fourier_transform = []
    for  i in freqs:
        real_integrand =lambda x: np.exp(x**2*(-b-a*(np.tan(np.pi/180 *theta))**(2)))*np.cos(-1*2*np.pi*i*x)
        imaginary_integrand = lambda x: np.exp(x**2*(-b-a*(np.tan(np.pi/180 *theta))**(2)))*np.sin(-1*2*np.pi*i*x)
        fhat = (quad(real_integrand, -np.Infinity,np.Infinity))[0]**2+(quad(imaginary_integrand, -np.Infinity,np.Infinity))[0]**2
        fourier_transform.append(fhat)
    mtf = np.abs(fourier_transform)
    m_inten = max(mtf)
    for i in range(len(mtf)):
        mtf[i]= mtf[i]/m_inten
    """
    return freqs,mtf





def function(a,freq):
    real_integrand =lambda x: np.exp(x**2*(-a)+2j*np.pi*x*freq)
    fhat = (quad(real_integrand, -np.Infinity,np.Infinity))[0]
    return fhat






"""
ROI selection -----------------------------------------------------------------------------------------------------

"""

def flatfield_correction(light, dark, image):
    size = len(image)
    light_dark = np.zeros((size,size))
    image_dark = np.zeros((size,size))
    tot = 0
    for i in range(size):
        for j in range(size):
            tot+= light[i][j]-dark[i][j]
            light_dark[i][j] = light[i][j]-dark[i][j]
            image_dark[i][j] = image[i][j]- dark[i][j]
    m = tot/(size**2)
    image_dark*m
    corrected_image = np.zeros((size,size))
    for i in range(size):
        for j in range(size):
            corrected_image[i][j] = image_dark[i][j]/light_dark[i][j]
    return corrected_image


"""
Edge Detection------------------------------------------------------------------------------------------------------

"""

def detect_edge_points(array, threshold):
    m = len(array[0])
    light_value = array[0][m-1]
    output_image = np.zeros((len(array),len(array[0])))
    for j in range(len(array)):
        for i in range(m-1,0,-1):
            if (0.5-threshold)*light_value <=  array[j][i]<= (0.5+threshold)*light_value:
                output_image[j][i] =1




    return output_image

def hough_transform(array, threshold1, plot):
    angles = np.linspace(-np.pi/2, np.pi/2,720, endpoint=False)
    h,theta, d = hough_line(array, theta= angles)
    angle_step = 0.5 * np.diff(theta).mean()
    d_step = 0.5 * np.diff(d).mean()
    bounds = [np.rad2deg(theta[0] - angle_step),
              np.rad2deg(theta[-1] + angle_step),
              d[-1] + d_step, d[0] - d_step]
    lines = []
    for _, angle, dist in zip(*hough_line_peaks(h, theta, d, threshold = threshold1)):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        ##plt.plot((x0, y0), slope=np.tan(angle + np.pi/2))
        lines.append((angle,dist))
    if plot==True:
        plt.show()
    return lines



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
def FFT(lsf_dist,lsf_inten):
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












def main():
    a= .1
    b= .5
    theta = 5
    object_edge = make_object_plane(theta,1000,1000,0,1)
    image = make_image_plane(object_edge,200)
    psf_kernal = make_kernal(a,b,7)
    dist, intensity = make_lsf(a,b, 5)
    freq,mtf = fft(a,b,theta)
    array= convolve(psf_kernal,image)
    esf = get_esf(array, -0.087266, 100,0.95,3)
    X2,Y2 =  make_scatter(sorted(esf))
    binned_esf = esf_bin_smooth(X2,Y2, .1)

    X_binned , Y_binned = make_scatter(binned_esf)
    
    X_avgfilter,Y_avgfilter, average= average_filter(binned_esf, 0.75)

    X_median, Y_median, median = median_filter(average,13)
    X_interp = np.linspace(min(X_median), max(X_median),1000)
    Y_interp = scipy.interpolate.pchip_interpolate(X_median, Y_median, X_interp)
    Yhat = scipy.signal.savgol_filter(Y_interp,51,2,1)
    xf1,yf1 = FFT(X_interp,Yhat)




    fig, ax = plt.subplots(1,2)
    ax[0].plot(freq,mtf)
    ax[0].plot(xf1,yf1)
    ax[1].plot(dist,intensity)
    ax[1].plot(X_interp,Yhat)
    plt.show()
main()