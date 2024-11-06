import matplotlib.pyplot as plt
import numpy as np
from math import floor
import scipy
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
import cv2 as cv
from pylab import rcParams
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


"""
Utilities----------------------------------------------------------------------

"""
script_dir= os.path.dirname(__file__)


def open_images(filename, roi_select = 1, row0=1,row1=1,col0=1,col1=1):
    """
    

    Parameters
    ----------
    filename : string
        Name of image in folder, \images\filename.
    roi_select: booleen
        Toggles the user roi selection.
    row0, row1:
    
    Returns
    -------
    TYPE
        Numpy array of the intended ROI rotated 90 degrees ccw.

    """
    rel_path = r"images/" + filename
    abs_file_path = os.path.join(script_dir,rel_path)
    image_array = np.rot90(np.array(cv.imread(abs_file_path,flags = 2 )))
    out = False
    if roi_select== False:
        return image_array[row0:row1,col0:col1],[row0,row1],[col0,col1]
    while out == False:
        plt.imshow(image_array)
        plt.show()
        roi_coords = input("Enter the coordinates for the wanted ROI.\nE.g. 0:100,0:100\n")
        coords_str = roi_coords.split(',')
        roi_rows = [int(i) for i in coords_str[0].split(':')]
        roi_cols = [int(i) for i in coords_str[1].split(':')]
        
        if roi_rows[1]-roi_rows[0] == roi_cols[1]-roi_cols[0]:
            plt.imshow(
                        image_array[roi_rows[0]:roi_rows[1] ,
                                    roi_cols[0]:roi_cols[1]]  )
            plt.show()
            
            confirmation = input('(Y/N) Is this the intended ROI?\n')
            
            if confirmation == 'Y':
                out = True
        else:
            print('Please enter a square ROI.')
        
    return image_array[roi_rows[0]:roi_rows[1] , roi_cols[0]:roi_cols[1]],roi_rows,roi_cols




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
    image = cv.copyMakeBorder(image, pad,pad,pad,pad,cv.BORDER_REPLICATE)
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
    dist =  np.linspace(-11.94,9.95, 1000)
    intensity = []
    for i in dist:
        intensity.append( np.exp(i**2*-xscaling_factor))
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
    
    N = len(image)
    avg_light_dark = np.sum(light-dark)/ N**2
    corrected_image = avg_light_dark*(image - dark)/(light-dark)
    
    max_light = np.array(corrected_image).max()
    
    normalized_cor_image = corrected_image/max_light
    
    return normalized_cor_image

"""
Edge Detection------------------------------------------------------------------------------------------------------

"""

def detect_edge_points(array, threshold):
    '''
    

    Parameters
    ----------
    array : numpy array
        DESCRIPTION.
    threshold : float
        A float between 0 and 0.5, determines the which points are chosen as
        edges.

    Returns
    -------
    edge_points : numpy array
        Binary array with edgepoints set to a value of 1.

    '''
    light_value = np.array(array).max()
    edge_points = np.zeros((len(array),len(array[0])))
    for j in range(len(array)):
        for i in range(len(array)-1,0,-1):
            if np.sum(edge_points[j])<5: #added to incase bright points after 
                                         #edge are present  
                if (0.5-threshold)*light_value <=  array[j][i]<= (0.5+threshold)*light_value:
                    edge_points[j][i] =1
                

    return edge_points

def hough_transform(array, threshold1, plot):
    '''
    

    Parameters
    ----------
    array : numpy array
        Array with edge points set to weight of 1 ande everything else set to 0.
    threshold1 : integer
        Determines which lines are suitable candidates. The number of edge 
        points intersected by the line is the line strength.
    plot : Booleen
        Toggles plotting

    Returns
    -------
    lines : List of tuples
        List of tuples for possible lines, formatted as [(radius,theta),...]

    '''
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
        lines.append((angle,dist,dist/np.cos(angle)))
    if plot==True:
        plt.imshow(array)
        plt.axline((x0, y0), slope=np.tan(angle + np.pi/2))
        plt.show()
    print("Lines:\n")
    for i in range(len(lines)):
        print(lines[i])
    return lines




def get_esf(array, theta, r, sampling_frequency, sample_number):
    '''

    Parameters
    ----------
    array : numpy array
        image array
        
    theta : float
        angle used to describe edge line
        
    r : float
        radius used to describe edge line
        
    sampling_frequency : float
        samples per pixel width
        
    sample_number : integer
        number of samples taken on each side of edge 

    Returns
    -------
    esf_x : list
        DESCRIPTION.
    esf_y : list
        DESCRIPTION.

    '''
    theta = theta #change sign because the array indexes downwards and the math
                   #is easier when we consider our edge in the 1st quadrant
    x_intercept = r/np.cos(theta) - len(array)*0.5*np.tan(theta)
    esf_x = []
    esf_y = []
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
                esf_x.append(dist1-1)
                esf_y.append(intensity1)
                                
            if 0<= y_sample2<= len(array) and 0<= x_sample2<= len(array[0]):
                dist2 = ((x_sample2-x_edge)/(np.abs(x_sample2-x_edge)))*((y_edge-y_sample2)**2+(x_edge-x_sample2)**2)**0.5
                intensity2 =  array[floor(y_sample2)][floor(x_sample2)]
                esf_x.append(dist2-1)
                esf_y.append(intensity2)
    return esf_x,esf_y


def esf_bin_smooth(esf_dist,esf_intensity,binsize):
    '''
    

    Parameters
    ----------
    esf_dist : array
        x-coords of esf
    esf_intensity : array
        list of values, identical indexes correspond to single points, aka 
        esf_0 is (x,y) = (esf_dist[0],esf_intensity[0])
    binsize : integer
        measured as percentage of pixel width, determines the range of 
        distances over which intensity values will be averaged

    Returns
    -------
    binned_esfx : array
        Binned postions
    binned_esfy : array
        Binned intensity

    '''
    minimum_dist = min(esf_dist)
    distance_range = max(esf_dist)-min(esf_dist)
    binned_esfx= []
    binned_esfy = []
    number_of_bins = np.abs(int((distance_range)/binsize))
    for i in range(number_of_bins):
        tot_intensity = 0
        k = 0
        for j in range(len(esf_dist)):
            if minimum_dist+(i-0.5)*binsize<esf_dist[j]<=minimum_dist+(i+0.5)*binsize:
                tot_intensity+= esf_intensity[j]
                k+= 1
        if k!= 0:
            binned_esfx.append(minimum_dist+i*binsize)
            binned_esfy.append(tot_intensity/k)

    return binned_esfx,binned_esfy


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
def average_filter(esfx,esfy,window_size):
    '''
    

    Parameters
    ----------
    esf : array
        Esf intensity values
    window_size : TYPE
        Size of window in which to calculate the median

    Returns
    -------
    med_esfx : Array
        Median distance values
    med_esfy : Array
        Median intensity values


    '''
    med_esfx= []
    med_esfy = []
    step= window_size
    i = 0
    while i<len(esfy):
        avg_dist = 0
        avg_inten = 0
        if i+step>len(esfy):
            for l in range(i,len(esfy)):
               avg_dist+=esfx[l]
               avg_inten+=esfy[l]
        else:
            for k in range(step):
                avg_dist+=esfx[i+k]
                avg_inten+=esfy[i+k]
        med_esfx.append(avg_dist/step)
        med_esfy.append(avg_inten/step)
        
        i+= step
        
    '''    
    for i in range(0,len(esfy), step):
        to_average = []
        avg_dist = 0
        avg_inten = 0
        for j in range(len(esfy)):
            if esfy[i]-window_size/2<esfy[j]<esfy[i]+window_size/2:
                avg_dist+=esfx[j]
                avg_inten +=esfy[j]
        med_esfx.append(avg_dist/len(to_average))
        med_esfy.append(avg_inten/len(to_average))
        out.append((avg_dist/len(to_average),avg_inten/len(to_average)))
    '''
    return med_esfx,med_esfy

def median_filter(esfx,esfy, window_size):
    out_dist= []
    out_intensity = []
    out= []
    for i in range(floor(window_size/2),len(esfy)-floor(window_size/2)):
        temp = []
        j = floor(window_size/2)
        temp_dist,temp_inten = esfx[i-j:i+j], esfy[i-j:i+j]
        median_dist = sorted(temp_dist)[floor(window_size/2)+1]
        median_inten = sorted(temp_inten)[floor(window_size/2)+1]
        out_dist.append(median_dist)
        out_intensity.append(median_inten)
        out.append((median_dist,median_inten))
    return out_dist,out_intensity

"""
Summary:
    Uses a 2-point kernel to approximate the derivative of a 1-D function via convolution.
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
    minten = max(y_out)
    for i in range(len(y_out)):
        y_out[i]=y_out[i]/minten
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

    m = X[0]
    for i in range(len(X)):
        X[i]= X[i]/m

    R = (max(lsf_dist)-min(lsf_dist))#range in units of pixels
    sr = N/(R) #sampling rate
    freq = n*sr/N #spatial frequency in cycles/mm


    return freq ,X

def intro():
    
    #filename = input("Enter the filename in the images folder you want to analyze.\n")
    ROI,rows,cols = open_images('image0008.bmp',roi_select=True,row0 =400,row1 =600,col0 =400,col1 =600)
    #filename = input("Enter the filename of the light frame.\n")
    light,x,y = open_images('image0008_light.bmp', False,rows[0],rows[1],cols[0],cols[1])
    #filename = input("Enter the filename of the dark frame.\n")
    dark,x,y = open_images('image0008_dark.bmp', False,rows[0],rows[1],cols[0],cols[1])
    corrected_ROI = flatfield_correction(light, dark, ROI)
    plt.imshow(corrected_ROI)
    plt.colorbar()
    plt.show()
    
    threshold = (rows[1]-rows[0])*0.85
    edge_points = detect_edge_points(corrected_ROI, 0.2)
    lines = hough_transform(edge_points,threshold,True)
    
    #line_input = input("Enter the correct edge line.E.g.  0.09162978572970237,34.0\n")
    #line_list = line_input.split(',')
    #r,theta = float(line_list[0]),float(line_list[1])
    r,theta = lines[1][1],lines[1][0]
    
    erf_x,erf_y = get_esf(corrected_ROI, theta, r,0.9, 20)
    
    plt.scatter(erf_x,erf_y, marker = '.')
    plt.show()
    
    
    binned_esfx,binned_esfy = esf_bin_smooth(erf_x,erf_y, .1)
    
    avg_erfx, avg_erfy = average_filter(binned_esfx,binned_esfy,5)
    
    med_erfx, med_erfy = median_filter(avg_erfx,avg_erfy, 5)
    
    plt.scatter(med_erfx,med_erfy,marker='.')
    plt.title('Median applied')
    plt.show()
    
    X_interp = np.linspace(min(med_erfx), max(med_erfx),1000)
    Y_interp = scipy.interpolate.pchip_interpolate(med_erfx, med_erfy, X_interp)
    Yhat = scipy.signal.savgol_filter(Y_interp,51,2,0)
    
    plt.plot(X_interp,Yhat)
    plt.title('Sav-Gol applied')
    plt.show()
    
    lsf_x, lsf_y = get_derivative(X_interp, Yhat)
    
    
    plt.plot(lsf_x,lsf_y)
    plt.plot()
    plt.show()
    
    mtf_x,mtf_y = FFT(lsf_x, lsf_y)
    
    plt.scatter(mtf_x,mtf_y, marker = '.')
    plt.xlim((0,1))
    
    freq_res  = (mtf_x[-1]-mtf_x[0])/len(mtf_x)
    print(freq_res)
#intro()
    

def reorder():
    
    #filename = input("Enter the filename in the images folder you want to analyze.\n")
    ROI,rows,cols = open_images('image0008.bmp',roi_select=True,row0 =400,row1 =600,col0 =400,col1 =600)
    #filename = input("Enter the filename of the light frame.\n")
    light,x,y = open_images('image0008_light.bmp', False,rows[0],rows[1],cols[0],cols[1])
    #filename = input("Enter the filename of the dark frame.\n")
    dark,x,y = open_images('image0008_dark.bmp', False,rows[0],rows[1],cols[0],cols[1])
    corrected_ROI = flatfield_correction(light, dark, ROI)
    plt.imshow(corrected_ROI)
    plt.colorbar()
    plt.show()
    
    threshold = (rows[1]-rows[0])*0.85
    edge_points = detect_edge_points(corrected_ROI, 0.2)
    lines = hough_transform(edge_points,threshold,True)
    
    #line_input = input("Enter the correct edge line.E.g.(0.09162978572970237, 34.0)\n")
    #line_list = line_input.split(',')
    #r,theta = float(line_list[0]),float(line_list[1])
    r,theta = lines[0][1],lines[0][0]
    
    erf_x,erf_y = get_esf(corrected_ROI, theta, r,0.9, 20)
    binx,biny = esf_bin_smooth(erf_x,erf_y, .1)
    plt.scatter(binx,biny, marker = '.')
    plt.show()
    
    Yhat = scipy.signal.savgol_filter(biny,51,2,0)
    
    plt.scatter(binx,Yhat,marker='.')
    plt.show()
    
    lsf_x,lsf_y = get_derivative(binx, Yhat)
    
    plt.scatter(lsf_x,lsf_y,marker='.')
    plt.show()
    
    binned_esfx,binned_esfy = esf_bin_smooth(lsf_x,lsf_y, .1)
    
    avg_erfx, avg_erfy = average_filter(binned_esfx,binned_esfy,5)
    
    med_erfx, med_erfy = median_filter(avg_erfx,avg_erfy, 5)
    
    plt.scatter(med_erfx,med_erfy,marker='.')
    plt.title('Median applied')
    plt.show()
    
    

    
    mtf_x,mtf_y = FFT(med_erfx, med_erfy)
    
    plt.scatter(mtf_x,mtf_y, marker = '.')
    plt.xlim((0,1))
    
    freq_res  = (mtf_x[-1]-mtf_x[0])/len(mtf_x)
    print(freq_res)
reorder()