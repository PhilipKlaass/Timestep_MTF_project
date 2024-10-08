import numpy as np
import scipy
from scipy.integrate import dblquad
from matplotlib import pyplot as plt
from scipy.integrate import quad
from matplotlib import cm
import cv2

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
    f = lambda x,y: np.exp(-xscaling_factor*x**2-yscaling_factor*y**2)*np.cos(xscaling_factor*yscaling_factor**1/2*(x**2+y**2))**2
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
    return freqs,mtf



def save_as_csv(array,filename):
    f = open(filename,'a')
    for element in array:
        for value in element:
            f.write(str(value)+"   ")
        f.write('\n')
    f.close()

def function(x,y,a,b):
    return np.exp(-a*x**2-b*y**2)

def main():
    a= .5
    b= .5
    theta = 5
    object_edge = make_object_plane(theta,1000,1000,0,1)
    image = make_image_plane(object_edge,200)
    psf_kernal = make_kernal(a,b,7)
    dist, intensity = make_lsf(a,b, 5)
    freq,mtf = fft(a,b,theta)
    image2 = convolve(psf_kernal,image)
    image2=  cv2.flip(image2, 0)
    image=  cv2.flip(image, 0)

    fig, ax = plt.subplots(2,3, figsize = (12,8))


    ax[0][0].imshow(object_edge, cmap= cm.gray, interpolation= 'none')
    ax[0][0].set_title("Object Plane")
    ax[0][1].imshow(image, cmap= cm.gray, interpolation= 'none')
    ax[0][2].set_xlim(80,120)
    ax[0][2].set_ylim(80,120)
    ax[0][1].set_title("Averaged Array")
    ax[0][2].imshow(image2, cmap= cm.gray, interpolation= 'none')
    ax[0][1].set_xlim(80,120)
    ax[0][1].set_ylim(80,120)
    ax[0][2].set_title("Edge with PSF Applied")
    ax[1][0].imshow(psf_kernal,cmap= cm.gray, interpolation= 'none')
    ax[1][0].set_title("PSF Kernel")
    
    a= 2.5
    b= 2.5
    psf_kernal = make_kernal(a,b,7)
    dist, intensity = make_lsf(a,b, theta)
    freq,mtf = fft(a,b,theta)
    dist1,inten = make_erf(theta,a,b)
    image2 = convolve(psf_kernal,image)
    ax[1][2].plot(freq,mtf, ".-",label = "a,b = 2.5")

    a= 1
    b= 1
    psf_kernal = make_kernal(a,b,7)
    dist, intensity = make_lsf(a,b, 5)
    freq,mtf = fft(a,b,theta)
    image2 = convolve(psf_kernal,image)
    ax[1][2].plot(freq,mtf, ".-",label = "a,b = 1")

    a= .5
    b= .5
    psf_kernal = make_kernal(a,b,7)
    dist, intensity = make_lsf(a,b, 5)
    freq,mtf = fft(a,b,theta)
    image2 = convolve(psf_kernal,image)
    ax[1][2].plot(freq,mtf, ".-",label = "a,b = 0.5")


    a= .25
    b= .25
    theta = 5
    psf_kernal = make_kernal(a,b,7)
    dist, intensity = make_lsf(a,b, 5)
    freq,mtf = fft(a,b,theta)
    image2 = convolve(psf_kernal,image)
    ax[1][2].plot(freq,mtf, ".-",label = "a,b = 0.25")

    a= .15
    b= .15
    theta = 5
    psf_kernal = make_kernal(a,b,7)
    dist, intensity = make_lsf(a,b, 5)
    freq,mtf = fft(a,b,theta)
    image2 = convolve(psf_kernal,image)

    #xf2 = []
    #f = open("mtfx.csv","r")
    #for line in f:
    #    line.strip("\n")
    #    xf2.append(float(line))
    #yf2 = []
    #f = open("mtf.csv","r")
    #for line in f:
    #    line.strip("\n")
    #    yf2.append(float(line))


    ax[1][2].plot(freq,mtf,".-", label = "a,b = 0.15")
    #ax[1][2].plot(xf2,yf2,"-.", label ="No Kernel")
    ax[1][2].set_xlabel("Cycles per pixel")
    #ax[1][2].set_ylabel("MTF")
    ax[1][2].set_xlim(0,2)
    ax[1][2].set_title("Simulation MTF Curves")

    ax[1][2].legend()
    

    



    a= 2.5
    b= 2.5
    psf_kernal = make_kernal(a,b,7)
    dist, intensity = make_lsf(a,b, 5)
    freq,mtf = fft(a,b,theta)
    image2 = convolve(psf_kernal,image)
    ax[1][1].plot(dist, intensity, ".-",label = "a,b = 2.5")

    a= 1
    b= 1
    psf_kernal = make_kernal(a,b,7)
    dist, intensity = make_lsf(a,b, 5)
    freq,mtf = fft(a,b,theta)
    image2 = convolve(psf_kernal,image)
    ax[1][1].plot(dist, intensity, ".-",label = "a,b = 1")

    a= .5
    b= .5
    psf_kernal = make_kernal(a,b,7)
    dist, intensity = make_lsf(a,b, 5)
    freq,mtf = fft(a,b,theta)
    image2 = convolve(psf_kernal,image)
    ax[1][1].plot(dist, intensity, ".-",label = "a,b = 0.5")


    a= .25
    b= .25
    theta = 5
    psf_kernal = make_kernal(a,b,7)
    dist, intensity = make_lsf(a,b, 5)
    freq,mtf = fft(a,b,theta)
    image2 = convolve(psf_kernal,image)
    ax[1][1].plot(dist, intensity, ".-",label = "a,b = 0.25")

    a= .15
    b= .15
    theta = 5
    psf_kernal = make_kernal(a,b,7)
    dist, intensity = make_lsf(a,b, 5)
    freq,mtf = fft(a,b,theta)
    image2 = convolve(psf_kernal,image)
    yf = np.abs(scipy.fft.fft(intensity))
    ax[1][2].plot(freq,yf)

    ax[1][1].plot(dist, intensity,".-", label = "a,b = 0.15")
    ax[1][1].set_title("Sample LSF")

    '''
    fig, ax = plt.subplots(2,4, figsize = (16,8))
    ax[0][0].imshow(object_edge, cmap= cm.gray, interpolation= 'none')
    ax[0][1].imshow(image, cmap= cm.gray, interpolation= 'none')
    ax[0][2].imshow(image2, cmap= cm.gray, interpolation= 'none')
    ax[0][3].plot(dist,intensity, ".-")
    ax[1][0].plot(freq,mtf)
    save_as_csv(image2, "sim_0001.csv")
    
    x= np.linspace(-4,4,25)
    y= np.linspace(-4,4,25)
    X,Y = np.meshgrid(x,y)
    Z = function(X,Y,a,b)
    
    ax[1][0] = plt.axes(projection = '3d')
    ax[1][0].plot_surface(X,Y,Z)
    ,r"$I = \int \int  e^{-ax^2-by^2} $"
    '''
    plt.show()

main()