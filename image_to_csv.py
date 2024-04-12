import skimage as ski
import matplotlib.pyplot as plt
import numpy as np
import os
import os.path
import shutil
import cv2
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
script_dir= os.path.dirname(__file__)

def open_images(*filename,full):
    out = ()
    for i in filename:
        rel_path = "images/" + i
        abs_file_path = os.path.join(script_dir,rel_path)
        img = ski.io.imread(abs_file_path)
        img_1 =img[350 : 550]
        img_roi = np.transpose(img_1)
        img_roi = img_roi[600:800]
        if full == True:
            out = out + (img,)
        else:
            out = out + (img_roi,)
    return out

def save_as_csv(array,csv_filename):
    f = open(csv_filename,'a')
    for element in array:
        for value in element:
            f.write(str(value)+"   ")
        if True:
            f.write('\n')
    f.close()

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


def main():
    
    roi, light, dark = open_images("image0006.tiff","image0006_light.bmp", "dark.bmp", full = True)
    corrected_roi1 = flatfield_correction(light,dark,roi)
    corrected_roi2  = cv2.rotate(corrected_roi1, cv2.ROTATE_90_CLOCKWISE)
    corrected_roi = cv2.flip(corrected_roi1, -1)
    #corrected_roi = cv2.flip(corrected_roi2, 0)
    fig, ax = plt.subplots(figsize = (9,9))
    plt.imshow(corrected_roi,interpolation='nearest', cmap = "gist_grey")

    
    x1, x2, y1, y2 = 800,1000,900,1100
    axins = inset_axes(ax, height = 4, width = 4, loc=1) # zoom = 6
    axins.imshow(corrected_roi, cmap = "gist_grey", interpolation='nearest')
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec="0.5")
    plt.xticks(visible=False)
    plt.yticks(visible=False)

    #plt.colorbar()
    plt.title("Image")
    #x = np.linspace(0,200, 250)
    #y= ((105.0)-x*np.cos(1.5271630954950384)) / np.sin(1.5271630954950384)
    #plt.plot(x,y, color = "r", lw= 2)
    #plt.xlim(0,200)
    plt.draw()
    plt.show()
    #save_as_csv(corrected_roi, "razor.csv")
main()

