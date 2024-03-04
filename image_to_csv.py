import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

def open_images(*filename):
    out = ()
    for i in filename:
        img = ski.io.imread(i)
        img_1 =img[400:600]
        img_roi = np.transpose(img_1)
        img_roi = img_roi[900:1100]
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
    
    roi, light, dark = open_images("image0006.tiff","image0006_light.bmp", "dark.bmp")
    corrected_roi = flatfield_correction(light,dark,roi)
    plt.imshow(corrected_roi,interpolation='nearest')
    plt.colorbar()
    plt.title("Region of Interest")
    plt.show()
    #save_as_csv(corrected_roi, "image0006_corrected_(400,600)-(900,1100).csv")
main()

