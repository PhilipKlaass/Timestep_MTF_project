import skimage as ski
import matplotlib.pyplot as plt
import numpy as np

def open_image(filename):
    img = ski.io.imread(filename)
    print(img)
    img_1 =img[350:550]
    img_roi = np.transpose(img_1)
    img_roi = img_roi[900:1100]
    return img

def save_as_csv(array,csv_filename):
    f = open(csv_filename,'a')
    for element in array:
        for value in element:
            f.write(str(value)+"   ")
        if True:
            f.write('\n')
    f.close()

def main():
    
    roi = open_image("razor0001.bmp")
    plt.imshow(roi,interpolation='nearest')
    plt.show()
    #save_as_csv(roi, "image0001.csv")
main()

