# -*- coding: utf-8 -*-
"""
Created on Sun Mar 23 14:38:06 2025

@author: klaas
"""

import MTF
import numpy as np
import simulate_edge
import ISO
import matplotlib.pyplot as plt
import csv
from PIL import Image
import os


cwd = os.getcwd()

test = Image.open(cwd+'\images\\new_setup\\dark0001.tiff')
test_arr = np.array(test)

m,n = test_arr.shape

dark = np.zeros((m,n))
light= np.zeros((m,n))
image= np.zeros((m,n))

for i in range(1,51):
    
    if i<10:
        dark_cur = Image.open(cwd+'\images\\new_setup\\dark000%d.tiff'%i)
        light_cur = Image.open(cwd+'\images\\new_setup\\light000%d.tiff'%i)
        image_cur = Image.open(cwd+'\images\\new_setup\\image000%d.tiff'%i)
    else:
        dark_cur = Image.open(cwd+'\images\\new_setup\\dark00%d.tiff'%i)
        light_cur = Image.open(cwd+'\images\\new_setup\\light00%d.tiff'%i)
        image_cur = Image.open(cwd+'\images\\new_setup\\image00%d.tiff'%i)
    
    dark_cur_arr = np.array(dark_cur)
    light_cur_arr = np.array(light_cur)
    image_cur_arr = np.array(image_cur)
    
    dark = dark+dark_cur_arr/50
    light =light+light_cur_arr/50
    image =image +image_cur_arr/50




'''
N = 100
A = simulate_edge.make_object_plane(16, 1000, 1000, 15,200)

B = simulate_edge.make_image_plane(A, N)

normal = simulate_edge.make_kernal(0.5, 0.5, 3)


C = simulate_edge.convolve(normal,B)

for i in range(50):
    dark += np.random.normal(0,0.01,(N,N))/50
    light += np.random.normal(0,0.01,(N,N))/50
    C += np.random.normal(0,0.01,(N,N))/50
dark = dark + np.ones((N,N))*15
light = light + np.ones((N,N))*235
'''
CC = MTF.flatfield_correction(light, dark, image)


CC = CC[600:800,950:1150]
plt.imshow(CC)
plt.colorbar()
plt.plot()
plt.show()

N = 200


from PIL import Image

I8 = (((CC - CC.min()) / (CC.max() - CC.min())) * 255.9).astype(np.uint8)

img = Image.fromarray(I8)
img.save("CC.tiff")


m = ISO.calc_slope(CC)

threshold = N*0.5
edge_points = MTF.detect_edge_points(CC, 0.2)
lines = MTF.hough_transform(edge_points,threshold,False)

print(lines)

r,theta = lines[0][1],lines[0][0]

'''
r = 0
k = True
while k ==  True:
    if image[0,r]<0.8:
        r+=1
    else:
        k = False
'''

#theta = np.arctan(1/m)

mm = 1/np.tan(theta)

plt.imshow(CC)
plt.colorbar()
plt.plot([r,N/m+r],[0,N-5],'r')
plt.plot([r,-N/mm+r],[0,N-5],'b')
plt.show()

erf_x,erf_y = MTF.get_esf(CC*255, theta, r, "total")
#erf_y = ISO.get_esf(CC, m)

#erf_x = np.linspace(-50, 50,400)

fig = plt.figure()
plt.scatter(erf_x, erf_y)
plt.show()

#erf_x_resampled,erf_y_resampled = MTF.lanczos_resampling(erf_x,erf_y,3)
#erf_x_resampled,erf_y_resampled = MTF.bin_esf(erf_x,erf_y,0.25)
erf_x_resampled,erf_y_resampled = MTF.bin_esf(erf_x,erf_y,0.25)

#esf = ISO.get_esf(CC, m)
#fig = plt.figure()
#plt.plot(esf)
#plt.show()


fig = plt.figure()
plt.scatter(erf_x_resampled,erf_y_resampled)
plt.show()

lsf_x, lsf_y = MTF.get_lsf(erf_x_resampled,erf_y_resampled,1)

fig = plt.figure()
plt.scatter(lsf_x, lsf_y)
plt.show()   


mtf_x,mtf_y = MTF.FFT(lsf_x,lsf_y)

print("ISO slope is %f" % m)
print("Hugh line slope is %f" % mm)

mtfdh = []
mtffreq = []

with open("MTFDH.csv",newline = '') as csvfile:
    file = csv.reader(csvfile)
    for row in file:
        mtfdh.append(float(row[0]))
with open("MTFFREQ.csv",newline = '') as csvfile:
    file = csv.reader(csvfile)
    for row in file:
        for i in range(len(row)):
            mtffreq.append(float(row[i]))

fig = plt.figure()
plt.plot(mtf_x,mtf_y,label = "Our code")
plt.plot(mtffreq,mtfdh,'r',label = "MTF_dh")
plt.xlim((0,1))
plt.title("MTF")
plt.xlabel("Spatial Frequency (per pixel)")
plt.legend()
plt.savefig('MTF_compare', dpi=400)
plt.show()