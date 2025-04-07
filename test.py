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
<<<<<<< HEAD
=======
import matplotlib
import csv
>>>>>>> origin/master


N = 100
A = simulate_edge.make_object_plane(16, 1000, 1000, 15,200)

B = simulate_edge.make_image_plane(A, N)

normal = simulate_edge.make_kernal(1,1, 3)

dark = np.zeros((N,N))
light = np.ones((N,N))

C = simulate_edge.convolve(normal,B)

for i in range(100):
    dark += np.random.normal(0,0.01,(N,N))/100
    light += np.random.normal(0,0.01,(N,N))/100
    C += np.random.normal(0,0.01,(N,N))/100
dark = dark + np.ones((N,N))*15
light = light + np.ones((N,N))*235

CC = MTF.flatfield_correction(light, dark, C)

plt.imshow(dark)
plt.colorbar()
plt.plot()
plt.show()

plt.imshow(light)
plt.colorbar()
plt.plot()
plt.show()



from PIL import Image

I8 = (((CC - CC.min()) / (CC.max() - CC.min())) * 255.9).astype(np.uint8)

img = Image.fromarray(I8)
img.save("CC.tiff")

m = ISO.calc_slope(C)

threshold = N*0.5
edge_points = MTF.detect_edge_points(CC, 0.2)
lines = MTF.hough_transform(edge_points,threshold,False)

r,theta = lines[0][1],lines[0][0]

theta = np.arctan(1/m)

mm = 1/np.tan(theta)

plt.imshow(CC)
plt.colorbar()
plt.plot([N/2,-N/m+N/2],[0,N],'r')
plt.plot([N/2,-N/mm+N/2],[0,N],'b')
plt.show()

erf_x,erf_y = MTF.get_esf(CC*255, theta, r, "total")
#erf_y = ISO.get_esf(CC, m)

#erf_x = np.linspace(-50, 50,400)

fig = plt.figure()
plt.scatter(erf_x, erf_y)
plt.show()

<<<<<<< HEAD
erf_x_resampled,erf_y_resampled = MTF.lanczos_resampling(erf_x,erf_y,4)
=======
#erf_x_resampled,erf_y_resampled = MTF.lanczos_resampling(erf_x,erf_y,3)
#erf_x_resampled,erf_y_resampled = MTF.bin_esf(erf_x,erf_y,0.25)
erf_x_resampled,erf_y_resampled = MTF.bin_esf(erf_x,erf_y,0.25)
>>>>>>> origin/master


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
plt.savefig('MTF_compare', dpi = 400)
plt.show()