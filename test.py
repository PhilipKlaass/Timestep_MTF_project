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
import matplotlib


a = np.linspace(-10,10, 1000)
b = np.ones(1000)

aa ,bb  = MTF.lanczos_resampling(a, b, 3)

plt.plot(aa,bb/max(bb))
plt.plot(a,b/max(b))
plt.show()

N = 100
A = simulate_edge.make_object_plane(16, 1000, 1000, 15,200)

B = simulate_edge.make_image_plane(A, N)

normal = simulate_edge.make_kernal(0.5, 0.5, 3)

dark = np.zeros((N,N))
light = np.ones((N,N))

C = simulate_edge.convolve(normal,B)

for i in range(100):
    dark += np.random.normal(0,0.05,(N,N))/100
    light += np.random.normal(0,0.05,(N,N))/100
    C += np.random.normal(0,0.05,(N,N))/100
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

mm = 1/np.tan(theta)

plt.imshow(CC)
plt.colorbar()
plt.plot([N/2,-N/m+N/2],[0,N],'r')
plt.plot([N/2,-N/mm+N/2],[0,N],'b')
plt.show()

erf_x,erf_y = MTF.get_esf(CC, theta, r, "total")

plt.scatter(erf_x, erf_y)
plt.show()

erf_x_resampled,erf_y_resampled = MTF.lanczos_resampling(erf_x,erf_y,3)

plt.scatter(erf_x_resampled,erf_y_resampled)
plt.show()

lsf_x, lsf_y = MTF.get_derivative(erf_x_resampled,erf_y_resampled)
M = max(lsf_y)
for i in range(len(lsf_y)):
    lsf_y[i] = lsf_y[i]/M

mtf_x,mtf_y = MTF.FFT(lsf_x,lsf_y)

print("ISO slope is %f" % m)
print("Hugh line slope is %f" % mm)



plt.plot(mtf_x,mtf_y)
plt.xlim((0,1))
plt.show()

