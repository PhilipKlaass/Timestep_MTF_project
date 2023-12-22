import numpy as np
import matplotlib.pyplot as plt
import random as rand
from PIL import Image

def simulated_MTF(x):
    c_1 = round(rand.randint(-9,9)*(10**(-2)),4)
    c_2 = round(rand.uniform(-9,9)*(10**(-3)),4)
    c_3 = round(rand.uniform(-9,9)*(10**(-4)),4)
    c_4 = round(rand.uniform(-9,9)*(10**(-6)),4)
    c_5 = round(rand.uniform(-9,9)*(10**(-8)),4)
    c_6 = round(rand.uniform(-9,9)*(10**(-10)),4)
    sum_coefficients = c_1+c_2+c_3+c_4+c_5+c_6
    if sum_coefficients > 1:
        c_0 = sum_coefficients-1
    else:
        c_0 =1- sum_coefficients
    fx  = np.cos(x/100)+ c_1*x+c_2*x**2+c_3*x**3+c_4*x**4+c_5*x**5+c_6*x**6
    return fx
print(simulated_MTF(2))
def mtf_plt(step):
    zero = 10
    x_axis = []
    y_axis = []
    x=0
    while x < zero:
        x_axis.append(x)
        y_axis.append(simulated_MTF(x))
        x+=step
    plt.scatter(x_axis,y_axis)
    plt.show()
mtf_plt(0.01)