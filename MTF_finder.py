import numpy as np
import random as rand
from scipy import integrate
from scipy.integrate import dblquad
from matplotlib import pyplot as plt
from scipy.fft import fft, fftfreq

def read_csv(file_name):
    f = open(file_name,"r")
    array_out = []
    for x in f:
        x.strip('\n')
        line = x.split("   ")
        row = []
        for y in line:
            row.append(y)
        array_out.append(row)
    return array_out


def find_edge(array):





def make_ESF(array, edge):



def smooth_esf():

def main():
    edge = read_csv('edge_csv.txt')
    print(edge)
main()