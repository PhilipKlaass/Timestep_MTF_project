import numpy as np 
import matplotlib.pyplot as plt
import scipy as scipy 
import random

def get_array(filename, size):
    f = open(str(filename), "r")
    array = np.ones((size,size), np.int8)
    m =0
    for line in f:
        n=0
        line_cleaned = line.strip("\n")
        line_list = line_cleaned.split("  ")
        for i in line_list:
            array[m][n] = int(i)
            n+=1
        m +=1
    return array

def detect_edge(array, pixel_pitch):
    size_i = len(array)-1
    size_j = len(array[0])-1
    dark_value = array[0][0]
    light_value = array[0][len(array[0])-1]
    if dark_value>light_value:
        (dark_value,light_value) = (light_value,dark_value)
    average_value = (dark_value+light_value)/2
    coords_for_edge = []
    for i in range(0,size_i):
        for j in range(0,size_j):
            if 0.9*average_value<array[i][j] and array[i][j]< 1.1*average_value:
                (x,y) = ((j+0.5)*pixel_pitch,(size_i-(i)+0.5)*pixel_pitch)
                coords_for_edge.append((x,y))
    return coords_for_edge


def find_line(point_coords):
    possible_lines = []
    h= 0
    while h <25:
        if len(possible_lines)<1000:
            for element in point_coords:
                r_max = (element[0]**2 + element[1]**2)**0.5
                for i in range(int((100-len(possible_lines))/len(point_coords))):
                    possible_lines.append((random.randint(1,100)*0.01*r_max,random.randint(-180,180)))
        for i in possible_lines:
            weight = 0 
            for j in point_coords:
                x= j[0]
                y= j[1]
                theta = np.arctan(y/x)
                r_prime = x*np.cos(theta)+y*np.sin(theta)
                if r_prime>0.85*i[0] and r_prime<1.15*i[1]:
                    weight += 1
            if weight<5:
                possible_lines.remove(i)



        h+=1
    return possible_lines

def display_accumulator_space(list):
    x =[]
    y =[]
    for i in list:
        x.append(i[0])
        y.append(i[1])
    plt.axes




def main():
    array =get_array("csv.txt",6)
    edge_coords  = detect_edge(array, 1)
    print(find_line(edge_coords))



main()