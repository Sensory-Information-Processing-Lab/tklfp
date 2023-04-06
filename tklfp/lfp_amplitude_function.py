from cmath import sqrt
from ctypes import sizeof
from random import randrange
import numpy as np
import math
import random

"""
* imput params: matrix of arrays representing individual neurons
* each array houses x, y, and z coordinates of respective structures
* ex: soma = [[1,1,1],
              [2,2,2],
              [3,3,3]]
* must be same number of soma and dendrite coordinates, i.e. each neuron has a soma and dendrite point in space
* each neuron maps to each electrode
"""

def compute_amp (soma, dendrite, electrode):
    length = compute_length(soma, dendrite)
    midpoint = compute_midpoint(soma, dendrite)
    distance = compute_distance(midpoint, electrode)
    angle = compute_angle(midpoint, electrode)
    LFP_amp_arr = np.empty([len(electrode), len(soma)])
    for arr in range(len(electrode)):
        for num in range(len(soma)):
            LFP_amp = (length[arr] * math.cos(angle[num])) / (4 * math.pi * 0.3 * distance[num] ** 2)
            LFP_amp_arr[arr][num] = round(LFP_amp, 15)
    return LFP_amp_arr

def compute_length (soma, dendrite):
    length_arr = []
    for arr in range(len(soma)):
        x1 = soma[arr][0]
        x2 = dendrite[arr][0]
        y1 = soma[arr][1]
        y2 = dendrite[arr][1]
        z1 = soma[arr][2]
        z2 = dendrite[arr][2]
        
        length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
        length_arr.append(length)
    return length_arr

def compute_midpoint (soma, dendrite):
    midpoint_arr = []
    for arr in range(len(soma)):
        x1 = soma[arr][0]
        x2 = dendrite[arr][0]
        y1 = soma[arr][1]
        y2 = dendrite[arr][1]
        z1 = soma[arr][2]
        z2 = dendrite[arr][2]
        midpoint = [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]
        midpoint_arr.append(midpoint)
    return midpoint_arr

def compute_distance (midpoint, electrode):
    distance_arr = []
    for arr in range(len(electrode)):
        for num in range(len(midpoint)):
            x1 = midpoint[num][0]
            x2 = electrode[arr][0]
            y1 = midpoint[num][1]
            y2 = electrode[arr][1]
            z1 = midpoint[num][2]
            z2 = electrode[arr][2]
            distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
            if (y1 > y2):
                distance = -distance
            distance_arr.append(distance)
    return distance_arr

def compute_angle (midpoint, electrode):
    angle_arr = []
    for arr in range(len(electrode)):
        for num in range(len(midpoint)):
            unit_vector1 = midpoint[num] / np.linalg.norm(midpoint[num])
            unit_vector2 = electrode[arr] / np.linalg.norm(electrode[arr])
            dot_product = np.dot(unit_vector1, unit_vector2)
            if dot_product > 1:
                dot_product = 1
            angle = np.arccos(dot_product)
            angle_arr.append(angle)
    return angle_arr
