from cmath import sqrt
from ctypes import sizeof
from random import randrange
import numpy as np
import math
import random
"""
soma: array containing the x, y, and z position coordinates of the soma
dendrite: array containing the x, y, and z position coordinates of the dendrites
electrode: array containing the x, y, and z position coordinates of the electrode
"""

def computeAmp (soma, dendrite, electrode):
    lfp_amp_arr = np.zeros(len(soma))
    for arr in range(len(soma)):
        length = computeLength(soma[arr], dendrite[arr])
        midpoint = computeMidPoint(soma[arr], dendrite[arr])
        distance = computeDistance(midpoint, electrode[arr])
        angle = computeAngle(midpoint, electrode[arr])
        LFPAmp = (length * math.cos(angle)) / (4 * math.pi * 0.3 * distance ** 2)
        lfp_amp_arr[arr] = LFPAmp
    return lfp_amp_arr

def computeLength (soma, dendrite):
    x1 = soma[0]
    x2 = dendrite[0]
    y1 = soma[1]
    y2 = dendrite[1]
    z1 = soma[2]
    z2 = dendrite[2]
    length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    return length

def computeMidPoint (soma, dendrite):
    x1 = soma[0]
    x2 = dendrite[0]
    y1 = soma[1]
    y2 = dendrite[1]
    z1 = soma[2]
    z2 = dendrite[2]
    midpoint = [(x1 + x2) / 2, (y1 + y2) / 2, (z1 + z2) / 2]
    print(midpoint)
    return midpoint

def computeDistance (midpoint, electrode):
    x1 = midpoint[0]
    x2 = electrode[0]
    y1 = midpoint[1]
    y2 = electrode[1]
    z1 = midpoint[2]
    z2 = electrode[2]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)
    if (y1 > y2):
        return -distance
    else:
        return distance

def computeAngle (midpoint, electrode):
    unitVector1 = midpoint / np.linalg.norm(midpoint)
    unitVector2 = electrode / np.linalg.norm(electrode)
    dotProduct = np.dot(unitVector1, unitVector2)
    angle = np.arccos(dotProduct)
    return angle

