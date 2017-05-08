'''
======================
3D surface (color map)
======================

Demonstrates plotting a 3D surface colored with the coolwarm color map.
The surface is made opaque by using antialiased=False.

Also demonstrates using the LinearLocator and custom formatting for the
z axis tick labels.
'''

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
import ipdb
import matplotlib.animation as animation
from alg import *




CURVES = []



# Make data.
X = np.arange(-1, 1, 0.1)
Y = np.arange(-1, 1, 0.1)
X, Y = np.meshgrid(X, Y)
Z = curve(X,Y)


x0 = 1
y0 = 0
eps = 0.01
z0 = curve(x0,y0)

trails = 500
iters = 100
res = np.zeros((trails,4))
path = np.zeros((trails,4))

for i in xrange(trails):
    print i
    pos = getSGDMomentumPath(x0,y0,iters,0.05,0.9)
    pos[pos < -10] = -10
    pos[np.isnan(pos)] = -10
    pos_diff = np.diff(pos)
    path[i,0] = np.sum(np.linalg.norm(pos,axis=1))
    if np.isnan(pos).any() or (curve(pos[-1,0],pos[-1,1]) < -100):
        res[i,0] = 1
    pos = getDSR3MomentumPath(x0,y0,iters,0.03,0.9,0.99,0.30)
    pos[pos < -10] = -10
    pos[np.isnan(pos)] = -10
    pos_diff = np.diff(pos)
    path[i,1] = np.sum(np.linalg.norm(pos,axis=1))
    if np.isnan(pos).any() or (curve(pos[-1,0],pos[-1,1]) < -100):
        res[i,1] = 1
    pos = getDSR2MomentumPath(x0,y0,iters,0.05,0.9,0.999,0.5)
    pos[pos < -10] = -10
    pos[np.isnan(pos)] = -10
    pos_diff = np.diff(pos)
    path[i,2] = np.sum(np.linalg.norm(pos,axis=1))
    if np.isnan(pos).any() or (curve(pos[-1,0],pos[-1,1]) < -100):
        res[i,2] = 1
    pos = getADAMPath(x0,y0,iters,0.05,0.9,0.999,10e-8)
    pos[pos < -10] = -10
    pos[np.isnan(pos)] = -10
    pos_diff = np.diff(pos)
    path[i,3] = np.sum(np.linalg.norm(pos,axis=1))
    if np.isnan(pos).any() or (curve(pos[-1,0],pos[-1,1]) < -100):
        res[i,3] = 1

print "result {}".format(np.mean(res,axis=0))
print "path {}".format(np.mean(path,axis=0))
# Plot the surface.

