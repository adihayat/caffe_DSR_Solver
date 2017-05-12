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
import alg




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

noise_list = [0.05,0.02,0.015,0.01,0.008,0.005,0.003,0.0025,0.002,0.001]
names = ['SGD','DSR4','NST','ADAM']
res_noise = np.zeros((len(noise_list),4))
path_noise = np.zeros((len(noise_list),4))



trails = 500
iters = 150

for n_idx,noise in enumerate(noise_list):
    print n_idx
    res = np.zeros((trails,4))
    path = np.zeros((trails,4))
    alg.GlobalNoisePower = noise

    for i in xrange(trails):
        #print i
        pos = getSGDMomentumPath(x0,y0,iters,0.05,0.9)
        pos[pos < -10] = -10
        pos[np.isnan(pos)] = -10
        pos_diff = np.diff(pos)
        if np.isnan(pos).any() or (curve(pos[-1,0],pos[-1,1]) < 0):
            res[i,0] = 1
            path[i,0] = np.sum(np.linalg.norm(pos,axis=1))
        pos = getDSR5MomentumPath(x0,y0,iters,0.05,0.9,0.999,10e-8,0.7)
        pos[pos < -10] = -10
        pos[np.isnan(pos)] = -10
        pos_diff = np.diff(pos)
        if np.isnan(pos).any() or (curve(pos[-1,0],pos[-1,1]) < 0):
            res[i,1] = 1
            path[i,1] = np.sum(np.linalg.norm(pos,axis=1))
        pos = getNesterovPath(x0,y0,iters,0.05,0.9)
        pos[pos < -10] = -10
        pos[np.isnan(pos)] = -10
        pos_diff = np.diff(pos)
        path[i,2] = np.sum(np.linalg.norm(pos,axis=1))
        if np.isnan(pos).any() or (curve(pos[-1,0],pos[-1,1]) < 0):
            res[i,2] = 1
            path[i,2] = np.sum(np.linalg.norm(pos,axis=1))
        pos = getADAMPath(x0,y0,iters,0.05,0.9,0.999,10e-8)
        pos[pos < -10] = -10
        pos[np.isnan(pos)] = -10
        pos_diff = np.diff(pos)
        if np.isnan(pos).any() or (curve(pos[-1,0],pos[-1,1]) < 0):
            res[i,3] = 1
            path[i,3] = np.sum(np.linalg.norm(pos,axis=1))

    print "result {}".format(np.mean(res,axis=0))
    mean_path = np.zeros(path.shape[1])
    for each in xrange(path.shape[1]):
        if (path[:,each] > 0).any():
            mean_path[each] = np.mean(path[path[:,each] > 0, each])

    print "path {}".format(mean_path)

    res_noise[n_idx] = np.mean(res,axis=0)
    path_noise[n_idx] = mean_path

plt.subplot(211)
for each in enumerate(res_noise.T):
    plt.plot(noise_list,each[1],label=names[each[0]])
    # Plot the surface.
plt.title("Res")
plt.legend(loc=4)
plt.subplot(212)
for each in enumerate(path_noise.T):
    plt.plot(noise_list,each[1],label=names[each[0]])
    # Plot the surface.
plt.title("path")
plt.legend(loc=4)
plt.show()

