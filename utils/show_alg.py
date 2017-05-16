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

def getLinePoint(x0,y0,z0,color,title):

    line,  = ax.plot([x0],[y0],[z0+eps],color=color,label=title)
    point,  = ax.plot([x0], [y0], [z0+eps], c=color, marker='o')
    return line , point

def updateLinePoint(line,point,pos,i):
    line.set_data(pos[:i,0],pos[:i,1])
    line.set_3d_properties(curve(pos[:i,0],pos[:i,1])+eps)
    point.set_data([pos[i,0]],[pos[i,1]])
    point.set_3d_properties([curve(pos[i,0],pos[i,1])+eps])


CURVES = []

iters = 100

fig = plt.figure()
ax = fig.gca(projection='3d')

# Make data.
X = np.arange(-2, 2, 0.1)
Y = np.arange(-3, 2, 0.1)
X, Y = np.meshgrid(X, Y)
Z = curve(X,Y)
alg.GlobalNoisePower = 0.0002

x0 = 1
y0 = 0
eps = 0.01
z0 = curve(x0,y0)

# Plot the surface.

def init_func():
    global CURVES
    global surf
    CURVES = []
    print "CLEAR"
    ax.clear()
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,rstride=2, cstride=2,
                           linewidth=0, antialiased=False,alpha=0.75)
    CURVES.append((getLinePoint(x0,y0,z0,'r','SGD'),getSGDMomentumPath(x0,y0,iters,0.05,0.9)))
    CURVES.append((getLinePoint(x0,y0,z0,'g','DSR'),getDSR4MomentumPath(x0,y0,iters,0.05,0.9,0.9,0.5,10,1.5)))
    CURVES.append((getLinePoint(x0,y0,z0,'y','NST'),getNesterovPath(x0,y0,iters,0.05,0.9)))
    CURVES.append((getLinePoint(x0,y0,z0,'m','ADM'),getADAMPath(x0,y0,iters,0.05,0.9,0.999,10e-8)))
    plt.legend()

# Add a color bar which maps values to colors.


def animate(i):
    for C in CURVES:
        updateLinePoint(C[0][0],C[0][1],C[1],i)



ani = animation.FuncAnimation(fig, animate, np.arange(1, iters), init_func=init_func,
                              interval=100)


plt.show()
