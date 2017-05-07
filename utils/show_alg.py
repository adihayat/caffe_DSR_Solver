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
'''
def curve(X,Y):
    return X**2 + 0.01*Y + Y**2

def curve_grad(X,Y,noise=0.10):
    return np.array((2*X + np.random.normal(0,noise,X.shape),0.01 + 2*Y + np.random.normal(0,noise,Y.shape)))
'''
def curve(X,Y):
    return X**2 + 0.5*Y**3 + 0.0005*Y

def curve_grad(X,Y,noise=0.1):
    return np.array((2*X + np.random.normal(0,noise,X.shape),1.5*Y**2 + 0.0005  +  np.random.normal(0,noise,Y.shape)))

def getSGDPath(x0,y0,iters,lr):
    pos = np.zeros((iters,2))
    pos[0,:] = np.array((x0,y0))
    for each in xrange(iters):
        if each == 0 :
            continue
        pos[each,:] = np.array((pos[each-1,0],pos[each-1,1])) - lr*(curve_grad(pos[each-1,0],pos[each-1,1]))
    return pos

def getSGDMomentumPath(x0,y0,iters,lr,m):
    pos = np.zeros((iters,2))
    pos[0,:] = np.array((x0,y0))
    momentum = np.zeros(2)
    for each in xrange(iters):
        if each == 0 :
            continue
        momentum = m * momentum + curve_grad(pos[each-1,0],pos[each-1,1])
        pos[each,:] = np.array((pos[each-1,0],pos[each-1,1])) - lr* momentum
    return pos

def getDSRMomentumPath(x0,y0,iters,lr,m,window_factor,target_ratio):
    pos = np.zeros((iters,2))
    pos[0,:] = np.array((x0,y0))
    momentum = np.zeros(2)
    line = np.zeros(2)
    path = 0
    for each in xrange(iters):
        if each == 0 :
            continue
        line = curve_grad(pos[each-1,0],pos[each-1,1]) + window_factor*line
        path = path * window_factor + np.linalg.norm(curve_grad(pos[each-1,0],pos[each-1,1]))
        ratio = np.linalg.norm(line) / path
        momentum = m * momentum + curve_grad(pos[each-1,0],pos[each-1,1])
        pos[each,:] = np.array((pos[each-1,0],pos[each-1,1])) - lr* momentum / (ratio + target_ratio)
    return pos


def Softmax(v):
    return np.exp(v)/np.sum(np.exp(v))

def getDSR2MomentumPath(x0,y0,iters,lr,m,window_factor,target_ratio):
    pos = np.zeros((iters,2))
    pos[0,:] = np.array((x0,y0))
    momentum = np.zeros(2)
    line = np.zeros(2)
    path = np.zeros(2)
    lr_fix = 1
    for each in xrange(iters):
        if each == 0 :
            continue
        g = curve_grad(pos[each-1,0], pos[each-1,1])
        momentum = m * momentum + g * (1-m)
        pos[each,:] = np.array((pos[each-1,0],pos[each-1,1])) - lr * momentum  * lr_fix / (1-m)
        diff = pos[each,:] - pos[each-1,:]
        line =  line * window_factor  +  momentum * (1 - window_factor)
        path =  path * (window_factor) + np.abs(momentum) * (1-window_factor)
        ratio = ((np.abs(line) / path))
        lr_fix = ratio / target_ratio


        #if np.isnan(ratio).any():
        #    ipdb.set_trace()
        print ratio, lr_fix
    return pos


def getADAMPath(x0,y0,iters,lr,beta1,beta2,eps_v):
    pos = np.zeros((iters,2))
    pos[0,:] = np.array((x0,y0))
    m = np.zeros(2)
    v = np.zeros(2)
    for each in xrange(iters):
        if each == 0 :
            continue
        g = curve_grad(pos[each-1,0] , pos[each-1,1] )
        m = beta1 * m + (1-beta1) * g
        v = beta2 * v + (1-beta2) * (g**2)
        m_caret = m /( 1- beta1)
        v_caret = v / (1- beta2)
        pos[each,:] = np.array((pos[each-1,0],pos[each-1,1])) - lr * m_caret / ( (v_caret**0.5) + eps_v )
    return pos

def getLinePoint(x0,y0,z0,color):

    line,  = ax.plot([x0],[y0],[z0+eps],color=color)
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
X = np.arange(-1, 1, 0.1)
Y = np.arange(-1, 1, 0.1)
X, Y = np.meshgrid(X, Y)
Z = curve(X,Y)


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
    CURVES.append((getLinePoint(x0,y0,z0,'r'),getSGDMomentumPath(x0,y0,iters,0.05,0.9)))
    CURVES.append((getLinePoint(x0,y0,z0,'g'),getDSRMomentumPath(x0,y0,iters,0.05,0.9,0.99,0.5)))
    CURVES.append((getLinePoint(x0,y0,z0,'y'),getDSR2MomentumPath(x0,y0,iters,0.05,0.9,0.999,0.5)))
    CURVES.append((getLinePoint(x0,y0,z0,'m'),getADAMPath(x0,y0,iters,0.05,0.9,0.999,10e-8)))

# Add a color bar which maps values to colors.


def animate(i):
    for C in CURVES:
        updateLinePoint(C[0][0],C[0][1],C[1],i)



ani = animation.FuncAnimation(fig, animate, np.arange(1, iters), init_func=init_func,
                              interval=100)


plt.show()
