import numpy as np

GlobalNoisePower = 0.0003

'''
def curve(X,Y):
    return 0.1*X**2 + 0.01*Y + Y**2

def curve_grad(X,Y,noise=0.10):
    return np.array((0.2*X + np.random.normal(0,noise,X.shape),0.01 + 2*Y + np.random.normal(0,noise,Y.shape)))
'''
def curve(X,Y):
    return 0.1*X**2 + 0.2*np.cos(Y) + 0.1*(Y**2)*(Y>0) + 0.00001*Y

def curve_grad(X,Y):
    return np.array((0.2*X + np.random.normal(0,GlobalNoisePower,X.shape),0.00001 -0.2*np.sin(Y) + 0.2*Y*(Y>0) + np.random.normal(0,GlobalNoisePower,Y.shape)))

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
        ratio = ((np.abs(line) / path))**0.5
        lr_fix = ratio / target_ratio


        #if np.isnan(ratio).any():
        #    ipdb.set_trace()
        #print ratio, lr_fix
    return pos


def getNesterovPath(x0,y0,iters,lr,m):
    pos = np.zeros((iters,2))
    pos[0,:] = np.array((x0,y0))
    momentum = np.zeros(2)
    for each in xrange(iters):
        if each == 0 :
            continue
        pos_for_grad = pos[each-1,:] - m * momentum
        g = curve_grad(pos_for_grad[0],pos_for_grad[1])
        momentum = m * momentum + g
        pos[each,:] = np.array((pos[each-1,0],pos[each-1,1])) - lr * momentum

        #if np.isnan(ratio).any():
        #    ipdb.set_trace()
        #print ratio, lr_fix
    return pos


def getDSR3MomentumPath(x0,y0,iters,lr,beta1,beta2,beta3 , eps_v):
    pos = np.zeros((iters,2))
    pos[0,:] = np.array((x0,y0))
    m = np.zeros(2)
    v = np.zeros(2)
    lr_fix = 1
    line = np.zeros(2)
    path = np.zeros(2)
    for each in xrange(iters):
        if each == 0 :
            continue
        g = curve_grad(pos[each-1,0] , pos[each-1,1] )
        m = beta1 * m + (1-beta1) * g
        v = beta2 * v + (1-beta2) * (g**2)
        m_caret = m /( 1- beta1)
        v_caret = v / (1- beta2)
        pos[each,:] = np.array((pos[each-1,0],pos[each-1,1])) - lr * m_caret * lr_fix / ( (v_caret**0.5) + eps_v )
        diff = pos[each,:] - pos[each-1,:]
        line =  line * beta3  +  diff * (1 - beta3)
        path =  path * (beta3) + np.abs(diff) * (1- beta3)
        ratio = ((np.abs(line) / path))
        lr_fix = ratio
        #print lr_fix
    return pos

def getDSR4MomentumPath(x0,y0,iters,lr,m,beta,target_ratio,mod,power):
    pos = np.zeros((iters,2))
    pos[0,:] = np.array((x0,y0))
    momentum = np.zeros(2)
    lr_fix = 1
    line = np.zeros(2)
    path = np.zeros(2)
    prev = pos[0,:]
    for each in xrange(iters):
        if each == 0 :
            continue
        pos_for_grad = pos[each-1,:] - m * momentum
        g = curve_grad(pos_for_grad[0],pos_for_grad[1])
        momentum = m * momentum + g
        pos[each,:] = np.array((pos[each-1,0],pos[each-1,1])) - lr * momentum * lr_fix
        if (each % mod == 0):
            diff = pos[each,:] - prev
            line =  line * beta  +  diff * (1 - beta)
            path =  path * (beta) + np.abs(diff) * (1- beta)
            ratio = ((np.abs(line) / (path + 1e-16)))
            lr_fix = (ratio / target_ratio)**power
            prev = pos[each,:]

    return pos

def getDSR5MomentumPath(x0,y0,iters,lr,beta1,beta2,eps_v,target_ratio):
    pos = np.zeros((iters,2))
    pos[0,:] = np.array((x0,y0))
    momentum = np.zeros(2)
    lr_fix = 1
    line = np.zeros(2)
    path = np.zeros(2)
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
        pos[each,:] = np.array((pos[each-1,0],pos[each-1,1])) - lr * lr_fix * m_caret / ( (v_caret**0.5) + eps_v )
        diff = pos[each,:] - pos[each-1,:]
        line =  line * beta1  +  diff * (1 - beta1)
        path =  path * (beta1) + np.linalg.norm(diff) * (1- beta1)
        ratio = ((np.abs(line) / (path)))**0.25
        lr_fix = ratio / target_ratio

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
