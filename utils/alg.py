import numpy as np
'''
def curve(X,Y):
    return X**2 + 0.01*Y + Y**2

def curve_grad(X,Y,noise=0.10):
    return np.array((2*X + np.random.normal(0,noise,X.shape),0.01 + 2*Y + np.random.normal(0,noise,Y.shape)))
'''
def curve(X,Y):
    return X**2 + 0.5*Y**3 + 0.001*Y

def curve_grad(X,Y,noise=0.01):
    return np.array((2*X + np.random.normal(0,noise,X.shape),1.5*Y**2 + 0.001  +  np.random.normal(0,noise,Y.shape)))

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
        ratio = ((np.abs(line) / path))
        lr_fix = ratio / target_ratio


        #if np.isnan(ratio).any():
        #    ipdb.set_trace()
        #print ratio, lr_fix
    return pos

def getDSR3MomentumPath(x0,y0,iters,lr,m,window_factor,target_ratio):
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
        ratio = 1.0/(path - np.abs(line) + target_ratio)
        lr_fix = ratio


        #if np.isnan(ratio).any():
        #    ipdb.set_trace()
        #print ratio, lr_fix
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
