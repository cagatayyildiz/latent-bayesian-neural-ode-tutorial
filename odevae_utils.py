from matplotlib import animation, rc
import torch, torch.nn as nn
from torchdiffeq import odeint

    
def integrate(ode_rhs,x0,t):
    ''' Performs forward integration with Dopri5 (RK45) solver with rtol=1e-6 & atol=1e-7.
        Higher-order solvers as well as smaller tolerances would give more accurate solutions.
        Inputs:
            ode_rhs    time differential function with signature ode_rhs(t,s)
            x0 - [N,d] initial values
            t  - [T]   integration time points
        Retuns:
            xt - [N,T,d] state trajectory computed at t
    '''
    return odeint(ode_rhs, x0, t, method='dopri5', rtol=1e-6, atol=1e-7).permute(1,0,2)


def integrate_L(ode_rhs, x0, t, L=1, method='dopri5', rtol=1e-6, atol=1e-7):
    ''' The same as the integrate function above except we concurrently integrate L vector fields. 
        Note that we replicate the initial values L times (one per vector field).
        Inputs:
            ode_rhs    time differential function with signature ode_rhs(t,s) with s of shape [L,N,d]
            x0 - [N,d] initial values
            t  - [T]   integration time points
        Retuns:
            xt - [L,N,T,d] state trajectory computed at t
    '''
    x0 = torch.stack([x0]*L)
    return odeint(ode_rhs, x0, t, method=method, rtol=rtol, atol=atol).permute(1,2,0,3)


def get_minibatch(t, Y, Nsub=None, tsub=None):
    ''' Extract Nsub subsequences with length tsub.
        Nsub=None ---> Pick all sequences
        tsub=None ---> No subsequences
        Inputs:
            t - [T]       integration time points (original dataset)
            Y - [N,T,...] observed sequences (original dataset)
            tsub - int    subsequence length
            Nsub - int    number of (sub)sequences 
        Returns:
            [tsub]       integration time points (in this minibatch)
            [N,tsub,...] observed (sub)sequences (in this minibatch)
            
            
    '''
    [N,T] = Y.shape[:2]
    Y_   = Y if Nsub is None else Y[torch.randperm(N)[:Nsub]] # pick Nsub random sequences
    t0   = 0 if tsub is None else torch.randint(0,1+len(t)-tsub,[1]).item()  # pick the initial value
    tsub = T if tsub is None else tsub
    tsub, Ysub = t[t0:t0+tsub], Y_[:,t0:t0+tsub] # pick subsequences
    return tsub, Ysub 