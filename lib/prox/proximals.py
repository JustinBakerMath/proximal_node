import abc
import numpy as np
import torch

#TODO: change entire gradient by dividing entire gradient by dt - combines to lr

class FixedGridMethods(metaclass=abc.ABCMeta):
    def __init__(self, func, grad_tol=1e-7, device='cpu', **unused_kwargs):
        self.func = func
        self.grad_tol = grad_tol
        self.grad = None
        self.device = device
        
    def check_halt(self):
            return False

class IdentityMethod(FixedGridMethods):
    def gradient(self, itr, dt, tk, tn, hk, zn):
        self.grad = -self.func(tn,zn)
        return self.grad

"""
Backward Euler
    - hatls: Suggested O(s^1.8) where s is the integration step size
"""
class backward_euler(FixedGridMethods):
    def gradient(self, itr, dt, tk, tn, hk, zn):
        self.grad = -dt*self.func(tn,zn)+zn-hk
        return  self.grad
    def check_halt(self):
        if torch.norm(self.grad)<self.grad_tol:
            return True
        else:
            return False
"""
Crank Nicolson
    - graident: Modified Crank-Nicolson scheme
    - halts: 
"""
class cn_implicit(FixedGridMethods):
    def gradient(self, itr, dt, tk, tn, hk, zn):
        if itr==0:
            self.grad_hk = -self.func(tk,hk)
        self.grad = (dt/2)*(-self.func(tn,zn))+(dt/2)*self.grad_hk+zn-hk
        return self.grad
    def check_halt(self):
        if torch.norm(self.grad)<self.grad_tol:
            return True
        else:
            return False


class VariableOrderMethods(metaclass=abc.ABCMeta):
    def __init__(self, func, grad_tol, device, order=None, **unused_kwargs):
        self.func = func
        self.order = order
        self.grad_tol = grad_tol
        self.device = device
    def check_halt(self):
        return False

"""
Implicit Euler
    -
"""
class euler3(VariableOrderMethods):
    def __init__(self, func, grad_tol, device, **unused_kwargs):
        super(euler3, self).__init__(func,grad_tol,device)

        self.gamma = np.array([[5,0,0],
                                  [-2,6,0],
                                  [-2, 3/14, 44/7]])
        self.m = 0
        self.M = 3
        self.z = [None]*(self.M+1)

    def gradient(self, itr, dt, tk, tn, h, zn):
        self.z[self.m]=zn
        sum_ = -dt*self.func(tn,zn)
        for i in range(self.m):
            sum_ = sum_ + self.gamma[self.m-1,i]*(zn-self.z[i])

        # print(self.m,sum_)
        self.grad = sum_
        return  self.grad
    def check_halt(self):
        if torch.norm(self.grad)<self.grad_tol:
            return True
        else:
            return False

class euler5(VariableOrderMethods):
    def __init__(self, func, grad_tol, device, **unused_kwargs):
        super(euler5, self).__init__(func,grad_tol,device)

        self.gamma = np.array([[11.17,0,0,0,0,0],
                                  [-7.5,19.43,0,0,0,0],
                                  [-1.05, -4.75, 13.98,0,0,0],
                                  [1.8,.05,-7.83,13.8,0,0],
                                  [6.2,-7.17,-1.33,1.63,11.52,0],
                                  [-2.83,4.69,2.46,-11.55,6.68,11.95]])
        self.m = 0
        self.M = 5
        self.z = [None]*(self.M+1)

    def gradient(self, itr, dt, tk, tn, h, zn):
        self.z[self.m]=zn
        sum_ = -dt*self.func(tn,zn)
        for i in range(self.m):
            sum_ = sum_ + self.gamma[self.m-1,i]*(zn-self.z[i])

        # print(self.m,sum_)
        self.grad = sum_
        return  self.grad
    def check_halt(self):
        if torch.norm(self.grad)<self.grad_tol:
            return True
        else:
            return False

class MultiStepMethods(metaclass=abc.ABCMeta):
    def __init__(self, func, grad_tol, device, order, **unused_kwargs):
        self.func = func
        self.order = order
        self.grad_tol = grad_tol
        self.device = device
        self.h = None
    def set_h(self,h):
        self.h = h
    def check_halt(self):
        return False


"""
Backward Differentiation Formula (2)
    -
"""
class bdf2(MultiStepMethods):
    def __init__(self, func, grad_tol, device, order=2, **unused_kwargs):
        super(bdf2, self).__init__(func, grad_tol,device,order)
    def gradient(self, itr, dt, tk, tn, hk, zn):
        h=self.h
        if len(h)<2:
            self.grad = -dt*self.func(tn,zn)+zn-h[-1]
            return self.grad
        else:
            self.grad = -dt*self.func(tn,zn)+2*(zn-h[-1])-(zn-h[-2])/2
            return self.grad

    def check_halt(self):
        if torch.norm(self.grad)<self.grad_tol:
            return True
        else:
            return False

"""
Backward Differentiation Formula (3)
    -
"""
class bdf3(MultiStepMethods):
    def __init__(self, func, grad_tol, device, order=3, **unused_kwargs):
        super(bdf3, self).__init__(func, grad_tol,device,order)
    def gradient(self, itr, dt, tk, tn, hk, zn):
        h=self.h
        if len(h)>=3:
            self.grad = -dt*self.func(tn,zn)+3*(zn-h[-1])-(3/2)*(zn-h[-2])+(1/3)*(zn-h[-3])
            return self.grad
        elif len(h)==2:
            self.grad = -dt*self.func(tn,zn)+2*(zn-h[-1])-(zn-h[-2])/2
            return self.grad
        else:
            self.grad = -dt*self.func(tn,zn)+zn-h[-1]
            return self.grad

    def check_halt(self):
        if torch.norm(self.grad)<self.grad_tol:
            return True
        else:
            return False

"""
Backward Differentiation Formula (4)
    -
"""
class bdf4(MultiStepMethods):
    def __init__(self, func, grad_tol, device, order=4, **unused_kwargs):
        super(bdf4, self).__init__(func, grad_tol,device,order)
    def gradient(self, itr, dt, tk, tn, hk, zn):
        h=self.h
        if len(h)>=4:
            self.grad = -dt*self.func(tn,zn)+4*(zn-h[-1])-3*(zn-h[-2])+(4/3)*(zn-h[-3])-(1/4)*(zn-h[-4])
            return self.grad
        elif len(h)==3:
            self.grad = -dt*self.func(tn,zn)+3*(zn-h[-1])-(3/2)*(zn-h[-2])+(1/3)*(zn-h[-3])
            return self.grad
        elif len(h)==2:
            self.grad = -dt*self.func(tn,zn)+2*(zn-h[-1])-(zn-h[-2])/2
            return self.grad
        else:
            self.grad = -dt*self.func(tn,zn)+zn-h[-1]
            return self.grad

    def check_halt(self):
        if torch.norm(self.grad)<self.grad_tol:
            return True
        else:
            return False
class bdfs(MultiStepMethods):
    def __init__(self, func, grad_tol, device, order, **unused_kwargs):
        super(bdfs, self).__init__(func, grad_tol,device,order)
    def gradient(self, itr, dt, tk, tn, hk, zn):
        h=self.h
        if len(h)>=4:
            self.grad = -dt*self.func(tn,zn)+4*(zn-self.h[-1])-3*(zn-h[-2])+(4/3)*(zn-h[-3])-(1/4)*(zn-h[-4])
            return self.grad
        elif len(h)==3:
            self.grad = -dt*self.func(tn,zn)+3*(zn-h[-1])-(3/2)*(zn-h[-2])+(1/3)*(zn-h[-3])
            return self.grad
        elif len(h)==2:
            self.grad = -dt*self.func(tn,zn)+2*(zn-h[-1])-(zn-h[-2])/2
            return self.grad
        else:
            self.grad = -dt*self.func(tn,zn)+zn-h[-1]
            return self.grad

    def check_halt(self):
        if torch.norm(self.grad)<self.grad_tol:
            return True
        else:
            return False