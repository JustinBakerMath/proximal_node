import abc
import torch

class ODESolver(metaclass=abc.ABCMeta):
    order:int

    def __init__(self, optimizer, h0, step_size, is_multistep, **unused_kwargs):

        if step_size is None:
            step_size = .1
        self.optimizer = optimizer
        self.h0 = h0
        self.is_multistep = is_multistep
        self.hk = h0
        self.grid_constructor = self._grid_constructor_from_step_size(optimizer, h0, step_size)

    @staticmethod
    def _grid_constructor_from_step_size(optimizer, hk, step_size):
        def _grid_constructor(t):
            start_time = t[0]
            end_time = t[-1]

            niters = torch.ceil((end_time - start_time) / step_size + 1).item()
            t_infer = torch.arange(0,niters, dtype=t.dtype, device=t.device) * step_size + start_time
            t_infer[-1] = t[-1]

            return t_infer
        return _grid_constructor

    def integrate(self, t):
        time_grid = self.grid_constructor(t)
        assert time_grid[0] == t[0] and time_grid[-1] == t[-1]

        solution = torch.empty(len(t), *self.h0.shape, dtype=self.h0.dtype, device=self.h0.device)
        solution[0] = self.h0

        j = 1

        for t0, t1 in zip(time_grid[:-1], time_grid[1:]):
            dt = t1 - t0
            hk1 = self._step_func(self.optimizer, t0, dt, t1, self.hk)
            
            while j < len(t) and t1 >= t[j]:
                solution[j] = self._linear_interp(t0, t1, self.hk[-1], hk1, t[j])
                j += 1
            
            hk=hk1
            self.hk=hk
            if self.is_multistep:
                self.h_list = self.h_list + [hk]
                self.prune_h()
        return solution

    def _cubic_hermite_interp(self, t0, y0, f0, t1, y1, f1, t):
        h = (t - t0) / (t1 - t0)
        h00 = (1 + 2 * h) * (1 - h) * (1 - h)
        h10 = h * (1 - h) * (1 - h)
        h01 = h * h * (3 - 2 * h)
        h11 = h * h * (h - 1)
        dt = (t1 - t0)
        return h00 * y0 + h10 * dt * f0 + h01 * y1 + h11 * dt * f1

    def _linear_interp(self, t0, t1, y0, y1, t):
        if t == t0:
            return y0
        if t == t1:
            return y1
        slope = (t - t0) / (t1 - t0)
        return y0 + slope * (y1 - y0)

class IdentitySolver(ODESolver):
    def _step_func(self, optimizer, tk, dt, t1, hk):
        f0 = optimizer.minimize(dt, tk, hk)
        return  f0

class single_step(ODESolver):
    def _step_func(self, optimizer, tk, dt, t1, hk):
        f0 = optimizer.minimize(dt, tk, hk)
        return  f0

class multi_step(ODESolver):
    def __init__(self, optimizer, h0, step_size, is_multistep, order, **unused_kwargs):
        super(multi_step, self).__init__(optimizer,h0,step_size,is_multistep)
        self.h_list = [h0]
        self.order = order
        self.len = 1
    def _step_func(self, optimizer, tk, dt, t1, hk):
        optimizer.proximal.set_h(self.h_list)
        f0 = optimizer.minimize(dt, tk, hk)
        return  f0
    def prune_h(self):
        if self.len > self.order:
            self.h = self.h[:-self.order]
            self.len=self.order
            print(self.len)

class variable_order(ODESolver):
    def _step_func(self, optimizer, tk, dt, t1, hk):
        optimizer.proximal.z[0]=hk
        for m in range(1,optimizer.proximal.M+1):
            optimizer.proximal.m = m
            f0 = optimizer.minimize(dt, tk, hk)
        return  f0