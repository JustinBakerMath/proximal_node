import numpy as np


from enum import Enum
import numpy as np
import torch
import warnings
from .solvers import multi_step

ORDER = {
    'identity':0,
    'euler':0,
    'euler3':3,
    'euler5':5,
    'bdf2':2,
    'bdf3':3,
    'bdf4':4,
    'crank_nicolson':0,
}


""" NORMS """

def _rms_norm(tensor):
    return tensor.pow(2).mean().sqrt()

def _zero_norm(tensor):
    return 0.

def _mixed_norm(tensor_tuple):
    if len(tensor_tuple) == 0:
        return 0.
    return max([_rms_norm(tensor) for tensor in tensor_tuple])

def _assert_increasing(name, t):
    assert (t[1:] > t[:-1]).all(), '{} must be strictly increasing or decreasing'.format(name)


def _assert_floating(name, t):
    if not torch.is_floating_point(t):
        raise TypeError('`{}` must be a floating point Tensor but is a {}'.format(name, t.type()))


def _flat_to_shape(tensor, length, shapes):
    tensor_list = []
    total = 0
    for shape in shapes:
        next_total = total + shape.numel()
        # It's important that this be view((...)), not view(...). Else when length=(), shape=() it fails.
        tensor_list.append(tensor[..., total:next_total].view((*length, *shape)))
        total = next_total
    return tuple(tensor_list)


class _TupleFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        f = self.base_func(t, _flat_to_shape(y, (), self.shapes))
        return torch.cat([f_.reshape(-1) for f_ in f])


class _TupleInputOnlyFunc(torch.nn.Module):
    def __init__(self, base_func, shapes):
        super(_TupleInputOnlyFunc, self).__init__()
        self.base_func = base_func
        self.shapes = shapes

    def forward(self, t, y):
        return self.base_func(t, _flat_to_shape(y, (), self.shapes))


class _ReverseFunc(torch.nn.Module):
    def __init__(self, base_func, mul=1.0):
        super(_ReverseFunc, self).__init__()
        self.base_func = base_func
        self.mul = mul

    def forward(self, t, y):
        return self.mul * self.base_func(-t, y)


class Perturb(Enum):
    NONE = 0
    PREV = 1
    NEXT = 2


class _PerturbFunc(torch.nn.Module):

    def __init__(self, base_func):
        super(_PerturbFunc, self).__init__()
        self.base_func = base_func

    def forward(self, t, y, *, perturb=Perturb.NONE):
        assert isinstance(perturb, Perturb), "perturb argument must be of type Perturb enum"
        # This dtype change here might be buggy.
        # The exact time value should be determined inside the solver,
        # but this can slightly change it due to numerical differences during casting.
        t = t.to(y.dtype)
        if perturb is Perturb.NEXT:
            # Replace with next smallest representable value.
            t = _nextafter(t, t + 1)
        elif perturb is Perturb.PREV:
            # Replace with prev largest representable value.
            t = _nextafter(t, t - 1)
        else:
            # Do nothing.
            pass
        return self.base_func(t, y)
class _StitchGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, out):
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None

def _nextafter(x1, x2):
    with torch.no_grad():
        if hasattr(torch, "nextafter"):
            out = torch.nextafter(x1, x2)
        else:
            out = np_nextafter(x1, x2)
    return _StitchGradient.apply(x1, out)

def np_nextafter(x1, x2):
    warnings.warn("torch.nextafter is only available in PyTorch 1.7 or newer."
                  "Falling back to numpy.nextafter. Upgrade PyTorch to remove this warning.")
    x1_np = x1.detach().cpu().numpy()
    x2_np = x2.detach().cpu().numpy()
    out = torch.tensor(np.nextafter(x1_np, x2_np)).to(x1)
    return out


def _check_timelike(name, timelike, can_grad):
    assert isinstance(timelike, torch.Tensor), '{} must be a torch.Tensor'.format(name)
    _assert_floating(name, timelike)
    assert timelike.ndimension() == 1, "{} must be one dimensional".format(name)
    if not can_grad:
        assert not timelike.requires_grad, "{} cannot require gradient".format(name)
    diff = timelike[1:] > timelike[:-1]
    assert diff.all() or (~diff).all(), '{} must be strictly increasing or decreasing'.format(name)


def _check_inputs(func, y0, t, opt_method, prox_method, int_step, opt_step, options, OPTIMIZERS, SOLVERS):

    shapes = None
    is_tuple = not isinstance(y0, torch.Tensor)
    if is_tuple:
        assert isinstance(y0, tuple), 'y0 must be either a torch.Tensor or a tuple'
        shapes = [y0_.shape for y0_ in y0]
        y0 = torch.cat([y0_.reshape(-1) for y0_ in y0])
        func = _TupleFunc(func, shapes)
    _assert_floating('y0', y0)

    if options is None:
        options = {}
    else:
        options = options.copy()

    if 'tol' not in options:
        options['tol']=1e-6
    if 'opt_iters' not in options:
        options['opt_iters']=100
    if 'device' not in options:
        options['device']='cpu'

    if opt_method is None:
        opt_method = 'grad_desc'
    if opt_method not in OPTIMIZERS:
        raise ValueError('Invalid method "{}". Must be one of {}'.format(opt_method,
                                                                         '{"' + '", "'.join(OPTIMIZERS.keys()) + '"}.'))
    if prox_method is None:
        prox_method = 'euler'
    if prox_method not in SOLVERS:
        raise ValueError('Invalid method "{}". Must be one of {}'.format(prox_method,
    
                                                                         '{"' + '", "'.join(SOLVERS.keys()) + '"}.'))
    if SOLVERS[prox_method] == multi_step:
        is_multistep = True
    else:
        is_multistep = False

    if 'order' in options:
        order = options['order']
    else:
        order = ORDER[prox_method]

    if 'grad_tol' not in options:
        options['grad_tol']=int_step**1.8
        

    # Backward compatibility: Allow t and y0 to be on different devices
    if t.device != y0.device:
        warnings.warn("t is not on the same device as y0. Coercing to y0.device.")
        t = t.to(y0.device)
    # ~Backward compatibility

    if is_tuple:

        if 'norm' in options:
            # If the user passed a norm then get that...
            norm = options['norm']
        else:
            # ...otherwise we default to a mixed Linf/L2 norm over tupled input.
            norm = _mixed_norm

        def _norm(tensor):
            y = _flat_to_shape(tensor, (), shapes)
            return norm(y)
        options['norm'] = _norm
    else:
        if 'norm' in options:
            pass
        else:
            options['norm'] = _rms_norm

    #Normalize Time
    _check_timelike('t', t, True)
    t_is_reversed = False
    if len(t) > 1 and t[0] > t[1]:
        t_is_reversed = True

    if t_is_reversed:
        t = -t
        func = _ReverseFunc(func, mul=-1.0)

    # Can only do after having normalised time
    _assert_increasing('t', t)

    func = _PerturbFunc(func)


    return shapes, func, y0, t, opt_method, prox_method, order, options, t_is_reversed, is_multistep
