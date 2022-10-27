from .proximals import *
from .misc import _check_inputs,  _flat_to_shape
from .optimizers import *
from .solvers import *

OPTIMIZERS = {
    'identity':IdentityOptimizer,
    'grad_desc':grad_desc,
    'fletch_reeves':grad_desc_fr,
    'nesterov':nesterov,
    'nesterov_restart':nesterov_restart,
    'lbfgs':lbfgs,
    'bbstep':bbstep
}

PROXIMALS = {
    'identity':IdentityMethod,
    'euler':backward_euler,
    'euler3':euler3,
    'euler5':euler5,
    'bdf2':bdf2,
    'bdf3':bdf3,
    'bdf4':bdf4,
    'crank_nicolson':cn_implicit,
}

SOLVERS = {
    'identity':IdentitySolver,
    'euler':single_step,
    'crank_nicolson':single_step,
    'bdf2':multi_step,
    'bdf3':multi_step,
    'bdf4':multi_step,
    'euler3':variable_order,
    'euler5':variable_order,
}


def odeint(func, y0, t, *, opt_method, prox_method, int_step, opt_step, options=None):
    """
    ---------------------
    odeint - initial value problem solver
    ---------------------
    func - function for integration
    y0 - initial value
    t - times at which exact solution is desired
    ---------------------
    opt_method - optimization method
    prox_method - integration method
    int_step - step size for integration
    opt_step - step size for optimization
    options - dictionary of extra options
    -------------------------------------
    options:
            opt_iters - max iterations for optimization method
    """

    shapes, func, y0, t, opt_method, prox_method, order, options, t_is_reversed, is_multistep = _check_inputs(func, y0, t, opt_method, prox_method, int_step, opt_step, options, OPTIMIZERS, SOLVERS)

    proximal = PROXIMALS[prox_method](func, grad_tol=options['grad_tol'], device=options['device'], order=order)

    optimizer = OPTIMIZERS[opt_method](proximal, step_size=opt_step,
                                        tol=options['tol'], max_itrs=options['opt_iters'],
                                        device=options['device'],
                                        is_multistep=is_multistep)


    solver = SOLVERS[prox_method](optimizer,y0,step_size=int_step,is_multistep=is_multistep,order=order)

    solution = solver.integrate(t)

    if shapes is not None:
        solution = _flat_to_shape(solution, (len(t),), shapes)

    return solution