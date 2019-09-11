# GP NMPC problem setup
import numpy as np
from casadi import *

def specifications():
    ''' Specify Problem parameters '''
    tf              = 240.      # final time
    nk              = 12        # sampling points
    x0              = np.array([1.,150.,0.])
    Lsolver         = 'mumps'  #'ma97'  # Linear solver
    c_code          = False    # c_code

    return nk, tf, x0, Lsolver, c_code
### 0.175 +-0.05

def DAE_system():
    # Define vectors with names of states
    states     = ['x','n','q']
    nd         = len(states)
    xd         = SX.sym('xd',nd)
    for i in range(nd):
        globals()[states[i]] = xd[i]

    # Define vectors with names of algebraic variables
    algebraics = []
    na         = len(algebraics)
    xa         = SX.sym('xa',na)
    for i in range(na):
        globals()[algebraics[i]] = xa[i]

    # Define vectors with banes of input variables
    inputs     = ['L','Fn']
    nu         = len(inputs)
    u          = SX.sym("u",nu)
    for i in range(nu):
        globals()[inputs[i]] = u[i]

    # Define model parameter names and values
    modpar    = ['u_m', 'k_s', 'k_i', 'K_N', 'u_d', 'Y_nx', 'k_m', 'k_sq',
    'k_iq', 'k_d', 'K_Np']
    modparval = [0.0923*0.62, 178.85, 447.12, 393.10, 0.001, 504.49,
    2.544*0.62*1e-4, 23.51, 800.0, 0.281, 16.89]

    nmp       = len(modpar)
    uncertainty = SX.sym('uncp', nmp)
    for i in range(nmp):
        globals()[modpar[i]] = SX(modparval[i] + uncertainty[i])

    # Additive measurement noise
#    Sigma_v  = [400.,1e5,1e-2]*diag(np.ones(nd))*1e-6

    # Additive disturbance noise
#    Sigma_w  = [400.,1e5,1e-2]*diag(np.ones(nd))*1e-6

    # Initial additive disturbance noise
#    Sigma_w0 = [1.,150.**2,0.]*diag(np.ones(nd))*1e-3

    # Declare ODE equations (use notation as defined above)

    dx   = u_m * L/(L+k_s+L**2./k_i) * x * n/(n+K_N) - u_d*x
    dn   = - Y_nx*u_m* L/(L+k_s+L**2./k_i) * x * n/(n+K_N)+ Fn
    dq   = (k_m * L/(L+k_sq+L**2./k_iq) * x - k_d * q/(n+K_Np)) * (sign(500. - n)+1)/2 * (sign(x - 10.0)+1)/2

    ODEeq =  [dx, dn, dq]

    # Declare algebraic equations
    Aeq = []

    # Define control bounds
    u_min      = np.array([120., 0.]) # lower bound of inputs
    u_max      = np.array([400., 40.]) # upper bound of inputs

    # Define objective to be minimized
    t           = SX.sym('t')

    return xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, states, algebraics, inputs, nd, na, nu, nmp, modparval



def integrator_model():
    """
    This function constructs the integrator to be suitable with casadi environment, for the equations of the model
    and the objective function with variable time step.
     inputs: NaN
     outputs: F: Function([x, u, dt]--> [xf, obj])
    """

    xd, xa, u, uncertainty, ODEeq, Aeq, u_min, u_max, states, algebraics, inputs, nd, na, nu, nmp, modparval\
        = DAE_system()

    dae = {'x': vertcat(xd), 'z': vertcat(xa), 'p': vertcat(u, uncertainty),
           'ode': vertcat(*ODEeq), 'alg': vertcat(*Aeq)}
    opts = {'tf': 240/12}  # interval length
    F = integrator('F', 'idas', dae, opts)

    return F
