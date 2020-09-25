from jax import numpy as jnp
from jax.experimental.ode import odeint

def integrate(vxvv,t,Omega0,A):
    """Integrate an orbit in the shearing sheet

Parameters
----------
vxvv: array
    Initial condition
t: array
    Times over which to integrate, t[0] is the initial time
Omega0: float
    Rotational frequency
A: float
    Oort A

Returns
-------
array
    Integrated orbit in the shearing sheet
    """
    return odeint(_ode,vxvv,t,Omega0,A)

def _ode(y,t,Omega0,A):
    vx= y[:2]
    vv= y[2:]
    acc= jnp.array([vv[0],vv[1],
                    2.*Omega0*vv[1]+4.*A*Omega0*vx[0],
                    -2.*Omega0*vv[0]],dtype='float64')
    return acc
