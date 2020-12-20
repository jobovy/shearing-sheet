from jax import numpy as jnp
from jax.ops import index_update
from jax.experimental.ode import odeint

def integrate(vxvv,t,Omega0,A,*args):
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
    return odeint(_ode,vxvv,t,Omega0,A,*args)

def _ode(y,t,Omega0,A,*args):
    vx= y[:2]
    vv= y[2:]
    acc= jnp.array([vv[0],vv[1],
                    2.*Omega0*vv[1]+4.*A*Omega0*vx[0],
                    -2.*Omega0*vv[0]],dtype='float64')
    if len(args) > 0:
        # args = (R0,m,1/tanalpha,S0,S0',xc,phis)
        gamma= args[1]/args[0]*(vx[0]*args[2]+vx[1]+2.*A*args[5]*t)\
            -args[1]*args[6]
        sm= _smooth(t)
        acc= index_update(acc,2,acc[2]
                          -sm*(args[4]+args[7]*vx[0]+args[8]*vx[0]**2/2.)\
                             *jnp.cos(gamma)\
                          +sm*(args[3]+args[4]*vx[0]+args[7]*vx[0]**2/2.
                               +args[8]*vx[0]**3/6.)\
                             *jnp.sin(gamma)*args[1]/args[0]*args[2])
        acc= index_update(acc,3,acc[3]
                          +sm*(args[3]+args[4]*vx[0]+args[7]*vx[0]**2/2.
                               +args[8]*vx[0]**3/6.)\
                             *jnp.sin(gamma)*args[1]/args[0])
    return acc

def _smooth(t):
    tform= 13.250177056952785
    tsteady= tform/2.
    from jax import lax
    xi= 2.*(t-tform)/(tsteady-tform)-1.
    return lax.cond(t > tsteady,
                    xi,lambda xi: (3./16.*xi**5.-5./8*xi**3.+15./16.*xi+.5),
                    xi,lambda xi: 1.)

def integrate2(vxvv,t,Omega0,A,R0,Omega0pp,*args):
    """Integrate an orbit in the shearing sheet using the 2nd-order equations

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
R0: float
    Galactocentric radius of the center of the sheet
Omega0pp: float
    d^2 Omega/dR^2 at R_0

Returns
-------
array
    Integrated orbit in the shearing sheet
    """
    return odeint(_ode2,vxvv,t,Omega0,A,R0,Omega0pp,*args)

def _ode2(y,t,Omega0,A,R0,Omega0pp,*args):
    vx= y[:2]
    vv= y[2:]
    xR0= vx[0]/R0
    acc= jnp.array([vv[0],vv[1],
                    2.*Omega0*vv[1]+4.*A*Omega0*vx[0]*(1.+xR0)
                    +2.*xR0*vv[1]*Omega0+vv[1]**2/R0
                    -R0*Omega0*Omega0pp*vx[0]**2-4.*A**2.*vx[0]**2/R0,
                    (-2.*Omega0*vv[0]*(1.+xR0)-2.*vv[0]*vv[1]/R0)/(1.+2.*xR0)],
                   dtype='float64')
    if len(args) > 0:
        # args = (R0,m,1/tanalpha,S0,S0',xc,phis)
        # Needs to be adjusted properly for xc^2 and for the expansion
        # of the ln R/R0 term (and y??)
        gamma= args[1]/args[0]*(vx[0]*args[2]+vx[1]+R0*(Omega0-args[5])*t)\
            -args[1]*args[6]
        sm= _smooth(t)
        acc= index_update(acc,2,acc[2]
                          -sm*(args[4]+args[7]*vx[0]+args[8]*vx[0]**2/2.)\
                             *jnp.cos(gamma)\
                          +sm*(args[3]+args[4]*vx[0]+args[7]*vx[0]**2/2.
                               +args[8]*vx[0]**3/6.)\
                             *jnp.sin(gamma)*args[1]/args[0]*args[2])
        acc= index_update(acc,3,acc[3]
                          +(sm*(args[3]+args[4]*vx[0]+args[7]*vx[0]**2/2.
                               +args[8]*vx[0]**3/6.)\
                             *jnp.sin(gamma)*args[1]/args[0])/(1.+2.*xR0))
    return acc


def integratef(vxvv,t,Omega0,A,R0,Omega0pp,d4PhidR4,*args):
    """Integrate an orbit in the shearing sheet using the full equation with 3rd order potential

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
R0: float
    Galactocentric radius of the center of the sheet
Omega0pp: float
    d^2 Omega/dR^2 at R_0
d4PhidR4: float
    Fourth derivative of the axisymmetric potential

Returns
-------
array
    Integrated orbit in the shearing sheet
    """
    return odeint(_odef,vxvv,t,Omega0,A,R0,Omega0pp,d4PhidR4,*args,mxstep=500)

def _odef(y,t,Omega0,A,R0,Omega0pp,d4PhidR4,*args):
    vx= y[:2]
    vv= y[2:]
    acc= jnp.array([vv[0],vv[1],
                    (R0+vx[0])/R0**2.*(vv[1]+Omega0*R0)**2.\
                    -R0*Omega0**2-(Omega0**2-4.*Omega0*A)*vx[0]
                    -(R0*Omega0*Omega0pp+4*A**2/R0-4*Omega0*A/R0)*vx[0]**2
                    -d4PhidR4*vx[0]**3/6.,
                    (-2.*(R0+vx[0])/R0**2*(vv[1]+R0*Omega0)*vv[0])/(R0+vx[0])**2.*R0**2.],
                   dtype='float64')
    if len(args) > 0:
        # args = (R0,m,1/tanalpha,S0,S0',xc,phis)
        gamma= args[1]/args[0]*(vx[0]*args[2]+vx[1]+R0*(Omega0-args[5])*t)\
            -args[1]*args[6]
        sm= _smooth(t)
        acc= index_update(acc,2,acc[2]
                          -sm*(args[4]+args[7]*vx[0]+args[8]*vx[0]**2/2.)\
                             *jnp.cos(gamma)\
                          +sm*(args[3]+args[4]*vx[0]+args[7]*vx[0]**2/2.
                               +args[8]*vx[0]**3/6.)\
                             *jnp.sin(gamma)*args[1]/args[0]*args[2])
        acc= index_update(acc,3,acc[3]
                          +(sm*(args[3]+args[4]*vx[0]+args[7]*vx[0]**2/2.
                               +args[8]*vx[0]**3/6.)\
                             *jnp.sin(gamma)*args[1]/args[0])/(R0+vx[0])**2.*R0**2.)#(1.+2.*vx[0]/R0))
        """
        acc= index_update(acc,2,acc[2]
                          -sm*(args[4]+args[7]*vx[0]+args[8]*vx[0]**2/2.)\
#                          +sm*(3.*args[3]*(R0/(R0+vx[0]))**4.)\
                          *jnp.cos(gamma)\
                          +sm*(args[3]+args[4]*vx[0]+args[7]*vx[0]**2/2.
                               +args[8]*vx[0]**3/6.)\
                             *jnp.sin(gamma)*args[1]/args[0]*args[2])
        acc= index_update(acc,3,acc[3]
                          +sm*(args[3]+args[4]*vx[0]+args[7]*vx[0]**2/2.
                               +args[8]*vx[0]**3/6.)\
 #                          +sm*(args[3]*(R0/(R0+vx[0]))**3.)\
                               *jnp.sin(gamma)*args[1]/args[0]/(R0+vx[0])**2.*R0**2.)
        """
    return acc

