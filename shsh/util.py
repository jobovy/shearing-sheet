import numpy
from galpy.orbit import Orbit

def cyl_to_sheet(R,phi,vR,vT,t,sheet):
    """Convert cylindrical coordinates to shearing-sheet coordinates

Parameters
----------
R: float or array
    Cylindrical radius
phi: float or array
    Cylindrical azimuthal angle
vR: float or array
    Galactocentric radial velocity
vT: float or array
    Galactocentric rotational velocity
t: float or array
    Time
sheet: dict
    Dictionary that describes the sheet

Returns
-------
float or array
    (x,y,vx,vy) sheet coordinates    
    """
    return (R-sheet['R0'],
            sheet['R0']*((phi-sheet['Omega0']*t + numpy.pi) \
                         % (2.*numpy.pi) - numpy.pi),
            vR,
            sheet['R0']*vT/R-sheet['R0']*sheet['Omega0'])

def galpy_to_sheet(orb,sheet):
    """Convert a galpy orbit to shearing-sheet coordinates

Parameters
----------
orb: galpy Orbit instance
    Orbit whose coordinates to convert (either initial condition or entire time series for an integrated orbit)
sheet: dict
    Dictionary that describes the sheet                                         
Returns
-------
float or array
    (x,y,vx,vy) sheet coordinates    
"""
    t= orb.time(use_physical=False)
    return cyl_to_sheet(orb.R(t,use_physical=False),
                        orb.phi(t,use_physical=False),
                        orb.vR(t,use_physical=False),
                        orb.vT(t,use_physical=False),
                        t,
                        sheet)

def sheet_to_cyl(x,y,vx,vy,t,sheet):
    """Convert from shearing-sheet coordinates to cylindrical coordinates

Parameters
----------
x: float or array
    x coordinate in the sheet
y: float or array
    y coordinate in the sheet
vx: float or array
    x velocity in the sheet
vy: float or array
    y velocity in the sheet
t: float or array
    Time
sheet: dict
    Dictionary that describes the sheet

Returns
-------
float or array
    (R,phi,vR,vT) cylindrical coordinates    
    """
    return (sheet['R0']+x,
            (y+sheet['R0']*sheet['Omega0']*t)/sheet['R0'],
            vx,
            (sheet['R0']+x)/sheet['R0']*(vy+sheet['R0']*sheet['Omega0']))

def sheet_to_galpy(x,y,vx,vy,t,sheet):
    """Convert shearing-sheet coordinates to a galpy Orbit

Parameters
----------
x: float or array
    x coordinate in the sheet
y: float or array
    y coordinate in the sheet
vx: float or array
    x velocity in the sheet
vy: float or array
    y velocity in the sheet
sheet: dict
    Dictionary that describes the sheet                                         
Returns
-------
galpy Orbit instance
    Orbit whose coordinates to convert (either initial condition or entire time series for an integrated orbit)
"""
    R,phi,vR,vT= sheet_to_cyl(x,y,vx,vy,t,sheet)
    if hasattr(R,'__len__') and len(R) > 1:
        out= Orbit([R[0],vR[0],vT[0],phi[0]])
        out.orbit= numpy.vstack((R,vR,vT,phi)).T
        out.t= t
        out._integrate_t_asQuantity= False
    else:
        out= Orbit([R,vR,vT,phi])
    return out
    
def potential_to_sheet(pot,R0):
    """Compute the sheet dictionary for a given axisymmetric potential

Parameters
----------
pot: galpy Potential or list of galpy Potential instances
    Axisymmetric potential to convert to the shearing sheet
R0: float
    Radius of the sheet's center

Returns
-------
dict
    Dictionary that describes the sheet
"""
    from galpy.potential import (omegac, evaluateRforces, evaluateR2derivs,
                                 vcirc)
    dvcircdR= 0.5*(-evaluateRforces(pot,R0,0.,use_physical=False)\
                   +R0*evaluateR2derivs(pot,R0,0.,use_physical=False))\
                   /vcirc(pot,R0,use_physical=False)
    return {'R0': R0,
            'Omega0': omegac(pot,R0,use_physical=False),
            'A': 0.5*(omegac(pot,R0,use_physical=False)-dvcircdR)}
