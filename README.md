# shearing-sheet

Simple [JAX](https://github.com/google/jax)-based implementation of the shearing sheet

## Author

Jo Bovy (University of Toronto) - bovy@astro.utoronto.ca

## Dependencies

Numpy, [JAX](https://github.com/google/jax), and galpy

## Usage

The math behind the shearing sheet is contained in [this notebook](nb/shearing-sheet.ipynb).

### Defining the sheet and coordinates in it

``shsh.util`` has utilities to define a shearing sheet based on a galpy potential. For example,
```
import shsh.util
from galpy.potential import LogarithmicHaloPotential
lp= LogarithmicHaloPotential(normalize=1.)
sheet= shsh.util.potential_to_sheet(lp,1.)
```
sets up the shearing sheet for a flat rotation curve at radius 1. The ``sheet`` here is a dictionary that holds the radius ``R0`` of the center, the angular frequency ``Omega0`` at the center, and the Oort constant ``A``.

To obtain shearing-sheet coordinates, you can use functions such as ``shsh.util.cyl_to_sheet`` or ``shsh.util.galpy_to_sheet``. The latter takes a galpy orbit and converts it to sheet coordinates (x,y,vx,vy). E.g.,
```
orb= Orbit().toPlanar()[0]
vxvv= shsh.util.galpy_to_sheet(orb,sheet)
```
computes the shearing-sheet coordinates of the Sun. You can similarly use ``shsh.util.sheet_to_galpy`` to convert back to a galpy orbit (arrays of sheet coordinates get mapped to an integated orbit).

### Integrating the equation of motion

To integrate the equation of motion and obtain the orbit in the shearing sheet, use ``shsh.integrate.integrate``. E.g., for the Sun's orbit
```
ts= numpy.linspace(0.,30.,1001)
out= shsh.integrate.integrate(vxvv,ts,sheet['Omega0'],sheet['A'])
```
``out`` is then a tuple that contains (x,y,vx,vy). To plot this and compare it to the full correct orbit in the logarithmic potential computed using ``galpy``, do
```
plot(out[0],out[1])
orb.turn_physical_off()
orb.integrate(ts,lp)
plot(*shsh.util.galpy_to_sheet(orb,sheet)[:2])
```
You can convert the shearing-sheet solution back to a ``galpy`` orbit and then plot it using ``galpy``'s convenient functions
```
conv= shsh.util.sheet_to_galpy(*out,ts,sheet)
conv.plot()
orb.plot(overplot=True)
```

### Using JAX

The shearing-sheet integration is done using JAX, so you can 'jit' it or define derivatives of the orbit integration. For example,
```
from jax import jit
jintegrate= jit(shsh.integrate.integrate)
```
Creates a jitted version of the integrator that works in the same way. To compute the gradient of the final x position of the solution, do
```
from jax import grad
gintegrate= grad(lambda x,*args: jintegrate(x,*args)[0][-1])
print(gintegrate(vxvv,ts,sheet['Omega0'],sheet['A']))
# (DeviceArray(1.9850945, dtype=float64), DeviceArray(0., dtype=float64),
#  DeviceArray(-0.70702823, dtype=float64), DeviceArray(0.9850945, dtype=float64))
```
