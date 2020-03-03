"""
Compare the velocity field from avortex doublet line obtained using numerinal or analytical integration

"""
#--- Legacy python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# --- General
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
# --- Local
from wiz.VortexDoublet import *
try:
    from pybra.curves import streamQuiver
    from pybra.tictoc import Timer
except:
    raise Exception('This script requires the package `pybra` from https://github.com/ebranlard/pybra/')

# --- Parameters
R      = 1
gamma_t=-10
XLIM  = np.asarray([0.01*R,5*R ]) # 
ZLIM  = np.asarray([-5*R,1*R]) # 
ZMax  = 100*R  # Extent of line for numerical integration
nQuad = 1000   # Numbero of quadrature points for numerical integration
nx    = 200                    # Number of points for velocity evaluation
nz    = nx

# --- Derived parameters
dmz_dz = gamma_t * R**2 * np.pi # doublet intensity per length

# --- Flow field and speed
zs = np.linspace(ZLIM[0],ZLIM[1],nx).astype(np.float32)
xs = np.linspace(XLIM[0]*1.1,XLIM[1]*1.1,nx).astype(np.float32)
[Z,X]=np.meshgrid(zs,xs)
Y=X*0
with Timer('Theory'):
    urt,uzt = doublet_line_polar_u    (X,Z,dmz_dz)
with Timer('Numerical'):
    urn,uzn = doublet_line_polar_u_num(X,Z,dmz_dz, 0, ZMax, nQuad)

Speed_t=np.log(np.sqrt(uzt**2+urt**2))
Speed_n=np.log(np.sqrt(uzn**2+urn**2))
# Speed_t=uzt
# Speed_n=uzn
supermin=min(np.min(Speed_t),np.min(Speed_n))
supermax=max(np.max(Speed_t),np.max(Speed_n))/2
print(supermin)
print(supermax)

# --- Plotting
fig,ax = plt.subplots(1,2, sharex=True, sharey=True)
levels=np.linspace(supermin,supermax,10)
im1=ax[0].contourf(Z/R,X/R,Speed_t, levels=levels) # the easiest way to get contourf to comply with cmap is to give levels
im2=ax[1].contourf(Z/R,X/R,Speed_n, levels=levels) # the easiest way to get contourf to comply with cmap is to give levels
ax[0].plot([0,ZLIM[1]],[0,0],'k-',lw=2)
ax[1].plot([0,ZLIM[1]],[0,0],'k-',lw=2)

# Streamlines and quiver
yseed=np.linspace(XLIM[0]*1.1/R,XLIM[1]*0.9/R,7)
start=np.array([yseed*0+(ZLIM[0]+ZLIM[1])/(2*R),yseed])
sp=ax[0].streamplot(zs/R,xs/R,uzt,urt,color='k',start_points=start.T,linewidth=0.7,density=30,arrowstyle='-')
qv=streamQuiver(ax[0],sp,n=[5,5,5,5,5,5,5],scale=40,angles='xy')

yseed=np.linspace(XLIM[0]*1.1/R,XLIM[1]*0.9/R,7)
start=np.array([yseed*0+(ZLIM[0]+ZLIM[1])/(2*R),yseed])
sp=ax[1].streamplot(zs/R,xs/R,uzn,urn,color='k',start_points=start.T,linewidth=0.7,density=30,arrowstyle='-')
qv=streamQuiver(ax[1],sp,n=[5,5,5,5,5,5,5],scale=40,angles='xy')
ax[0].set_xlabel('z/R [-]')
ax[1].set_xlabel('z/R [-]')
ax[0].set_ylabel('r/R [-]')

# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(im1, cax=cbar_ax)

ax[0].set_xlim(ZLIM/R)
ax[0].set_ylim(XLIM/R)
ax[1].set_xlim(ZLIM/R)
ax[1].set_ylim(XLIM/R)

plt.show()

