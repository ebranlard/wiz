import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from floris.VC import options_dict
from pybra.clean_exceptions import *
from pybra.colors import darkrainbow
from pybra.colors import manual_colorbar
from helper_functions import *
from vortexcylinder.WindFarm import *

# --- Plot options
Compute=False
cmap=darkrainbow
cmap='coolwarm'
nStreamlines=0
U0 =8
minSpeed=U0-1
maxSpeed=U0+1
levelsLines=np.linspace(minSpeed,maxSpeed,14)
levelsLines=np.array([0.999,0.998,0.997,0.996,0.995,0.99])*U0
levelsLines=np.sort(np.unique(levelsLines))

levelsLines=levelsLines/U0
minSpeed=minSpeed/U0
maxSpeed=maxSpeed/U0
print(levelsLines)

# ---
nx=1000
ny=1000

# input_file="LayoutFarm.json"
input_file="anholt_v0.json"
resolution=Vec3(nx, ny, 2)
D=126

x0=634392.0
y0=6264751.0
bounds=[x0-8000, x0+15000 , y0-3000 ,y0+25000 ,89,90] # xmin xmax .. zmin zmax

# bounds=[-30*D-xmin, xmax+20*D , -30*D+ymin ,ymax+30*D ,89,90] # xmin xmax .. zmin zmax
# bounds=None
VC_Opts=options_dict()


# param='Rfact'
# Vals = [1,1.2]

# --- Parametric computatoin
if Compute:
    planes=[]
    titles=[]
    VC_Opts['no_induction'] = False
    VC_Opts['blend']        = False
    planes.append(get_HH_plane_vel(input_file, VC_Opts, resolution, bounds_to_set=bounds))
    p=planes[0]

    x=p[0]
    y=p[1]
    u=p[2]/U0
    v=p[3]/U0

    savePlane(x,y,u,v,'_data/Farm_')
else:
    x,y,u,v=loadPlane('_data/Farm_')

# Plot and show
fig=plt.figure()
ax = fig.add_subplot(111)
im = plotPlane((x-x0)/1000,(y-y0)/1000,u,v,ax,minSpeed=minSpeed,maxSpeed=maxSpeed,
        nStreamlines=nStreamlines,levelsLines=levelsLines, cmap=cmap, axial=True, colors='k')
# ax.set_xlim([-5,1])

fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
divider = make_axes_locatable(ax)
cbar_ax = divider.append_axes("right", size="2%", pad=0.10)
manual_colorbar(fig,cmap,cax=cbar_ax, norm=mcolors.Normalize(vmin=minSpeed, vmax=maxSpeed))
# fig.colorbar(im, cax=cbar_ax)

plt.show()
# 

