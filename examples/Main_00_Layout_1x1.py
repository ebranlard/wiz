import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from floris.VC import options_dict
from pybra.clean_exceptions import *
from pybra.colors import darkrainbow
from pybra.colors import manual_colorbar
from helper_functions import *

# --- Plot options
bCompute=True
cmap=darkrainbow
cmap='coolwarm'
nStreamlines=0

U0 =8
minSpeed=2/U0
maxSpeed=10/U0
# levelsLines=np.linspace(minSpeed,maxSpeed,14)
levelsLines=np.sort([1.01,1.0,0.99,0.98,0.95,0.9,0.8,0.7,0.6,0.5])

# ---
nx=200
ny=nx

input_file="Layout1x1.json"
resolution=Vec3(nx, ny, 2)
D=126
bounds=[-2.5*D,D/2,0,2*D,89,90] # xmin xmax .. zmin zmax
VC_Opts=options_dict()
VC_Opts['no_induction']=False
VC_Opts['blend']=True

param='no_induction'
Vals= [True,False]

# param='Rfact'
# Vals = [1,1.2]

# --- Parametric computatoin
if bCompute:
    planes=[]
    for i,v in enumerate(Vals):
        VC_Opts[param]=v
        planes.append(get_HH_plane_vel(input_file, VC_Opts, resolution, bounds_to_set=bounds))
        savePlane(planes[-1],'_data/Layout11_{}_'.format(i),U0=U0)
else:
    planes=[]
    for i,v in enumerate(Vals):
        x,y,u,v=savePlane(x,y,u,v,'_data/Layout11_{}_'.format(i))
        planes.append((x,y,u,v))


# Plot and show
fig, axes = plt.subplots(nrows=len(Vals), ncols=1, sharex=True, sharey=True)
for ax,p in zip(axes.flat,planes):
    x=p[0]
    y=p[1]
    u=p[2]/U0
    v=p[3]/U0
    im = plotPlane(x/(D/2),y/(D/2),u,v,ax,minSpeed=minSpeed,maxSpeed=maxSpeed,
            nStreamlines=nStreamlines,levelsLines=levelsLines, cmap=cmap, axial=True, colors='k')
#     im= licPlot(x/D,y/D,u,v,ax,nLICKernel=nLICKernel,texture=None,kernel=None,minSpeed=minSpeed,maxSpeed=maxSpeed,accentuation=accentuation,offset=offset,spread=spread,axial=False,nStreamlines=11,cmap=cmap)


    savePlane(x,y,u,v,'_data/Farm_')
else:
    x,y,u,v=loadPlane('_data/Farm_')
ax.set_xlim([-5,1])
# ax.set_xlim([-2,4])
ax.set_ylim([0,2])

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# divider = make_axes_locatable(axes[-1])
# cbar_ax = divider.append_axes("right", size="5%", pad=0.20)
# manual_colorbar(fig,cmap,cax=cbar_ax, norm=mcolors.Normalize(vmin=minSpeed, vmax=maxSpeed))
fig.colorbar(im, cax=cbar_ax)

plt.show()

# 
