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
U0 =8
cmap=darkrainbow
cmap='coolwarm'
nStreamlines=0
minSpeed=2
maxSpeed=10
levelsLines=np.linspace(minSpeed,maxSpeed,14)

# ---
nx=200
ny=100

input_file="Layout1x2.json"
resolution=Vec3(nx, ny, 3)
VC_Opts=options_dict()


titles=[]
titles.append('Without induction (original)')
titles.append('With induction')
titles.append('With induction and blending')

# --- Parametric computatoin
if bCompute:
    planes=[]

    VC_Opts['no_induction']=True
    planes.append(get_HH_plane_vel(input_file, VC_Opts, resolution))
    savePlane(planes[-1],'_data/Layout12_0_'.format(i),U0=U0)

    VC_Opts['no_induction']=False
    VC_Opts['blend']=False
    planes.append(get_HH_plane_vel(input_file, VC_Opts, resolution))
    savePlane(planes[-1],'_data/Layout12_1_'.format(i),U0=U0)

    VC_Opts['no_induction']=False
    VC_Opts['blend']=True
    planes.append(get_HH_plane_vel(input_file, VC_Opts, resolution))
    savePlane(planes[-1],'_data/Layout12_2_'.format(i),U0=U0)
else:
    planes=[]
    planes.append(loadPlane('_data/Layout12_0_'))
    planes.append(loadPlane('_data/Layout12_1_'))
    planes.append(loadPlane('_data/Layout12_2_'))


# Plot and show
fig, axes = plt.subplots(nrows=len(titles), ncols=1, sharex=True, sharey=True)
for ax,p,t in zip(axes.flat,planes,titles):
    x=p[0]
    y=p[1]
    u=p[2]
    v=p[3]
    im = plotPlane(x/D,y/D,u,v,ax,minSpeed=minSpeed,maxSpeed=maxSpeed,
            nStreamlines=nStreamlines,levelsLines=levelsLines, cmap=cmap, axial=True, colors='k')
    ax.title.set_text(t)

ax.set_xlim([4.5,8.5])
ax.set_ylim([-1.5,1.5])

fig.subplots_adjust(right=0.8)
cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# divider = make_axes_locatable(axes[-1])
# cbar_ax = divider.append_axes("right", size="5%", pad=0.20)
# manual_colorbar(fig,cmap,cax=cbar_ax, norm=mcolors.Normalize(vmin=minSpeed, vmax=maxSpeed))
fig.colorbar(im, cax=cbar_ax)

plt.show()

# 
