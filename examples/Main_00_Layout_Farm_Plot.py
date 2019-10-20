import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from floris.VC import options_dict
from pybra.clean_exceptions import *
from pybra.colors import *
from helper_functions import *
from vortexcylinder.WindFarm import *

# --- Plot options
Compute=False
cmap=darkrainbow
cmap='coolwarm'
nStreamlines=0
U0 =8
# levelsLines=np.linspace(minSpeed,maxSpeed,14)
levelsLines=np.array([0.999,0.998,0.997,0.996,0.995,0.99])*U0
levelsLines=np.sort(np.unique(levelsLines))

# levelsLines=levelsLines/U0
# minSpeed=0.5
# maxSpeed=1.001
minSpeed=0.5
maxSpeed=1.03
levelsLines=np.sort([1.05,1.0,0.999,0.998,0.99,0.98,0.95,0.9,0.5])
print(levelsLines)



# NOTE: KEEP ME, debug col map
# x=np.linspace(minSpeed,maxSpeed,100)
# y=np.array([minSpeed,maxSpeed])
# [u,v]=np.meshgrid(x,y)
# v=v*0


# ---
D=126
x0=634392.0
y0=6264751.0
# x0=0
# y0=0

#     savePlane(x,y,u,v,'_data/Farm_1000new_')
x,y,u,v=loadPlane('_data/Farm_1000new_')
# x,y,u,v=loadPlane('_data_1000b/Farm_1000')
# x,y,u,v=loadPlane('_data_3000/Farm_')
# x,y,u,v=loadPlane('_data/Farm_')

# Plot and show
cmap,valuesOri=get_cmap(minSpeed,maxSpeed)
fig, ax = plt.subplots(figsize=(12.0, 12.0))
fig.subplots_adjust(left=0.000, right=1.0, top=1.0, bottom=0.000,hspace=0.19)
im = plotPlane((x-x0)/1000,(y-y0)/1000,u,v,ax,minSpeed=minSpeed,maxSpeed=maxSpeed,
        nStreamlines=nStreamlines,levelsLines=levelsLines, cmap=cmap, axial=False, colors='k')
ax.axis('off')
ax.set_xlim([-6,18])
ax.set_ylim([-1,23])

# fig.subplots_adjust(right=0.8)
# # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# divider = make_axes_locatable(ax)
# cbar_ax = divider.append_axes("right", size="2%", pad=0.10)
# cbar = manual_colorbar(fig,cmap,cax=cbar_ax, norm=mcolors.Normalize(vmin=minSpeed, vmax=maxSpeed))
# # cbar = fig.colorbar(cax, ticks=[-1, 0, 1], orientation='horizontal')
# cbar.set_ticks(valuesOri)
# cbar.set_ticklabels([str(v) for v in valuesOri])
# cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
# fig.colorbar(im, cax=cbar_ax)
print('Saving..')
fig.savefig('LayoutFarm1000c.png',bbox_to_inches='tight',dpi=800)

# plt.show()
# 

