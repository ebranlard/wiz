import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from floris.VC import options_dict
from pybra.clean_exceptions import *
from pybra.colors import *
from helper_functions import *


fontsize = 12
plt.rc('font', family='serif')
plt.rc('font', size=12)

# --- Plot options
bCompute=True

bColorBar=False
nStreamlines=0

U0 =10
minSpeed=0.5
maxSpeed=1.03
# levelsLines=np.linspace(minSpeed,maxSpeed,14)
levelsLines=np.sort([1.05,1.0,0.99,0.98,0.95,0.9,0.5])

# ---
nx=1000
ny=200

input_file="Layout1x1.json"
resolution=Vec3(nx, ny, 2)
D=126
bounds=[-0.1*D,120*D,-2.1*D,2.1*D+10,89,90]
VC_Opts=options_dict()
VC_Opts['no_induction']=True
param='no_induction'
Vals= [True,False]

# --- Parametric computatoin
if bCompute:
    planes=[]
    planes.append(get_HH_plane_vel(input_file, VC_Opts, resolution, bounds_to_set=bounds))
    savePlane(planes[-1],'_data/Layout11_Decay_',U0=U0)

planes=[]
planes.append(loadPlane('_data/Layout11_Decay_'))


# Plot and show
# --------------------------------------------------------------------------------}
# ---  
# --------------------------------------------------------------------------------{
cmap,valuesOri=get_cmap(minSpeed,maxSpeed)
fig, ax = plt.subplots(figsize=(35.0, 2.0))
p=planes[0]
x=p[0]
y=p[1]
u=p[2]
uw=(u-1)*0.4
u=1+uw
v=p[3]
im = plotPlane(x/(D),y/(D),u,v,ax,minSpeed=minSpeed,maxSpeed=maxSpeed,
        nStreamlines=nStreamlines,levelsLines=levelsLines, cmap=cmap, axial=True, colors='k')

ax.axis('off')
# ax.set_ylabel('r/D [-]')
# ax.title.set_text(t)
# ax.tick_params(direction='in')
# ax.set_xlabel('z/D [-]')
# ax.set_xlim([-2,2])
ax.set_ylim([-2,2])

# if bColorBar:
#     fig.subplots_adjust(left=0.08, right=0.83, top=0.93, bottom=0.11,hspace=0.17)
#     cbar_ax = fig.add_axes([0.88, 0.11, 0.04, 0.82])
#     cbar=fig.colorbar(im, cax=cbar_ax)
#     cbar.set_ticks(levelsLines)
#     cbar.set_ticklabels([str(v) if v not in [0.99] else '' for v in levelsLines])
#     cbar.ax.tick_params(axis='both', direction='in',length=18,color=(0.5,0.5,0.5))
# else:
fig.subplots_adjust(left=0.001, right=0.999, top=0.99, bottom=0.01,hspace=0.19)

# fig.savefig('Layout11Decay.pdf',bbox_to_inches='tight',dpi=100)
fig.savefig('Layout11Decay.png',bbox_to_inches='tight',dpi=600)


plt.show()

# 
