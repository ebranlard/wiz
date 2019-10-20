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
nStreamlines=0

U0 =10
minSpeed=0.5
maxSpeed=1.03
# levelsLines=np.linspace(minSpeed,maxSpeed,14)
levelsLines=np.sort([1.05,1.0,0.99,0.98,0.95,0.9,0.5])

# ---
nx=10
ny=10

input_file="Layout1x1.json"
resolution=Vec3(nx, ny, 2)
D=126
bounds=[-2*D,2*D,0-10,2*D+10,89,90] # xmin xmax .. zmin zmax
VC_Opts=options_dict()
VC_Opts['no_induction']=False
VC_Opts['blend']=False
VC_Opts['Rfact']=1.1
VC_Opts['GammaFact']=1.43
param='no_induction'
Vals= [True,False]

# --- Parametric computatoin
titles=['FLORIS (original)', 'FLORIS (with induction)']
if bCompute:
    planes=[]
    for i,v in enumerate(Vals):
        VC_Opts[param]=v
        planes.append(get_HH_plane_vel(input_file, VC_Opts, resolution, bounds_to_set=bounds))
        savePlane(planes[-1],'_data/Layout11_{}_'.format(i),U0=U0)
else:
    planes=[]
    for i,v in enumerate(Vals):
        x,y,u,v=loadPlane('_data/Layout11_{}_'.format(i))
        planes.append((x,y,u*U0,v*U0))


# Plot and show
# --------------------------------------------------------------------------------}
# ---  
# --------------------------------------------------------------------------------{

cmap,valuesOri=get_cmap(minSpeed,maxSpeed)
# figsize=(9,8)
# fig.savefig(fname, bbox_to_inches='tight',dpi=500) 
fig, axes = plt.subplots(nrows=len(Vals), ncols=1, sharex=True, sharey=True)
try:
    n=len(axes)
except:
    axes=[axes]
for i,(ax,p,t) in enumerate(zip(axes,planes,titles)):
    x=p[0]
    y=p[1]
    u=p[2]/U0
    v=p[3]/U0
    im = plotPlane(x/(D),y/(D),u,v,ax,minSpeed=minSpeed,maxSpeed=maxSpeed,
            nStreamlines=nStreamlines,levelsLines=levelsLines, cmap=cmap, axial=True, colors='k')

    ax.set_ylabel('r/D [-]')
    ax.title.set_text(t)
    ax.tick_params(direction='in')

    if i==1:
        ax.set_xlabel('z/D [-]')
ax.set_xlim([-2,2])
ax.set_ylim([0,2])

fig.subplots_adjust(left=0.08, right=0.83, top=0.93, bottom=0.11,hspace=0.17)
cbar_ax = fig.add_axes([0.88, 0.11, 0.04, 0.82])
print(valuesOri)
cbar=fig.colorbar(im, cax=cbar_ax)
cbar.set_ticks(levelsLines)
cbar.set_ticklabels([str(v) if v not in [0.99] else '' for v in levelsLines])
cbar.ax.tick_params(axis='both', direction='in',length=18,color=(0.5,0.5,0.5))

fig.savefig('ColorBar.png',bbox_to_inches='tight',dpi=1200)


# plt.show()

# 
