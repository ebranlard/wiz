import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from floris.induction import options_dict
from pybra.clean_exceptions import *
from pybra.colors import darkrainbow
from pybra.colors import manual_colorbar
from helper_functions import *
from pybra.tictoc import Timer

fontsize = 12
plt.rc('font', family='serif')
plt.rc('font', size=12)

# --- Plot options
bExport=False
bCompute=True
bColorBar=False
nStreamlines=0
U0 =6
minSpeed=0.5
maxSpeed=1.03
levelsLines=np.sort([1.05,1.0,0.99,0.98,0.95,0.9,0.5])

# ---
ny=100 # 200
nx=ny*4

input_file="Layout_1x2.json"
D=126
resolution=Vec3(nx, ny, 2)
bounds=[-4*D,14*D,0-10,2*D+10,89,90] # xmin xmax .. zmin zmax
Ind_Opts=options_dict()
Ind_Opts['Rfact']=1.0
Ind_Opts['GammaFact']=1.0
Ind_Opts['Ground']=True


titles=[]
titles.append('FLORIS (original)')
titles.append('FLORIS (with induction')
# titles.append('With induction and blending')

# --- Parametric computatoin
if bCompute:
    with Timer('time'):
        planes=[]

        print('-----------------------------------------------------------------')
        Ind_Opts['no_induction']=True
        planes.append(get_HH_plane_vel(input_file, Ind_Opts, resolution, bounds_to_set=bounds))
        savePlane(planes[-1],'_data/Layout12_0_',U0=U0)

        print('-----------------------------------------------------------------')
        Ind_Opts['no_induction']=False
        Ind_Opts['blend']=False
        planes.append(get_HH_plane_vel(input_file, Ind_Opts, resolution, bounds_to_set=bounds))
        savePlane(planes[-1],'_data/Layout12_1_',U0=U0)


#     Ind_Opts['no_induction']=False
#     Ind_Opts['blend']=True
#     planes.append(get_HH_plane_vel(input_file, Ind_Opts, resolution, bounds_to_set=bounds))
#     savePlane(planes[-1],'_data/Layout12_2_'.format(i),U0=U0)

planes=[]
planes.append(loadPlane('_data/Layout12_0_'))
planes.append(loadPlane('_data/Layout12_1_'))
# planes.append(loadPlane('_data/Layout12_2_'))


# Plot and show
cmap,valuesOri=get_cmap(minSpeed,maxSpeed)
fig, axes = plt.subplots(nrows=len(titles), ncols=1, sharex=True, sharey=True, figsize=(6.0*3.6, 6.0))
for i,(ax,p,t) in enumerate(zip(axes.flat,planes,titles)):
    x=p[0]
    y=p[1]
    u=p[2]
    v=p[3]
    im = plotPlane(x/D,y/D,u,v,ax,minSpeed=minSpeed,maxSpeed=maxSpeed,
            nStreamlines=nStreamlines,levelsLines=levelsLines, cmap=cmap, axial=True, colors='k')
    ax.title.set_text(t)

    ax.set_ylabel('r/D [-]')
    ax.title.set_text(t)
    ax.tick_params(direction='in')

    if i==1:
        ax.set_xlabel('z/D [-]')

ax.set_xlim([-4,14])
ax.set_ylim([0,2])

if bColorBar:
    fig.subplots_adjust(left=0.08, right=0.83, top=0.93, bottom=0.11,hspace=0.17)
    cbar_ax = fig.add_axes([0.88, 0.11, 0.04, 0.82])
    cbar=fig.colorbar(im, cax=cbar_ax)
    cbar.set_ticks(levelsLines)
    cbar.set_ticklabels([str(v) if v not in [0.99] else '' for v in levelsLines])
    cbar.ax.tick_params(axis='both', direction='in',length=18,color=(0.5,0.5,0.5))
else:
    fig.subplots_adjust(left=0.035, right=0.990, top=0.96, bottom=0.08,hspace=0.17)

if bExport:
    fig.savefig('Layout12.pdf',bbox_to_inches='tight',dpi=300)
        # fig.savefig('Layout12.png',bbox_to_inches='tight',dpi=600)
else:
    plt.show()

# 
