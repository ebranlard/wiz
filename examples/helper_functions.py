import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


# --- Local
import floris.tools as wfct
from floris.utilities import Vec3

try:
    from pybra.colors import *
    from pybra.lic import lic, licImage
    from pybra.curves import streamQuiver
    from pybra.clean_exceptions import *
except:
    raise Exception('This script requires the package `pybra` from https://github.com/ebranlard/pybra/')

def get_cmap(minSpeed,maxSpeed):
    DS=0.001
    # MathematicaDarkRainbow=[(60 /255,86 /255,146/255), (64 /255,87 /255,142/255), (67 /255,107/255,98 /255), (74 /255,121/255,69 /255), (106/255,141/255,61 /255), (159/255,171/255,67 /255), (207/255,195/255,77 /255), (223/255,186/255,83 /255), (206/255,128/255,76 /255), (186/255,61 /255,58 /255)]

    #     ManuDarkOrange  = np.array([198 ,106,1   ])/255.;
    #     ManuLightOrange = np.array([255.,212,96  ])/255.;
    # (1,212/255,96/255),  # Light Orange
    # (159/255,159/255,204/255), # Light Blue
    #     MathematicaLightGreen = np.array([158,204,170 ])/255.;
    # (159/255,159/255,204/255), # Light Blue
    seq=[
    (63/255 ,63/255 ,153/255), # Dark Blue
    (159/255,159/255,204/255), # Light Blue
    (158/255,204/255,170/255), # Light Green
    (1,212/255,96/255),  # Light Orange
    (1,1,1),  # White
    (1,1,1),  # White
    (1,1,1),  # White
    (138/255 ,42/255 ,93/255), # DarkRed
    ]
    valuesOri=np.array([
    minSpeed,  # Dark Blue
    0.90,
    0.95,
    0.98,
    1.00-DS , # White
    1.00    , # White
    1.00+DS , # White
    maxSpeed         # DarkRed
    ])
    values=(valuesOri-min(valuesOri))/(max(valuesOri)-min(valuesOri))

    valuesOri=np.around(valuesOri[np.where(np.diff(valuesOri)>DS)[0]],2)

    cmap= make_colormap(seq,values=values)
    return cmap,np.concatenate((valuesOri,[maxSpeed]))





def get_HH_plane_vel(input_file, Ind_Opts, resolution=Vec3(232, 114, 3),bounds_to_set=None):
    fi = wfct.floris_utilities.FlorisInterface(input_file)
    # Calculate wake
    fi.calculate_wake(Ind_Opts=Ind_Opts)

    #fd=fi.get_flow_data(resolution = resolution, Ind_Opts=Ind_Opts)
    
    # # Initialize the horizontal cut
    hor_plane = wfct.cut_plane.HorPlane(
        fi.get_flow_data(resolution = resolution, Ind_Opts=Ind_Opts, bounds_to_set=bounds_to_set),
        fi.floris.farm.turbines[0].hub_height
    )
    u_mesh = hor_plane.u_mesh.reshape(hor_plane.resolution[1], hor_plane.resolution[0])
    v_mesh = hor_plane.v_mesh.reshape(hor_plane.resolution[1], hor_plane.resolution[0])
    return hor_plane.x1_lin,hor_plane.x2_lin, u_mesh, v_mesh, fi


def savePlane(x,y=None,u=None,v=None,base='',U0=None):
    if u is None:
        p=x
        base=y
        x=p[0]
        y=p[1]
        u=p[2]
        v=p[3]
    if U0 is not None:
        u=u/U0
        v=v/U0

    base = os.path.normpath(base)
    np.save(base+'x.npy',x)
    np.save(base+'y.npy',y)
    np.save(base+'u.npy',u)
    np.save(base+'v.npy',v)


def loadPlane(base=''):
    base = os.path.normpath(base)
    x = np.load(base+'x.npy')
    y = np.load(base+'y.npy')
    u = np.load(base+'u.npy')
    v = np.load(base+'v.npy')
    return x,y,u,v


def plotPlane(x,y,u,v,ax,minSpeed=None,maxSpeed=None, cmap='coolwarm', colors='w', linewidths=0.8, alpha=0.3,nStreamlines=0,levelsContour=None,levelsLines=None,axial=False):
    if axial:
        Speed=u
    else:
        Speed=np.sqrt(u**2+v**2)
    if minSpeed is None:
        minSpeed = Speed.min()
        maxSpeed = Speed.max()

    if not ax:
        fig, ax = plt.subplots()
    Z = np.ma.masked_where(np.isnan(Speed), Speed)

    # Plot the cut-through
    im = ax.pcolormesh(x, y, Z, cmap=cmap, vmin=minSpeed, vmax=maxSpeed)
    #im = ax.contourf  (x, y, Z, levels=levelsContour, cmap=cmap, vmin=minSpeed, vmax=maxSpeed)

    if levelsLines is not None:
        rcParams['contour.negative_linestyle'] = 'solid'
        cs=ax.contour(x, y, Z, levels=levelsLines, colors=colors, linewidths=linewidths, alpha=alpha)
#         ax.clabel(cs,list(levelsLines))

    if nStreamlines>0:
        yseed=np.linspace(min(y)*0.9,max(y)*0.9,nStreamlines)
        start=np.array([yseed*0,yseed])
        sp=ax.streamplot(x,y,u,v,color='k',start_points=start.T,linewidth=0.7,density=30,arrowstyle='-')
#         sp=ax.streamplot(x,y,u,v,color='k',linewidth=0.7,density=2)
        qv=streamQuiver(ax,sp,n=5,scale=40,angles='xy')




    ax.set_aspect('equal')

    return im
