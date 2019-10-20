import numpy as np
from floris.VC import options_dict
from pybra.clean_exceptions import *
from helper_functions import *

# --- Plot options
Compute=True

U0=8
nx=1000
ny=1000

input_file="anholt_v0.json"
resolution=Vec3(nx, ny, 2)

x0=634392.0
y0=6264751.0
bounds=[x0-6000, x0+18000 , y0-1000 ,y0+23000 ,89,90] # xmin xmax .. zmin zmax
VC_Opts=options_dict()

# --- Parametric computatoin
if Compute:
    VC_Opts['no_induction'] = False
    VC_Opts['blend']        = False
    x,y,u,v,_=get_HH_plane_vel(input_file, VC_Opts, resolution, bounds_to_set=bounds)
    x=x
    y=y
    u=u/U0
    v=v/U0
    savePlane(x,y,u,v,'_data/Farm_1000new_')

