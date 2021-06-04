#--- Legacy python 2.7
from __future__ import division
from __future__ import print_function
# --- General
import unittest
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
# --- Local
from wiz.WindTurbine import WindTurbine
from wiz.Solver import Ct_const_cutoff
from wiz.plotting_functions import get_contours, get_cmap

def main():
    Ground    = False
    no_wake   = False # Used when coupled to FLORIS
    only_ind  = False  # 
    R         = 65
    h_hub     = 0.0*R
    r_bar_cut = 0.11
    r_bar_tip = 0.90
    CT0       = 0.95
    Lambda    = np.inf
    U0        = 10
    R         = 65
    nCyl      = 1
    root      = False
    longi     = False
    tang      = True
    wd        =-30*np.pi/180     # Wind direction
    ye        = -30*np.pi/180

    # --- Plotting Options
    rcParams['contour.negative_linestyle'] = 'solid'
    minSpeed=0.5
    maxSpeed=1.05
    levelsContour, levelsLines= get_contours(CT0)
    cmap,_=get_cmap(minSpeed,maxSpeed,alpha=0.6)


    WT=WindTurbine(R=R,r_hub=[0,h_hub,0],e_shaft_yaw0=[0,0,1],e_vert=[0,1,0],Ground=Ground)
    vr_bar    = np.linspace(0,1.0,100)
    Ct_AD     = Ct_const_cutoff(CT0,r_bar_cut,vr_bar,r_bar_tip) # TODO change me
    WT.update_loading(r=vr_bar*R, Ct=Ct_AD, Lambda=Lambda, nCyl=nCyl)
    WT.update_yaw_pos( wd-ye)
    WT.update_wind([U0*np.sin(wd),0,U0*np.cos(wd)])

    # --- Flow field and speed
    nx=120
    nz=100
    x = np.linspace(-2*R,2*R,nx)
    z = np.linspace(-5*R,3*R,nz)
    [X,Z]=np.meshgrid(x,z)
    Y=Z*0+h_hub
    ux,uy,uz = WT.compute_u(X,Y,Z,root=root,longi=longi,tang=tang,no_wake=no_wake,only_ind=only_ind)

    print('gamma_t {:.4f}  In article: {:.4f}'.format(WT.gamma_t[0],-0.60414*U0                   ))
    print('chi     {:.4f}  In article: {:.4f}'.format(WT.chi       ,ye*(1+0.3*(1-np.sqrt(1-CT0))) ))

    U0_norm = U0*np.cos(ye);      #<<<<<<<<<<<<<<<< IMPORTANT for normalization of axial velocity field
    Speed=np.sqrt(uz**2)/U0_norm  #<<<<<<<<<<<<<<<<<<<<<<<< IMPORTANT, this is normal to rotor
    Speed = np.ma.masked_where(np.isnan(Speed), Speed)


    # --- Plot
    fig=plt.figure()
    ax=fig.add_subplot(111)
    im=ax.contourf(Z/R,X/R,Speed,levels=levelsContour, cmap=cmap, vmin=minSpeed, vmax=maxSpeed)
    cs=ax.contour(Z/R, X/R, Speed, levels=levelsLines, colors='k', linewidths=0.8, alpha=1.0, linestyles='solid')
    Rotor=WT.rotor_disk_points()
    ax.plot(Rotor[2,:]/R,Rotor[0,:]/R,'k--')
    cb=fig.colorbar(im)
    ax.set_xlabel('z/R [-]')
    ax.set_ylabel('x/R [-]')
    ax.set_aspect('equal')
    deg=180/np.pi
    ax.set_title('yaw_pos = {:.1f} - yaw_wind={:.1f} - chi={:.1f} - yaw_err={:.1f}'.format(WT.yaw_pos*deg,WT.yaw_wind*deg,WT.chi*deg,WT.yaw_error*deg))
    plt.show()

if __name__ == "__main__":
    main()
#     unittest.main()
