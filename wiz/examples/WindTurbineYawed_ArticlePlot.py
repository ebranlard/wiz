""" 
Velocity field for a yawed wind turbine at two different thrust coefficients CT, with a yaw angle of 30 deg

Reproduces Figure 8 of:
  [1] Branlard, Meyer Forstig, (2020), Assessing the blockage effect of wind turbines and wind farms
using an analytical vortex model, Wind Energy


TODO: use the more conventional coordinate system with x downstream instead of z downstream

"""
#--- Legacy python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# --- General
import numpy as np
import matplotlib
from matplotlib import rcParams
import matplotlib.pyplot as plt
# --- Local
from wiz.WindTurbine import WindTurbine

def main():
    # --- Wind turbine configuration
    R     = 10   # Rotor radius
    U0    = 10   # Free stream velocity [m/s]
    Yaw   = -30  # Yaw angle [deg]
    Model = 'VC' # Induction model VC, VD, SS, VCVD
    HH_R  = 99   # HubHeight in [R]
    vCT         = [0.4     ,0.95]  # Thrust coefficients
    gamma_rings = np.array([-0.21341,-0.60414])*U0 # TODO adapt using equation (15) from [1]

    # --- Plot parameters
    LW=0.8
    lineColorVC='k'
    lineAlpha=1.0
    minSpeed=0.5
    maxSpeed=1.05
    cmap = matplotlib.cm.get_cmap('viridis')
    nx=120
    nz=100
    ZMIN =-5 #NOTE: using vortex cylinder coordinates with z downstream for now..
    ZMAX = 3
    XMIN =-2
    XMAX = 2
    vx=np.linspace(XMIN*R,XMAX*R,nx)
    vz=np.linspace(ZMIN*R,ZMAX*R,nz)

    for iCT,(CT,gamma_t) in enumerate(zip(vCT,gamma_rings)):

        # --- Define WT
        WT=WindTurbine(R=R,e_shaft_yaw0=[0,0,1],e_vert=[0,1,0],r_hub=[0,HH_R*R,0],Ground=HH_R<9, Model=Model)
        WT.update_wind([U0*np.sin(Yaw*np.pi/180),0,U0*np.cos(Yaw*np.pi/180)])
        WT.set_chi(Yaw*(1+0.3*(1-np.sqrt(1-CT)))*np.pi/180)
        WT.gamma_t=np.array([gamma_t]) # alternative, use update_loading

        # --- Compute velocity, normalized by free stream normal to disk
        U0_n=U0*np.cos(Yaw*np.pi/180);
        X,Z = np.meshgrid(vx,vz) 
        Y=X*0+HH_R*R
        U,V,W=WT.compute_u(X,Y,Z,longi=False,root=False)
        U=U/U0_n
        W=W/U0_n

        # --- Plot
        if CT==0.95:
            levelsLines=np.sort(np.array([0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1.,1.01]))
        else:
            levelsLines=np.sort(np.array([0.7,0.8,0.85,0.9,0.95,0.98,0.99,0.995,1.01]))
        levelsContour = np.sort(np.concatenate((levelsLines,[1.4,0.2])))
        fig, ax = plt.subplots(figsize=(6.4,3.10))
        fig.subplots_adjust(left=0.10, right=0.88, top=0.98, bottom=0.150,hspace=0.0, wspace=0)
        # Contours
        im = ax.contourf  (Z/R, X/R, W, levels=levelsContour, cmap=cmap, vmin=minSpeed, vmax=maxSpeed)
        rcParams['contour.negative_linestyle'] = 'solid'
        cs=ax.contour(Z/R, X/R, W, levels=levelsLines, colors=lineColorVC, linewidths=0.8, alpha=lineAlpha, linestyles='-')
        ax.tick_params(direction='in')
        # Rotor and wake
        m=np.tan(WT.chi)
        zz=np.linspace(0,max(vz),100); xx=m*zz;
        ax.plot(zz,xx+1,'k-',lw=3);
        ax.plot(zz,xx-1,'k-',lw=3);
        zz=np.linspace(min(vz),max(vz),100); xx=m*zz;
        ax.plot(zz,xx+0,'k-.',lw=1);
        #
        if iCT==1:
            cbar_ax = fig.add_axes([0.895, 0.15, 0.02, 0.832])
            cbar=fig.colorbar(im, cax=cbar_ax)
        ax.set_ylim([XMIN, XMAX])
        ax.set_xlim([ZMIN, ZMAX])
        ax.set_xlabel('x/R [-]')
        ax.set_ylabel('y/R [-]')
        if CT==0.95:
            ax.text(-4.15,0.15,'0.99',rotation = 70,va = 'center' ,ha = 'center',fontsize = 11)
            ax.text(-2.9 ,0.15,'0.98',rotation = 70,va = 'center' ,ha = 'center',fontsize = 11)
            ax.text(-1.75,0.15,'0.95',rotation = 70,va = 'center' ,ha = 'center',fontsize = 11)
            ax.text(-1.1 ,0.15,'0.9' ,rotation = 70,va = 'center' ,ha = 'center',fontsize = 11)
            ax.text(-0.53,0.15,'0.8' ,rotation = 70,va = 'center' ,ha = 'center',fontsize = 11)
            ax.text(-0.21,0.15,'0.7' ,rotation = 70,va = 'center' ,ha = 'center',fontsize = 11)
            ax.text(0.11 ,0.15,'0.6' ,rotation = 70,va  = 'center',ha = 'center',fontsize = 11)
            ax.text(0.41 ,0.15,'0.5' ,rotation = 70,va  = 'center',ha = 'center',fontsize = 11)
            ax.text(1.72 ,0.70,'1.01',rotation = 30,va = 'center' ,ha = 'center',fontsize = 11)
        else:
            ax.text(-3.44,0.2 ,'0.995',rotation = 70 ,va = 'center',ha = 'center',fontsize = 11)
            ax.text(-2.38,0.2 ,'0.99' ,rotation = 70 ,va = 'center',ha = 'center',fontsize = 11)
            ax.text(-1.60,0.2 ,'0.98' ,rotation = 70 ,va = 'center',ha = 'center',fontsize = 11)
            ax.text(-0.78,0.2 ,'0.95' ,rotation = 70 ,va = 'center',ha = 'center',fontsize = 11)
            ax.text(-0.22,0.2 ,'0.9'  ,rotation = 70 ,va = 'center',ha = 'center',fontsize = 11)
            ax.text(0.16 ,0.2 ,'0.8'  ,rotation = 70 ,va = 'center',ha = 'center',fontsize = 11)
            ax.text(0.70 ,0.1 ,'0.85' ,rotation = 70 ,va = 'center',ha = 'center',fontsize = 11)
            ax.text(1.05 ,0.76,'1.01' ,rotation = 45 ,va = 'center',ha = 'center',fontsize = 11)
            ax.text(2.5  ,0.30,'0.995',rotation = -10,va = 'center',ha = 'center',fontsize = 11)

if __name__ == "__main__":
    main()
    plt.show()
