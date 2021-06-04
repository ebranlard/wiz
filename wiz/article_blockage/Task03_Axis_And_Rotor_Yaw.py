""" 
Figure 7 of 
  Branlard, Meyer Forstig, (2020), Assessing the blockage effect of wind turbines and wind farms
using an analytical vortex model, Wind Energy
"""

#--- Legacy python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# --- General
import matplotlib.pyplot as plt
import numpy as np
import os
# --- Local
from wiz.VortexCylinder import vc_tang_u
from helper_functions import *

def main():
    HH         = 9999 ; Horizontal = True
    # HH         = 1.5; Horizontal = False
    Lambda     = np.inf
    Yaw        = -30
    U0=1
    R=1

    nDATAx=64;
    nDATAz=50;
    R=1;
    ZMIN=-5; ZMAX=3;
    XMIN=-2; XMAX=2;

    # --- Models and color map definitions
    _,_,CMAP = getModelsCMAP() # see helper_functions

    colAD1=fColrs(2)
    colAD2=fColrs(1)

    fig1, ax1 = plt.subplots(figsize=(6.4,4.4))
    fig1.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.13, hspace=0.20, wspace=0.20)
    fig2, ax2 = plt.subplots(figsize=(6.4,4.4))
    fig2.subplots_adjust(left=0.11, right=0.98, top=0.97, bottom=0.13, hspace=0.20, wspace=0.20)


    epsilon=10^-1;

    CTs=[0.4,0.95]
    for iCT, CT in enumerate(CTs):

        U0  = np.cos(Yaw*np.pi/180)
        chi = Yaw*(1+0.3*(1-np.sqrt(1-CT)))
        m   = np.tan(chi*np.pi/180)        

        # --- Setting up models
        VC  = setupModel(CT = CT, HH = HH, Lambda = Lambda, Horizontal = Horizontal, Yaw=Yaw, Model='VC')

        vz_axis  = np.linspace(ZMIN,ZMAX,nDATAz)*R              ;
        vx_axis  = vz_axis*m                                 ;
        vs_axis  = np.sign(vz_axis)*np.sqrt(vz_axis**2+vx_axis**2);

        # --- Axial survey
        _,_,VCW_axis  = VC.fU(vz_axis*m, 0*vz_axis, vz_axis   )/U0;

        gamma_rings=VC.WT.gamma_t
        w_axis_th=gamma_rings/2*(1+vs_axis/(np.sqrt(R**2+vs_axis**2)))

        ax=ax1
        ax.plot([0,0],[-5,3],'k--',lw=0.5)
        if CT==0.4:
            ax.plot(vs_axis/R, VCW_axis,'-'      , color=colAD1,       label=r'VC ($C_T=0.4$)')
            ax.plot(vs_axis/R,1+w_axis_th/U0 ,'.', color=colAD1, ms=5, label=r'VC ($C_T=0.4$, straight)')
        else:
            ax.plot(vs_axis/R, VCW_axis,'-'      , color=colAD2,      label=r'VC ($C_T=0.95$)')
            ax.plot(vs_axis/R,1+w_axis_th/U0 ,'.', color=colAD2, ms=5,label=r'VC ($C_T=0.95$, straight)')

        ax.grid(linewidth='0.5',color=[0.8,0.8,0.8], ls='--')
        ax.set_ylabel(r'$U_x/U_0$ [-]')
        ax.set_xlabel(r'$\xi/R$ [-]')
        ax.legend(loc='center left',fontsize=14)
        ax.set_xlim([-5 , 3])
        ax.set_ylim([0.3, 1])
        ax.set_xticks([-4,-2,0,2])
        ax.set_yticks([0.4,0.6,0.8,1])
        ax.tick_params(direction='in')
        ax.title.set_text('TwoCtsAxisYaw')

        # --- Radial survey
        vx_rotor = np.linspace(XMIN,XMAX,nDATAx)*R              ;
        _,_,VCW_rotor = VC.fU(vx_rotor, 0*vx_rotor, vx_rotor*0)/U0;

        ax=ax2
        if CT==0.4:
            ax.plot(vx_rotor/R, VCW_rotor,'-'    , color=colAD1, label=r'VC ($C_T=0.4$)')
        else:
            ax.plot(vx_rotor/R, VCW_rotor,'-'    , color=colAD2, label=r'VC ($C_T=0.95$)')

        ax.grid(linewidth='0.5',color=[0.8,0.8,0.8], ls='--')
        ax.set_ylabel(r'$U_x/U_0$ [-]')
        ax.set_xlabel(r'y/R [-]')
        ax.legend(loc='lower right',fontsize=14)
        ax.set_xlim([-2 , 2])
        ax.set_ylim([0.2, 1.2])
        ax.set_xticks([-2,-1,0,1,2])
        ax.tick_params(direction='in')
        ax.title.set_text('TwoCtsRotorYaw')

if __name__ == '__main__':
    main()
    plt.show()
