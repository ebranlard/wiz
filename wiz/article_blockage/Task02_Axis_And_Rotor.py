""" 
Figure 4 of 
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
    HH     = 9999  ; Horizontal = True
    Lambda = np.inf
    Yaw    = 0
    U0     = 1
    R      = 1

    # --- Plot parameters
    component='Axial'

    # --- Models and color map definitions
    _,_,CMAP = getModelsCMAP() # see helper_functions

    # colAD1=CMAP[3]
    # colAD2=CMAP[1]
    colAD1=fColrs(2)
    colAD2=fColrs(1)
    styAD1='o'
    styAD2='d'


    fig1, ax1 = plt.subplots(figsize=(6.4,4.4))
    fig1.subplots_adjust(left=0.11, right=0.975, top=0.97, bottom=0.13, hspace=0.20, wspace=0.20)
    fig2, ax2 = plt.subplots(figsize=(6.4,4.4))
    fig2.subplots_adjust(left=0.11, right=0.975, top=0.97, bottom=0.13, hspace=0.20, wspace=0.20)

    # fig1, ax1= plt.subplots(figsize=(6.4,3.60))
    # fig1.subplots_adjust(left=0.10, right=0.88, top=0.98, bottom=0.130,hspace=0.0, wspace=0)
    # fig2, ax2= plt.subplots(figsize=(6.4,3.60))
    # fig2.subplots_adjust(left=0.10, right=0.88, top=0.98, bottom=0.130,hspace=0.0, wspace=0)

    CTs=[0.4,0.95]
    for iCT, CT in enumerate(CTs):

        # --- Setting up models
        VC  = setupModel(CT = CT, HH = HH, Lambda = Lambda, Horizontal = Horizontal, Yaw=Yaw, Model='VC')

        # --- Axial survey
        vz=np.linspace(-5,3,45)
        vy=0
        ADU,VCV,VCW = VC.fAxial(vz)
        ax=ax1
        ax.plot([0,0],[-5,3],'k--',lw=0.5)
        if CT==0.4:
            ax.plot(vz, VCW.ravel(),'-'    , color=colAD1, label=r'VC ($C_T=0.4$)')
        else:
            ax.plot(vz, VCW.ravel(),'-'    , color=colAD2, label=r'VC ($C_T=0.95$)')

        ax.grid(linewidth='0.5',color=[0.8,0.8,0.8], ls='--')
        ax.set_ylabel(r'$U_x/U_0$ [-]')
        ax.set_xlabel(r'x/R [-]')
        ax.legend(loc='center left',fontsize=14)
        ax.set_xlim([-5 , 3])
        ax.set_ylim([0.3, 1])
        ax.tick_params(direction='in')
        ax.title.set_text('TwoCtsAxis')

        # --- Radial survey
        vr=np.linspace(-2,2,100);
        VCU,VCV,VCW = VC.fRadial(vr)
        ax=ax2
        if CT==0.4:
            ax.plot(vr, VCW.ravel(),'-'    , color=colAD1, label=r'VC ($C_T=0.4$)')
            ax.plot(vr, VCU.ravel(),'-'    , color=colAD1)
        else:
            ax.plot(vr, VCW.ravel(),'-'    , color=colAD2, label=r'VC ($C_T=0.95$)')
            ax.plot(vr, VCU.ravel(),'-'    , color=colAD2)

        ax.text(0.51,0.71,r'$U_x$')
        ax.text(0.51,0.16,r'$U_r$')
        ax.grid(linewidth='0.5',color=[0.8,0.8,0.8], ls='--')
        ax.set_ylabel(r'$U/U_0$ [-]')
        ax.set_xlabel(r'r/R [-]')
        ax.legend(loc='center right',fontsize=13)
        ax.set_xlim([0 , np.max(vr)])
        ax.set_ylim([0, 1.1])
        ax.set_xticks([0,0.5,1,1.5,2])
        ax.tick_params(direction='in')
        ax.title.set_text('TwoCtsRotor')

if __name__ == '__main__':
    main()
    plt.show()
