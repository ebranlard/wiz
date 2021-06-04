""" 
Figure 5 of 
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
    Yaw        = 0
    U0=1
    R=1

    # --- Models and color map definitions
    Models,_,_ = getModelsCMAP() # see helper_functions
    Model = Models[1]
    # --- Plot parameters
    LW=0.8
    minSpeed=0.5
    maxSpeed=1.05
    nStreamlines=0
    levelsContour=None
    levelsLines=None
    lineColorVC='k'
    lineAlpha=1.0
    levelsLines   = np.sort([1.15,1.05,1.0,0.99,0.98,0.95,0.9,0.8,0.7,0.5,0.2])
    levelsContour = np.sort(levelsLines)
    component='Axial'
    cmap,valuesOri=get_cmap(minSpeed,maxSpeed,alpha=0.6)

    CTs=[0.4]
    CTs=[0.4,0.95]

    for iCT, CT in enumerate(CTs):
        fig, ax = plt.subplots(figsize=(6.4,2.60))
        fig.subplots_adjust(left=0.10, right=0.88, top=0.96, bottom=0.190,hspace=0.0, wspace=0)

        # --- Contour plots
        vy=np.linspace(0,2.0,50)
        vz=np.linspace(-5,1,51)

        fact=1
        gamma_t = -(1-np.sqrt(1-fact*CT))# <<< NOTE gamma factor here!!!
        print('gamma_t',gamma_t)
        dmz_dz = gamma_t * R**2 * np.pi # doublet intensity per length
        VCY,VCZ = np.meshgrid(vy,vz) 

        # --- Plotting models
        if Model['name'] in ['VC','VD','SS']:
            VC  = setupModel(CT=CT, HH=HH, Lambda=Lambda, Horizontal=Horizontal, Yaw=Yaw, Model=Model['name'])
            VCY,VCZ,VCV,VCW = VC.fVert(vy,vz)
        else:
            raise Exception('Model not supported')

        im,_,_ = plotPlaneVert(VCY,VCZ,VCV,VCW,ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed, cmap=cmap, linewidths=LW, levelsContour=levelsContour,levelsLines=levelsLines,component=component,colors=lineColorVC,alpha=lineAlpha,ls='solid' , axequal=False)
        fig.subplots_adjust(left=0.10, right=0.88, top=0.96, bottom=0.190,hspace=0.0, wspace=0)

        ax.set_ylabel('r/R [-]')
        ax.set_xlabel('x/R [-]')

        ax.set_xlim([np.min(vz),np.max(vz)])
        ax.set_ylim([np.min(vy),np.max(vy)])
        if CT==0.4:
            ax.text(-2.41,  0.32,'0.99',fontsize=11,ha='center',va='center',rotation=90)
            ax.text(-1.53,  0.32,'0.98',fontsize=11,ha='center',va='center',rotation=90)
            ax.text(-0.85 , 0.32,'0.95',fontsize=11,ha='center',va='center',rotation=90)
            ax.text(-0.22,  0.32,'0.9 ',fontsize=11,ha='center',va='center',rotation=90)
        else:
            ax.text(-4.4  , 0.33,'0.99',fontsize=11,ha='center',va='center',rotation=90)
            ax.text(-3.10,  0.33,'0.98',fontsize=11,ha='center',va='center',rotation=90)
            ax.text(-1.96,  0.33,'0.95',fontsize=11,ha='center',va='center',rotation=90)
            ax.text(-1.25,  0.33,'0.9 ',fontsize=11,ha='center',va='center',rotation=90)
            ax.text(-0.65,  0.33,'0.8 ',fontsize=11,ha='center',va='center',rotation=90)
            ax.text(-0.22,  0.33,'0.7 ',fontsize=11,ha='center',va='center',rotation=90)
        ax.title.set_text('Elementary_CT0{:2d}_HH99'.format(round(CT*100)))
        if iCT==1:
            cbar_ax = fig.add_axes([0.895, 0.19, 0.02, 0.772])
            cbar=fig.colorbar(im, cax=cbar_ax)

if __name__ == '__main__':
    main()
    plt.show()


