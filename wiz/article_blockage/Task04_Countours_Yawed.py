""" 
Figure 8 of 
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
    # --- Plot parameters
    LW=0.8
    nStreamlines=0
    levelsContour=None
    levelsLines=None
    lineColorVC='k'
    lineAlpha=1.0

    minSpeed=0.5
    maxSpeed=1.05

    component='Axial'
    cmap,valuesOri=get_cmap(minSpeed,maxSpeed,alpha=0.6)

    nx=120
    nz=100

    ZMIN=-5
    ZMAX=3
    XMIN=-2
    XMAX=2
    vx=np.linspace(XMIN,XMAX,nx)
    vz=np.linspace(ZMIN,ZMAX,nz)

    HH     = 99
    Lambda = 9.9
    Yaw    = 0

    gamma_rings = [-0.21341,  -0.60414] # values to match mid velocity of AD

    for iCT,(CT0,gamma_t) in enumerate(zip([0.4,0.95],gamma_rings)):

        if CT0==0.95:
            levelsLines=np.sort(np.array([0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1.,1.01]))
        else:
            levelsLines=np.sort(np.array([0.7,0.8,0.85,0.9,0.95,0.98,0.99,0.995,1.01]))
        levelsContour = np.sort(np.concatenate((levelsLines,[1.4,0.2])))

        Yaw=-30 
        U0_n=np.cos(Yaw*np.pi/180);
        chi= Yaw*(1+0.3*(1-np.sqrt(1-CT0))); # deg
        chi=chi*np.pi/180
        m= np.tan(chi)

        Horizontal   = True
        VCH  = setupModel(CT = CT0, HH = HH, Lambda = Lambda, Horizontal = Horizontal, Yaw=Yaw, gamma_t=gamma_t)
        VCH.X,VCH.Z=np.meshgrid(vx,vz)
        VCH.X,VCH.Z,VCH.U,VCH.W = VCH.fHorz(vx,vz)
        VCH.U= VCH.U/U0_n
        VCH.W= VCH.W/U0_n

        fig, ax = plt.subplots(figsize=(6.4,3.10))
        im, _, _ = plotPlane(VCH, Horizontal=Horizontal, ax=ax, minSpeed=minSpeed, maxSpeed=maxSpeed, cmap=cmap, linewidths=LW, nStreamlines=nStreamlines,levelsContour=levelsContour,levelsLines=levelsLines,component=component,colors=lineColorVC,alpha=lineAlpha, ls='solid', axequal=False)
        # ax.title.set_text(data.base+'_'+component)

        ax.title.set_text('YawVCInductionCT0{:2d}'.format(round(CT0*100)))

        fig.subplots_adjust(left=0.10, right=0.88, top=0.98, bottom=0.150,hspace=0.0, wspace=0)
        if iCT==1:
            cbar_ax = fig.add_axes([0.895, 0.15, 0.02, 0.832])
            cbar=fig.colorbar(im, cax=cbar_ax)

        zz=np.linspace(0,max(vz),100); xx=m*zz;
        ax.plot(zz,xx+1,'k-',lw=3);
        ax.plot(zz,xx-1,'k-',lw=3);
        zz=np.linspace(min(vz),max(vz),100); xx=m*zz;
        ax.plot(zz,xx+0,'k-.',lw=1);
        ax.set_ylim([XMIN, XMAX])
        ax.set_xlim([ZMIN, ZMAX])
        ax.set_xlabel('x/R [-]')
        ax.set_ylabel('y/R [-]')

        if CT0==0.95:
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
    #     end

if __name__ == '__main__':
    main()
    plt.show()
