"""
References:
    [1] E. Branlard, A. Meyer Forsting, Using a cylindrical vortex model to assess the induction zone n front ofaligned and yawedd rotors
"""
#--- Legacy python 2.7
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# --- General
import unittest
import matplotlib.pyplot as plt
import numpy as np
# --- Local
from vortexcylinder.VortexCylinder import cylinder_tang_semi_inf_u

def main(test=False):
    if test:
        nZ=25
        nX=25
    else:
        nZ=200
        nX=200

    U0  = 1   ;
    R   = 1   ;
    ZMIN=-5*R; ZMAX=3*R;
    XMIN=-2*R; XMAX=2*R;
    CT0 = 0.95;
    gamma_t=-U0*(1-np.sqrt(1-CT0))*0.85;

    # --- Along z-Axis
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    for CT0 in [0.95,0.4]:
        gamma_t = -U0*(1-np.sqrt(1-CT0))*0.85;
        Zcp=np.linspace(ZMIN,ZMAX,nZ)
        Xcp=Zcp*0
        Ycp=Zcp*0
        ur,uz=cylinder_tang_semi_inf_u(Xcp,Ycp,Zcp,gamma_t,R)
        ax.plot(Zcp/R,(uz+U0)/U0,label='CT = {}'.format(CT0))
    ax.set_xlabel('z/R [-]')
    ax.set_ylabel('U/U0 [-]')
    ax.legend()
    ax.set_title('TwoCTsAxis')
    ax.set_xlim([-5, 3])
    ax.set_ylim([0.3, 1])

    # --- Along r-axis
    fig = plt.figure()
    ax  = fig.add_subplot(111)
    for CT0 in [0.95,0.4]:
        gamma_t = -U0*(1-np.sqrt(1-CT0))*0.85;
        Xcp=np.linspace(0,2*R,nZ)
        Zcp=Xcp*0
        Ycp=Xcp*0
        ur,uz=cylinder_tang_semi_inf_u(Xcp,Ycp,Zcp,gamma_t,R)
        ax.plot(Xcp/R,(uz+U0)/U0,label='u_z, CT = {}'.format(CT0))
        ax.plot(Xcp/R,ur        ,label='u_r, CT = {}'.format(CT0))
    ax.set_xlabel('r/R [-]')
    ax.set_ylabel('U/U0 [-]')
    ax.legend()
    ax.set_title('TwoCTsRotor')
    ax.set_ylim([0, 1.1])
    ax.set_xlim([0, XMAX])

    # --- Velocity field, CT=0.95
    CT0     = 0.95                       ;
    gamma_t = -U0*(1-np.sqrt(1-CT0))*0.85;
    z=np.linspace(3*R,-5*R,nZ)
    x=np.linspace(-2*R,2*R,nX)
    Z,X=np.meshgrid(z,x)
    Y=Z*0;
    ur,uz = cylinder_tang_semi_inf_u(X,Y,Z,gamma_t,R)

    # --- Plot the contours of axial induction
    levels=[0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1.01]
    fig=plt.figure()
    ax = fig.add_subplot(111)
    cs=plt.contour(Z,X,(uz+U0)/U0,levels=levels,colors='k')
    ax.clabel(cs,levels)
    ax.plot([ZMIN,ZMAX],[0,0],'k:')
    ax.plot([3*R,0,0,3*R],[-R,-R,R,R],'k-',linewidth=3)  # Rotor and Cyl
    ax.set_aspect('equal','box')
    ax.set_xlabel('z/R [-]')
    ax.set_ylabel('r/R [-]')
    ax.set_xlim([ZMIN,R])
    ax.set_ylim([XMIN,XMAX])
    ax.set_title('NoYawNoSwirlAxialInductionCT{:03d}'.format(int(CT0*100)))

    # --- Plot the contours of radial induction
    if test:
        levels=[0.005,0.01,0.05,0.1,0.2]
    else:
        levels=[0.01,0.05]
    fig=plt.figure()
    ax = fig.add_subplot(111)
    cs=plt.contour(Z,X,ur,levels=levels,colors='k')
    ax.clabel(cs,levels)
    ax.plot([ZMIN,ZMAX],[0,0],'k:')
    ax.plot([3*R,0,0,3*R],[-R,-R,R,R],'k-',linewidth=3)  # Rotor and Cyl
    ax.set_aspect('equal','box')
    ax.set_xlabel('z/R [-]')
    ax.set_ylabel('r/R [-]')
    ax.set_xlim([ZMIN,1])
    ax.set_ylim([XMIN,XMAX])
    ax.set_title('NoYawNoSwirlRadialInductionCT{:03d}'.format(int(CT0*100)))


    # ---
    if not test:
        plt.show()
    else:
        plt.close('all')

class Test(unittest.TestCase):
    def test_Article_Induction_NonYaw(self):
        import sys
        if sys.version_info >= (3, 0):
            main(test=True)
        else:
            print('Test skipped due to travis display error')

if __name__ == "__main__":
    main()


