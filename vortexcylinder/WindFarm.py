#--- Legacy python 2.7
from __future__ import division
from __future__ import print_function
# --- General
import unittest
import numpy as np
# --- Local
try:
    from .VortexCylinder import vc_tang_u, vcs_tang_u
except:
    try:
        from vortexcylinder.VortexCylinder import vc_tang_u, vcs_tang_u
    except:
        from VortexCylinder import vc_tang_u, vcs_tang_u


def gridLayout(nx,xSpacing,nz,zSpacing=None,hub_height=0,mirror=False):
    """ Returned list of turbine positions on a grid layout
          y is vertical, positive upward
          x is lateral
          z is longi, positive along the wake
      If mirror is true, mirrored turbines are placed at y=-hubheight
    """
    if mirror:
        ny=2
    else:
        ny=1
    nWT = nz * nx * ny
    xWT = np.zeros((nWT))
    yWT = np.zeros((nWT))
    zWT = np.zeros((nWT))
    k   = 0
    for i in np.arange(1,nz+1).reshape(-1):
        for j in np.arange(1,nx+1).reshape(-1):
            k = k + 1
            xWT[k-1] = (j - int(np.floor(nx / 2)) - 1) * xSpacing
            zWT[k-1] = (i - 1) * zSpacing
            yWT[k-1] = hub_height
    if mirror:
        for i in np.arange(1,nz+1).reshape(-1):
            for j in np.arange(1,nx+1).reshape(-1):
                k = k + 1
                xWT[k-1] =  (j - int(np.floor(nx / 2)) - 1) * xSpacing
                zWT[k-1] =  (i - 1) * zSpacing
                yWT[k-1] = - hub_height
    return xWT,yWT,zWT


def windfarm_gridlayout_CTconst(Xcp,Ycp,Zcp,R,CT,U0,nxWT,xWTSpacing,nzWT,zWTSpacing,hub_height=0,mirror=False): 

    # Wind farm layout
    xWT,yWT,zWT= gridLayout(nxWT,xWTSpacing,nzWT,zWTSpacing,hub_height=hub_height,mirror=mirror)

    # Approximate a-CT-gamma relation
    a = 0.5 * (1 - np.sqrt(1 - CT))
    gamma_t = - 2 * U0 * a

    # All turbine are identical with constant CT/gamma 
    nWT      = len(xWT)
    nr       = 1
    vR       = np.zeros((nWT,nr)) + R
    vgamma_t = np.zeros((nWT,nr)) + gamma_t

    return vcs_tang_u(Xcp,Ycp,Zcp,vgamma_t,vR,xWT,yWT,zWT,epsilon=0)








# --------------------------------------------------------------------------------}
# --- TEST 
# --------------------------------------------------------------------------------{
class TestWindFarm(unittest.TestCase):

    def test_WF_layout(self):
        # test funciton that generates a grid layout, with possible mirror
        x,y,z=gridLayout(nx=2,xSpacing=2,nz=2,zSpacing=3,hub_height=2,mirror=True)
        np.testing.assert_almost_equal(x,[-2, 0,-2, 0, -2,  0, -2,  0])
        np.testing.assert_almost_equal(y,[ 2, 2, 2, 2, -2, -2, -2, -2])
        np.testing.assert_almost_equal(z,[ 0, 0, 3, 3,  0,  0,  3,  3])

        #from mpl_toolkits.mplot3d import Axes3D
        #import matplotlib.pyplot as plt
        #fig=plt.figure()
        #ax= fig.add_subplot(111,projection='3d')
        #ax.plot(z,x,y,'+')
        #ax.set_xlabel('z')
        #ax.set_ylabel('x')
        #ax.set_zlabel('y')
        #plt.show()

if __name__ == "__main__":
    unittest.main()
