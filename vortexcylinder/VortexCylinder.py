"""
References:
    [1] E. Branlard, M. Gaunaa - Cylindrical vortex wake model: right cylinder - Wind Energy, 2014
    [2] E. Branlard - Wind Turbine Aerodynamics and Vorticity Based Method, Springer, 2017
    [3] E. Branlard, A. Meyer Forsting, Using a cylindrical vortex model to assess the induction zone n front of aligned and yawedd rotors, in Proceedings of EWEA Offshore Conference, 2015
"""
#--- Legacy python 2.7
from __future__ import division
from __future__ import print_function
# --- General
import unittest
import numpy as np
import numpy.matlib
# --- Local
try:
    from .elliptic import ellipticPiCarlson, ellipe, ellipk
except:
    try:
        from vortexcylinder.elliptic import ellipticPiCarlson, ellipe, ellipk
    except:
        from elliptic import ellipticPiCarlson, ellipe, ellipk



def cylinder_tang_semi_inf_u(Xcp,Ycp,Zcp,gamma_t=-1,R=1,cartesianOut=False,epsilon=0):
    """ Induced velocity from a semi infinite cylinder extending along the z axis, starting at z=0
    INPUTS:
      Xcp,Ycp,Zcp: vector or matrix of control points coordinates
      gamma_t: tangential vorticity sheet strength of the cylinder
      R: cylinder radius
      epsilon : Regularization parameter, e.g. epsilon=0.0001*R
    Reference: [1]"""
    EPSILON_AXIS=1e-7; # relative threshold for using axis formula

    # --- Main corpus
    Xcp=np.asarray(Xcp)
    Ycp=np.asarray(Ycp)
    Zcp=np.asarray(Zcp)
    if Xcp.shape==(0,):
        if CartesianOut:
            return np.array([]),np.array([]),np.array([]),
        else:
            return np.array([]), np.array([])


    r = np.sqrt(Xcp ** 2 + Ycp ** 2)
    z = Zcp
    ur = np.full(r.shape,np.nan)
    uz = np.full(r.shape,np.nan)
    # Enforcing axis formula : v_z=-Gamma/(2) *( 1 + z / sqrt(R^2+z^2))
    Iz = r < (EPSILON_AXIS * R)
    ur[Iz] = 0
    uz[Iz] = gamma_t/2 * (1 + z[Iz] / np.sqrt(z[Iz]** 2 + R**2))

    # --- From this point on, variables have the size of ~Iz..
    bnIz = np.logical_not(Iz)
    r = r[bnIz]
    z = z[bnIz]

    # Eliptic integrals
    if epsilon==0:
        k_2  = 4 * r * R / ((R + r)**2 + z**2)
        k0_2 = 4 * r * R/  ((R + r)**2       )
    else:
        epsilon2= r*0+epsilon**2
        epsilon2[z<0]=0 # No regularization when z<0 # TODO
        k_2  = 4 * r * R / ((R + r)**2 + z**2 + epsilon2)
        k0_2 = 4 * r * R/  ((R + r)**2        + epsilon2)
    k = np.sqrt(k_2)
    EE = ellipe(k_2)
    KK = ellipk(k_2)
    #     PI = ellippi(k0_2,k_2)
    PI = ellipticPiCarlson(k0_2,k_2)
    # --- Special values
    PI[PI==np.inf]==0
    PI[r==R]=0 ; # when r==R, PI=0 TODO, check
    KK[KK==np.inf]=0 ; # when r==R, K=0  TODO, check
    # ---
    ur[bnIz] = -gamma_t/(2*np.pi) * np.multiply(np.sqrt(R/r) , np.multiply((2-k_2)/k,KK) - np.multiply(2.0/k, EE))
    # Term 1 has a singularity at r=R, # T1 = (R-r + np.abs(R-r))/(2*np.abs(R-r))
    T1=np.zeros(r.shape) 
    T1[r==R] = 1/2
    T1[r<R]  = 1
    if (epsilon!=0):
        # TODO, more work needed on regularization
        epsilon2= r*0+epsilon**2
        b=z>=0
        T1[b]=1/2*(1 + (R-r[b])*np.sqrt(1+epsilon2[b]/(R+r[b])**2)/np.sqrt((R-r[b])**2 +epsilon2[b]))
    uz[bnIz] = gamma_t/2*( T1 + np.multiply(np.multiply(z,k) / (2 * np.pi * np.sqrt(r * R)),(KK + np.multiply((R - r)/(R + r),PI))))
    
    if not cartesianOut:
        return ur,uz
    else:
        psi = np.arctan2(Ycp,Xcp)     ;
        ux=ur*np.cos(psi)
        uy=ur*np.sin(psi)
        return ux,uy,uz

def cylinder_tang_semi_inf_u_raw(Xcp,Ycp,Zcp,gamma_t=-1,R=1):
    """ Induced velocity from a semi infinite cylinder extending along the z axis, starting at z=0
    This routine has no special handling, see cylinder_tang_semi_inf_u for Documentation
    """
    r = np.sqrt(Xcp ** 2 + Ycp ** 2)
    z = Zcp
    ur = np.full(r.shape,np.nan)
    uz = np.full(r.shape,np.nan)
    k_2  = 4 * r * R / ((R + r)**2 + z**2)
    k0_2 = 4 * r * R/  ((R + r)**2       )
    k = np.sqrt(k_2)
    EE = ellipe(k_2)
    KK = ellipk(k_2)
    PI = ellipticPiCarlson(k0_2,k_2)
    ur = -gamma_t/(2*np.pi) * np.multiply( np.sqrt(R/r), (np.multiply((2 - k_2) / k,KK) - np.multiply(2.0 / k,EE)))
    T1 = (R-r + np.abs(R-r))/(2*np.abs(R-r))
    uz = gamma_t/2*( T1 + np.multiply(np.multiply(z,k)/(2*np.pi * np.sqrt(r * R)),(KK + np.multiply((R - r)/(R + r),PI))))

    return ur,uz

def cylinders_tang_semi_inf_u(Xcp,Ycp,Zcp,gamma_t,R,Xcyl,Ycyl,Zcyl,epsilon=0):
    """ 
    Computes the velocity field for nCyl*nr cylinders, extending along z:
        nCyl: number of main cylinders
        nr  : number of concentric cylinders within a main cylinder 

    TODO: angles

    INPUTS: 
        Xcp,Ycp,Zcp: cartesian coordinates of control points where the velocity field is not be computed
        gamma_t: array of size (nCyl,nr), distribution of gamma for each cylinder as function of radius
        R      : array of size (nCyl,nr), 
        Xcyl,Ycyl,Zcyl: array of size nCyl) giving the center of the rotor
        All inputs should be numpy arrays
    """ 
    Xcp=np.asarray(Xcp)
    Ycp=np.asarray(Ycp)
    Zcp=np.asarray(Zcp)
    ux = np.zeros(Xcp.shape)
    uy = np.zeros(Xcp.shape)
    uz = np.zeros(Xcp.shape)
    
    nCyl,nr = R.shape
    for i in np.arange(nCyl):
        for j in np.arange(nr):
#             print('.',end='')
            if np.abs(gamma_t[i,j]) > 0:
                ux1,uy1,uz1 = cylinder_tang_semi_inf_u(Xcp-Xcyl[i],Ycp-Ycyl[i],Zcp-Zcyl[i],gamma_t[i,j],R[i,j],cartesianOut=True,epsilon=epsilon)
                ux = ux + ux1
                uy = uy + uy1
                uz = uz + uz1
#     print('')
    return ux,uy,uz
    








# --------------------------------------------------------------------------------}
# --- TEST 
# --------------------------------------------------------------------------------{
class TestCylinder(unittest.TestCase):
    def test_singularities(self):
        import warnings
        warnings.filterwarnings('error')
        # ---- r=1, z=0
        ur,uz=cylinder_tang_semi_inf_u(1,0,0)
        np.testing.assert_almost_equal(uz,-1/4)
        ur,uz=cylinder_tang_semi_inf_u(1,0,0,epsilon=1e-1)
        np.testing.assert_almost_equal(uz,-1/4)
        # ---- r=0, z=0
        ur,uz=cylinder_tang_semi_inf_u(0,0,0)
        np.testing.assert_almost_equal(uz,-1/2)
        ur,uz=cylinder_tang_semi_inf_u(0,0,0,epsilon=1e-1)
        np.testing.assert_almost_equal(uz,-1/2)

        # --- At r=1, z=-1, PI=infinity
        ur,uz   = cylinder_tang_semi_inf_u(1,0,-1)
        np.testing.assert_almost_equal(uz,-0.08934,decimal=4)
        ure,uze = cylinder_tang_semi_inf_u(1,0,-1,epsilon = 1e-1)
        np.testing.assert_almost_equal(uz,uze)
        np.testing.assert_almost_equal(ur,ure)

    def test_regularization(self):
        # TODO!
        r=np.array([0,0.99,1,1.01])
        # --- Regularization should have no impact when z<<-epsilon
        z=r*0-1
        ur , uz  = cylinder_tang_semi_inf_u(r,0,z)
        ure, uze = cylinder_tang_semi_inf_u(r,0,z,epsilon=1e-1)
        np.testing.assert_almost_equal(uz,uze)
        np.testing.assert_almost_equal(ur,ure)
        # --- Regularization at the rotor
#         r=np.linspace(0,2,200)
#         z=r*0-0.000001
#         ur , uz  = cylinder_tang_semi_inf_u(r,0,z)
#         ure, uze = cylinder_tang_semi_inf_u(r,0,z,epsilon=1e-1)
#         z0=r*0
#         ure0, uze0= cylinder_tang_semi_inf_u(r,0,z0,epsilon=1e-1)
#         import matplotlib.pyplot as plt
#         plt.figure()
#         plt.plot(r,uz)
#         plt.plot(r,uze)
#         plt.plot(r,uze0)
#         plt.xlabel('r') 
#         plt.ylabel('uz') 
# #         plt.figure()
# #         plt.plot(r,ur)
# #         plt.plot(r,ure)
# #         plt.xlabel('r') 
# #         plt.ylabel('ur') 
#         plt.show()

    def test_multirotor(self):
        import warnings
        warnings.filterwarnings('error')
        # --- Typical values used
        U0      = 1
        R       = 1
        CT0     = 0.95                       ;
        gamma_t = -U0*(1-np.sqrt(1-CT0))/2
        # --- Typical Control points used
        N=101
        z=np.linspace(-5*R,3*R,N)
        x=np.linspace(-4*R,4*R,N)
        Z,X=np.meshgrid(z,x)
        Y=Z*0;

        # --- One cylinder should match standard call
        nCyl,nr = 1, 1
        Xcyl=[0]
        Ycyl=[0]
        Zcyl=[0]
        vR       = np.zeros((nCyl,nr))
        vgamma_t = np.zeros((nCyl,nr))
        vR[0,0]      = R
        vgamma_t[0,0]= gamma_t

        ux,uy,uz             = cylinders_tang_semi_inf_u(X,Y,Z,vgamma_t,vR,Xcyl,Ycyl,Zcyl)
        ux_ref,uy_ref,uz_ref = cylinder_tang_semi_inf_u(X,Y,Z,gamma_t,R,cartesianOut=True)
        np.testing.assert_almost_equal(ux,ux_ref)
        np.testing.assert_almost_equal(uz,uz_ref)

        # --- Two cylinders at same location but opposite sign should give zero
        nCyl,nr = 2, 1
        Xcyl=[0,0]
        Ycyl=[0,0]
        Zcyl=[0,0]
        vR       = np.zeros((nCyl,nr))
        vgamma_t = np.zeros((nCyl,nr))
        vR[:,0]      = R
        vgamma_t[0,0]= gamma_t
        vgamma_t[1,0]= -gamma_t
        ux,uy,uz             = cylinders_tang_semi_inf_u(X,Y,Z,vgamma_t,vR,Xcyl,Ycyl,Zcyl)
        np.testing.assert_almost_equal(ux,ux*0)
        np.testing.assert_almost_equal(uz,uz*0)

        # --- Playground
#         nCyl,nr = 2, 1
#         Xcyl=[2.1*R,-2.1*R]
#         Zcyl=[0,0]
#         Ycyl=[0,0]
#         N=101
#         vR       = np.zeros((nCyl,nr))
#         vgamma_t = np.zeros((nCyl,nr))
#         vR[0,0]            = R
#         vR[nCyl-1,0]       = R
#         vgamma_t[0,0]      = gamma_t
#         vgamma_t[nCyl-1,0] = gamma_t
#         ux,uy,uz=cylinders_tang_semi_inf_u(X,Y,Z,vgamma_t,vR,Xcyl,Ycyl,Zcyl,epsilon=0e-1*R)
#         uz=uz+U0 #<<<<<<<<<<<<<<<<<<
# 
#         # --- Plot the contours of axial induction
#         import matplotlib.pyplot as plt
#         levels=[0.5,0.6,0.7,0.8,0.9,0.95,0.98,0.99,1.01]
#         fig=plt.figure()
#         ax = fig.add_subplot(111)
# #         plt.contourf(Z,X,uz/U0,30)
#         plt.contourf(Z,X,np.sqrt(uz**2+ux**2),30)
# #         plt.contourf(Z,X,(uz+U0)/U0,levels=levels)
# #         cs=plt.contour(Z,X,(uz+U0)/U0,levels=levels,colors='k')
# #         ax.clabel(cs,levels)
#         xs=np.array([0,Xcyl[0]-R,Xcyl[0]-R/2,Xcyl[0],Xcyl[0]+R/2,Xcyl[0]+R])
#         print(xs)
#         seed_points=np.array([xs*0-3,xs])
#         strm= ax.streamplot(Z,X,uz,ux,color='k',start_points=seed_points.T)
# #         ax.plot(seed_points[0],seed_points[1],'bo')
#         ax.set_aspect('equal','box')
#         ax.set_xlabel('z/R [-]')
#         ax.set_ylabel('r/R [-]')
#         plt.colorbar()
# #         fig.colorbar(strm.lines)
# #         ax.set_xlim([ZMIN,R])
# #         ax.set_ylim([XMIN,XMAX])
# 
# #         fig=plt.figure()
# #         ax = fig.add_subplot(111)
# #         xs=x[0:-1:2]
# #         seed_points=np.array([xs*0+1,xs])
# #         strm= ax.streamplot(Z,X,uz+U0,ux,color=np.sqrt(ux**2+uz**2),start_points=seed_points.T)
# #         ax.plot(seed_points[0],seed_points[1],'bo')
# #         fig.colorbar(strm.lines)
#         plt.show()



    def test_axis(self):
        # Test that cylinder gives axis formula, eq 36.72 in reference [2]
        gamma_t, R= -1, 10
        z=np.linspace(-2*R,2*R,20)
        x=z*0
        y=z*0
        ur,uz=cylinder_tang_semi_inf_u(x,y,z,gamma_t,R)
        uz_ref = gamma_t/2*(1 + z/np.sqrt(z**2 + R**2))
        np.testing.assert_almost_equal(uz, uz_ref,decimal=7)
        np.testing.assert_almost_equal(ur, uz*0,decimal=7)

        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.plot(z,uz)
        #plt.plot(z,ur)
        #plt.show()

    def test_axis_approx(self):
        # Test the approximate axis formula, close to axis, eq 36.72 in reference [2]
        gamma_t, R= -1, 10
        z=np.linspace(-2*R,2*R,21)
        x=z*0+ 0.1*R
        y=z*0
        ur,uz=cylinder_tang_semi_inf_u(x,y,z,gamma_t,R)
        uz_ref = gamma_t/2*(1 + z/np.sqrt(z**2 + R**2))
        ur_ref =-gamma_t/4*(x*R**2/(z**2 + R**2)**(3/2))
        np.testing.assert_almost_equal(ur, ur_ref,decimal=3)
        np.testing.assert_almost_equal(uz, uz_ref,decimal=3)
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.plot(z,ur_ref)
        #plt.plot(z,ur,'--')
        #plt.plot(z,uz_ref)
        #plt.plot(z,uz,'--')
        #plt.show()

    def test_rotor(self):
        # Test that induction on the rotor is constant, equal to gamma/2, see [1]
        gamma_t, R= -1, 10
        eps=10**-6 *R
        x=np.linspace(-(R-eps), R-eps, 10)
        y=x*0
        z=x*0
        ur,uz=cylinder_tang_semi_inf_u(x,y,z,gamma_t,R)
        uz_ref=[gamma_t/2]*len(x)
        np.testing.assert_almost_equal(uz,uz_ref,decimal=7)

    def test_farwake(self):
        # Test that induction in the far wake is constant, equal to gamma, see [1]
        gamma_t, R= -1, 10
        eps=10**-5 *R
        x=np.linspace(-(R-eps), R-eps, 10)
        y=x*0
        z=x*0 + 10**4*R
        ur,uz=cylinder_tang_semi_inf_u(x,y,z,gamma_t,R)
        uz_ref,ur_ref = [gamma_t]*len(x) ,  x*0
        np.testing.assert_almost_equal(uz,uz_ref,decimal=5)
        np.testing.assert_almost_equal(ur,ur_ref,decimal=7)

    def test_rings(self):
        pass
        #TODO
        try:
            from .VortexRing import ring_u
        except:
            try:
                from vortexcylinder.VortexRing import ring_u
            except:
                from VortexRing import ring_u

        # Test that induction on the rotor is similat to the one from a bunch of rings
        gamma_t, R= -1, 10
        eps=10**-6 *R
        x=np.linspace(-(R-eps), R-eps, 10)
        y=x*0
        z=x*0
        ur,uz=cylinder_tang_semi_inf_u(x,y,z,gamma_t,R)
        #         uz_ref=[gamma_t/2]*len(x)
        #         np.testing.assert_almost_equal(uz,uz_ref,decimal=7)
        #print(ur)
        #print(uz)


if __name__ == "__main__":
#     TestCylinder().test_singularities()
#     TestCylinder().test_singularities()
#     TestCylinder().test_rings()
    unittest.main()
