"""
References:
    [1] E. Branlard, M. Gaunaa - Cylindrical vortex wake model: skewed cylinder, application to yawed or tilted rotors - Wind Energy, 2015
    [2] E. Branlard - Wind Turbine Aerodynamics and Vorticity Based Method, Springer, 2017
"""
#--- Legacy python 2.7
from __future__ import division
from __future__ import print_function
# --- General
import unittest
import numpy as np
import numpy.matlib
# --- Local

def skewedcylinder_tang_semi_inf_u(Xcp,Ycp,Zcp,gamma_t=-1,R=1,m=0,ntheta=180):
    """ Induced velocity from a skewed semi infinite cylinder of tangential vorticity.
    The cylinder axis is defined by x=m.z, m=tan(chi). The rotor is in the plane z=0.
    See Reference below.
    INPUTS:
       Xcp,Ycp,Zcp: vector or matrix of control points Cartesian Coordinates
       gamma_t    : tangential vorticity of the vortex sheet (circulation per unit of length oriented along psi). (for WT rotating positively along psi , gamma psi is negative)
       R          : radius of cylinder
       m =tan(chi): tangent of wake skew angle
    INPUTS (Optional):
       ntheta    : number of points used for integration
       Reference: [1,2]"""
    EPSILON_AXIS=1e-7; # relative threshold for using axis formula

    # --- Main corpus
    shape_in = np.asarray(Xcp).shape
    Xcp=np.asarray(Xcp).ravel()
    Ycp=np.asarray(Ycp).ravel()
    Zcp=np.asarray(Zcp).ravel()
#     if Xcp.shape==(0,):
#         if CartesianOut:
#             return np.array([]),np.array([]),np.array([]),
#         else:
#             return np.array([]), np.array([])
    # --- Conversion to polar coordinates
    # r,vpsi,z   : polar coordinates of control points
    vr   = np.sqrt(Xcp**2+Ycp**2)
    vpsi = np.arctan2(Ycp,Xcp)
    vz   = Zcp;

    # --- Performing integration over theta for all control points
    vtheta = np.pi/2 + np.linspace(0, 2*np.pi, ntheta)
    # Constants of theta
    c  = 1 + m**2
    bx = - R * np.cos(vtheta)
    by = - R * np.sin(vtheta)
    bz =   R * m * np.cos(vtheta)
    
    u_z   = np.zeros(vr.shape)
    u_x   = np.zeros(vr.shape)
    u_y   = np.zeros(vr.shape)
    u_psi = np.zeros(vr.shape)
    # ---- Loop on all control points to find velocity
    for i,(r,psi,z) in enumerate(zip(vr,vpsi,vz)):
        # Functions of theta in the integrand
        a = R**2 + r** 2 + z**2 - 2*R*r*np.cos(vtheta - psi)
        b = 2 * m * R * np.cos(vtheta) - 2 * m * r * np.cos(psi) - 2 * z
        ax = R * z * np.cos(vtheta)
        ay = R * z * np.sin(vtheta)
        az = R * (R - r * np.cos(vtheta - psi))
        ap = R * z * np.sin(vtheta - psi)
        bp = -   R * np.sin(vtheta - psi)
        # Integrand
        dI_x   = 2 *(ax * np.sqrt(c)+ np.multiply(bx,np.sqrt(a)))/(np.multiply(np.sqrt(a),(2 * np.sqrt(a * c)+ b)))
        dI_y   = 2 *(ay * np.sqrt(c)+ np.multiply(by,np.sqrt(a)))/(np.multiply(np.sqrt(a),(2 * np.sqrt(a * c)+ b)))
        dI_z   = 2 *(az * np.sqrt(c)+ np.multiply(bz,np.sqrt(a)))/(np.multiply(np.sqrt(a),(2 * np.sqrt(a * c)+ b)))
        dI_psi = 2 *(ap * np.sqrt(c)+ np.multiply(bp,np.sqrt(a)))/(np.multiply(np.sqrt(a),(2 * np.sqrt(a * c)+ b)))
        # Integrations
        u_x[i]    = gamma_t/(4*np.pi) * np.trapz(dI_x  , vtheta)
        u_y[i]    = gamma_t/(4*np.pi) * np.trapz(dI_y  , vtheta)
        u_z[i]    = gamma_t/(4*np.pi) * np.trapz(dI_z  , vtheta)
        u_psi[i]  = gamma_t/(4*np.pi) * np.trapz(dI_psi, vtheta)
    
    # Reverting to original shape
    u_z   =  u_z.reshape(shape_in)   
    u_x   =  u_x.reshape(shape_in)   
    u_y   =  u_y.reshape(shape_in)   
    u_psi =  u_psi.reshape(shape_in) 
    vpsi  =  vpsi.reshape(shape_in) 
    # --- Projections onto r, zeta, xi
    coschi = 1/np.sqrt(1+m**2)
    sinchi = m/np.sqrt(1+m**2)
    u_r = np.multiply(u_x,np.cos(vpsi)) + np.multiply(u_y,np.sin(vpsi))
    u_zeta = u_z * coschi + u_x * sinchi
    u_xi = - u_z * sinchi + u_x * coschi
    return u_x,u_y,u_z,u_r,u_psi,u_zeta,u_xi
#     ur = np.full(r.shape,np.nan)
#     Iz = r < (EPSILON_AXIS * R)
#     # --- From this point on, variables have the size of ~Iz..
#     bnIz = np.logical_not(Iz)
#     if not cartesianOut:
#         return ur,uz
#     else:
#         psi = np.arctan2(Ycp,Xcp)     ;
#         ux=ur*np.cos(psi)
#         uy=ur*np.sin(psi)
#         return ux,uy,uz

def skewedcylinders_tang_semi_inf_u(Xcp,Ycp,Zcp,gamma_t,R,m,Xcyl,Ycyl,Zcyl,ntheta=180):
    """ 
    Computes the velocity field for nCyl*nr cylinders, extending along z:
        nCyl: number of main cylinders
        nr  : number of concentric cylinders within a main cylinder 

    TODO: angles

    INPUTS: 
        Xcp,Ycp,Zcp: cartesian coordinates of control points where the velocity field is not be computed
        gamma_t: array of size (nCyl,nr), distribution of gamma for each cylinder as function of radius
        R      : array of size (nCyl,nr), 
        m      : array of size (nCyl,nr), 
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
            print('.',end='')
            if np.abs(gamma_t[i,j]) > 0:
                ux1,uy1,uz1,_,_,_,_ = skewedcylinder_tang_semi_inf_u(Xcp-Xcyl[i],Ycp-Ycyl[i],Zcp-Zcyl[i],gamma_t[i,j],R[i,j],m[i,j],ntheta=ntheta)
                ux = ux + ux1
                uy = uy + uy1
                uz = uz + uz1
    print('')
    return ux,uy,uz
    




# --------------------------------------------------------------------------------}
# --- TEST 
# --------------------------------------------------------------------------------{
class TestSkewedCylinder(unittest.TestCase):
    def test_rotor(self):
        """ See paragraph "Properties on the rotor disk" of [1] """
        # data
        gamma_t,R,chi = -5, 10, 30*np.pi/180
        m=np.tan(chi) # tan(chi) 
        eps=10**-1 *R
        # --- At rotor center (see also next test, stronger)
        u_x,u_y,u_z,u_r,u_psi,u_zeta,u_xi=skewedcylinder_tang_semi_inf_u(0,0,0,gamma_t,R,m)
        uz0=gamma_t/2
        np.testing.assert_almost_equal(u_x        ,np.tan(chi/2)*uz0      ,decimal=7)
        np.testing.assert_almost_equal(u_z        ,uz0 ,decimal=7)
        np.testing.assert_almost_equal(u_zeta     ,uz0 ,decimal=7)

        # --- At psi=pi/2 (i.e. x=0), z=0 (Eq 9 from [1]), ux,uz,uzeta,uxi constant!
        y=np.linspace(0,R-eps,4)
        x=y*0
        z=y*0
        u_x,u_y,u_z,u_r,u_psi,u_zeta,u_xi=skewedcylinder_tang_semi_inf_u(x,y,z,gamma_t,R,m)
        uz0=np.asarray([gamma_t/2]*len(x))
        np.testing.assert_almost_equal(u_zeta     ,uz0                    ,decimal=7)
        np.testing.assert_almost_equal(u_z        ,uz0                    ,decimal=7)
        np.testing.assert_almost_equal(u_xi/u_zeta,[-np.tan(chi/2)]*len(x),decimal=7)
        np.testing.assert_almost_equal(u_x /u_z   ,[ np.tan(chi/2)]*len(x),decimal=7)
        np.testing.assert_almost_equal(u_x        ,uz0*np.tan(chi/2)      ,decimal=7)

        # --- Component zeta over the entire plane is g_t/2 (Eq 10 from [1])
        vR,vPsi = np.meshgrid(np.linspace(0,R-eps,5), np.linspace(0,2*np.pi,12))
        x=vR*np.cos(vPsi)
        y=vR*np.sin(vPsi)
        z=x*0
        u_x,u_y,u_z,u_r,u_psi,u_zeta,u_xi=skewedcylinder_tang_semi_inf_u(x,y,z,gamma_t,R,m)
        uz0=gamma_t/2
        np.testing.assert_almost_equal(u_zeta , uz0 ,decimal=5)

        # --- Plane y=0 (anti-)symmetry - x,z,zeta,xi: symmetric - y: anti-symmetric
        x,y = np.meshgrid(np.linspace(-R/3,R/3,3), [-R/2,R/2] )
        z=x*0
        u_x,u_y,u_z,u_r,u_psi,u_zeta,u_xi=skewedcylinder_tang_semi_inf_u(x,y,z,gamma_t,R,m)
        np.testing.assert_almost_equal(u_x   [0,:], u_x   [1,:])
        np.testing.assert_almost_equal(u_z   [0,:], u_z   [1,:])
        np.testing.assert_almost_equal(u_zeta[0,:], u_zeta[1,:])
        np.testing.assert_almost_equal(u_xi  [0,:], u_xi  [1,:])
        np.testing.assert_almost_equal(u_y   [0,:],-u_y   [1,:]) # anti-symmetric

        # --- Radial anti-symmetry of components x,y,z about their origin value
        r0   = R/2 # cannot do negative r here since definition of r is positive
        psi0 = np.linspace(0,np.pi,10)
        vPsi = np.array([psi0 , psi0+np.pi] ).T
        x=r0*np.cos(vPsi)
        y=r0*np.sin(vPsi)
        z=x*0
        u_x,u_y,u_z,u_r,u_psi,u_zeta,u_xi       =skewedcylinder_tang_semi_inf_u(x,y,z,gamma_t,R,m)
        u_x0,u_y0,u_z0,u_r0,u_psi0,u_zeta0,u_xi0=skewedcylinder_tang_semi_inf_u(0,0,0,gamma_t,R,m)
        np.testing.assert_almost_equal(u_x   [:,0]+u_x   [:,1], 2*u_x0   )
        np.testing.assert_almost_equal(u_y   [:,0]+u_y   [:,1], 2*u_y0   )
        np.testing.assert_almost_equal(u_z   [:,0]+u_z   [:,1], 2*u_z0   )
        np.testing.assert_almost_equal(u_zeta[:,0]+u_zeta[:,1], 2*u_zeta0)
        np.testing.assert_almost_equal(u_xi  [:,0]+u_xi  [:,1], 2*u_xi0  )


    def test_farwake(self):
        """ See paragraph "Properties on the rotor disk" of [1] """
        # data
        gamma_t,R,chi = -5, 10, 30*np.pi/180
        m=np.tan(chi) # tan(chi) 
        eps=10**-1 *R
        z0=1000*R # Far wake
        # --- At rotor center (see also next test, stronger)
        #u_x,u_y,u_z,u_r,u_psi,u_zeta,u_xi=skewedcylinder_tang_semi_inf_u(0,0,0,gamma_t,R,m)
        #uz0=gamma_t/2
        #np.testing.assert_almost_equal(u_x        ,np.tan(chi/2)*uz0      ,decimal=7)
        #np.testing.assert_almost_equal(u_z        ,uz0 ,decimal=7)
        #np.testing.assert_almost_equal(u_zeta     ,uz0 ,decimal=7)
        # --- Component zeta over the entire plane is g_t/2 (Eq 10 from [1])
        vR,vTheta = np.meshgrid(np.linspace(0,R-eps,5), np.linspace(0,2*np.pi,12))
        x=vR*np.cos(vTheta)+z0*m
        y=vR*np.sin(vTheta)
        z=x*0 + z0
        u_x,u_y,u_z,u_r,u_psi,u_zeta,u_xi=skewedcylinder_tang_semi_inf_u(x,y,z,gamma_t,R,m)
        np.testing.assert_almost_equal(u_zeta , gamma_t               , decimal=5)
        np.testing.assert_almost_equal(u_xi  , -gamma_t*np.tan(chi/2) , decimal=5)
        np.testing.assert_almost_equal(u_z   ,  gamma_t               , decimal=5)
        np.testing.assert_almost_equal(u_x   ,  gamma_t*np.tan(chi/2) , decimal=5)
        
        #print('ux',u_x)
        #print('uy',u_y)
        #print('uz',u_z)
        #print('ur',u_r)
        #print('upsi',u_psi)
        #print('uzeta',u_zeta)
        #print('uxi',u_xi)

#     def test_singularities(self):
#         # TODO!
# 
#     def test_regularization(self):
#         #TODO
# 
#     def test_multirotor(self):
#         #TODO
# 
#     def test_rings(self):
#         #TODO

if __name__ == "__main__":
#     TestCylinder().test_singularities()
#     TestCylinder().test_singularities()
#     TestCylinder().test_rings()
    unittest.main()
