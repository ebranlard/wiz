"""
References:
    [1] E. Branlard - Wind Turbine Aerodynamics and Vorticity Based Method, Springer, 2017
"""
#--- Legacy python 2.7
from __future__ import division
from __future__ import print_function
# --- General
import unittest
import numpy as np
import numpy.matlib
from scipy.special import ellipk, ellipe
# import warnings
# warnings.filterwarnings('error')

def ring_u(Xcp,Ycp,Zcp,Gamma=-1,R=1,cartesianOut=False,epsilon=0):
    """ 
    Compute the induced velocity from a vortex ring located at z=0
    """
    EPSILON = 1e-07
    # --- Main corpus
    Xcp=np.asarray(Xcp)
    Ycp=np.asarray(Ycp)
    Zcp=np.asarray(Zcp)
    if Xcp.shape==(0,):
        if cartesianOut:
            return np.array([]),np.array([]),np.array([]),
        else:
            return np.array([]), np.array([])

    r = np.sqrt(Xcp**2 + Ycp**2)
    z = Zcp
    ur = np.full(r.shape,np.nan)
    uz = np.full(r.shape,np.nan)

    # Enforcing  Axis formula : v_z=-Gamma/(2R) *1 / (1+(z/R)^2)^(3/2)  
    Iz = r < (EPSILON * R)
    ur[Iz] = 0
    uz[Iz] = Gamma/(2*R)*(1.0/((1 +(z[Iz]/R)**2)**(3.0/2.0)))

    # Value on the neighborhood of the ring itself..
    if epsilon==0:
        Ir = np.logical_and(np.abs(r-R)<(EPSILON*R), np.abs(z)<EPSILON)
        ur[Ir]=0
        uz[Ir]=Gamma/(4*R) # NOTE: this is arbitrary
    else:
        Ir = np.logical_and(np.abs(r-R)<(EPSILON*R), np.abs(z)<EPSILON)
        ur[Ir]=0
        uz[Ir]=Gamma/(4*np.pi*R)*(np.log(8*R/epsilon)-1/4) # Eq 35.36 from [1]

    # --- From this point on, variables have the size of ~Iz..
    bnIz = np.logical_and(np.logical_not(Iz), np.logical_not(Ir))
    r = r[bnIz]
    z = z[bnIz]

    # Formulation uses Formula from Yoon 2004
    a = np.sqrt((r+R)**2 + (z)**2)
    m = 4 * r * R / (a ** 2)
    A = (z)**2 + r**2 + R**2
    B = - 2*r*R
    K = ellipk(m)
    E = ellipe(m)
    I1 = np.multiply(4.0/a, K)
    I2 = 4.0 / a ** 3.0 * E / (1 - m)
    ur[bnIz] =Gamma/(4*np.pi)*R*(np.multiply(((z)/B),(I1 - np.multiply(A,I2))))
    uz[bnIz] =Gamma/(4*np.pi)*R*(np.multiply((R + np.multiply(r,A) / B),I2) - np.multiply(r/B,I1))

    if not cartesianOut:
        return ur,uz
    else:
        psi = np.arctan2(Ycp,Xcp)     ;
        ux=ur*np.cos(psi)
        uy=ur*np.sin(psi)
        return ux,uy,uz

def rings_u(Xcp,Ycp,Zcp,Gamma_r,Rr,Xr,Yr,Zr,cartesianOut=False,epsilon=0):
    """ 
    Compute the induced velocity from nRings vortex rings
        nRings: number of main rings
    TODO: angles

    INPUTS: 
        Xcp,Ycp,Zcp: cartesian coordinates of control points where the velocity field is not be computed
        Gamma_t : array of size (nRings), intensity of each rings
        R       : array of size (nRings), radius of each rings
        Xr,Yr,Zr: arrays of size (nRings), center of each rings
    """
    Xcp=np.asarray(Xcp)
    Ycp=np.asarray(Ycp)
    Zcp=np.asarray(Zcp)
    if cartesianOut:
        ux = np.zeros(Xcp.shape)
        uy = np.zeros(Xcp.shape)
        uz = np.zeros(Xcp.shape)
    else:
        ur = np.zeros(Xcp.shape)
        uz = np.zeros(Xcp.shape)
    for ir,(Gamma,R,xr,yr,zr) in enumerate(zip(Gamma_r,Rr,Xr,Yr,Zr)):
        if np.abs(Gamma) > 0:
            #print('.',end='')
            u1 = ring_u(Xcp-xr,Ycp-yr,Zcp-zr,Gamma,R,cartesianOut=cartesianOut,epsilon=epsilon)
            if cartesianOut:
                ux = ux + u1[0]
                uy = uy + u1[1]
                uz = uz + u1[2]
            else:
                ur = ur + u1[0]
                uz = uz + u1[1]
    if cartesianOut:
        return ux,uy,uz
    else:
        return ur,uz

# --------------------------------------------------------------------------------}
# --- TEST 
# --------------------------------------------------------------------------------{
class TestRing(unittest.TestCase):
    def test_singularities(self):
#         import warnings
#         warnings.filterwarnings('error')
        # ---- r=0, z=0, uz=Gamma/(2R)
        ur,uz=ring_u(0,0,0)
        np.testing.assert_almost_equal(uz,-1/2)
        # ---- r=1, z=0, NOTE: arbitrary
        ur,uz=ring_u(1,0,0)
        np.testing.assert_almost_equal(uz,-1/4)

    def test_axis(self):
        # Test that ring gives axis formula, eq 35.11 in reference [1]
        Gamma, R = -1, 10
        z=np.linspace(-2*R,2*R,20)
        x=z*0
        y=z*0
        ur,uz=ring_u(x,y,z,Gamma,R)
        uz_ref = Gamma/(2*R)*(1.0/(1 + (z/R)**2)**(3.0/2.0))
        np.testing.assert_almost_equal(uz,uz_ref,decimal = 7)
        np.testing.assert_almost_equal(ur,uz*0  ,decimal = 7)
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.plot(z,uz)
        #plt.plot(z,uz_ref,'k--')
        #plt.plot(z,ur)
        #plt.show()

    def test_axis_approx(self):
        # Test the approximate axis formula, close to axis, eq 35.22 in reference [1]
        Gamma, R= -1, 10
        z=np.linspace(-2*R,2*R,21)
        r=z*0+ 0.1*R
        y=z*0
        ur,uz=ring_u(r,y,z,Gamma,R)
        uz_ref =   Gamma/(2*R)*1/(1 + (z/R)**2)**(3/2)
        ur_ref = 3*Gamma/(4*R)*1/(1 + (z/R)**2)**(5/2) * r*z/R**2
        np.testing.assert_almost_equal(ur, ur_ref,decimal=3)
        np.testing.assert_almost_equal(uz, uz_ref,decimal=3)
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.plot(z,ur_ref)
        #plt.plot(z,ur,'--')
        #plt.plot(z,uz_ref)
        #plt.plot(z,uz,'--')
        #plt.show()
# 
    def test_rotor(self):
        pass
        # Test that induction on the rotor is constant, equal to gamma/2, see [1]
#         gamma_t, R= -1, 10
#         eps=10**-6 *R
#         x=np.linspace(0, 2*R, 1000)
#         x=np.linspace(R-eps, R+eps, 100)
#         y=x*0
#         z=x*0
#         ur,uz=ring_u(x,y,z,gamma_t,R)
        #uz_ref=[gamma_t/2]*len(x)
        #np.testing.assert_almost_equal(uz,uz_ref,decimal=7)
#         import matplotlib.pyplot as plt
#         plt.figure()
#         plt.plot(z,ur_ref)
#         plt.plot(x,ur,'--',label='ur')
#         plt.plot(z,uz_ref)
#         plt.plot(x,uz,'--',label='uz')
#         plt.legend()
#         plt.show()
# 
# 
#     def test_rings(self):
#         try:
#             from .VortexRing import ring_u
#         except:
#             try:
#                 from vortexcylinder.VortexRing import ring_u
#             except:
#                 from VortexRing import ring_u
# 
#         # Test that induction on the rotor is similat to the one from a bunch of rings
#         gamma_t, R= -1, 10
#         eps=10**-6 *R
#         x=np.linspace(-(R-eps), R-eps, 10)
#         y=x*0
#         z=x*0
#         ur,uz=cylinder_tang_semi_inf_u(x,y,z,gamma_t,R)
# #         uz_ref=[gamma_t/2]*len(x)
# #         np.testing.assert_almost_equal(uz,uz_ref,decimal=7)
#         print(ur)
#         print(uz)


if __name__ == "__main__":
#     TestRing().test_singularities()
#     TestRing().test_rotor()
    unittest.main()

