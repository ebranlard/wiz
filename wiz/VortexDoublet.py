#--- Legacy python 2.7
from __future__ import division
import numpy as np
import unittest
    

def doublet_u(Xcp,Ycp,Zcp,m,x0=[0,0,0]): 
    """
    Velocity field induced by one doublet located at x0 with vector intensity m
         u = 1/(4pi)       *( 3* r * m.r/|r|**5 - m/|r|**3)
         u = 1/(4pi |r|**3)*( 3* r * m.r/|r|**2 - m)
    Xcp, Ycp, Zcp: Cartesian coordinates of control points
    """
    
    Xcp=np.asarray(Xcp)
    Ycp=np.asarray(Ycp)
    Zcp=np.asarray(Zcp)
    
    DX = Xcp - x0[0]
    DY = Ycp - x0[1]
    DZ = Zcp - x0[2]

    r_norm3 = (DX**2 + DY**2 + DZ**2 )**(3/2) # |r|**3
    r_norm3=np.asarray(r_norm3)

    # --- Avoiding singularity by introducing a temporarily fake norm
    bSing= r_norm3<1e-16
    r_norm3[bSing]=1 # Temporary hack, replaced at the end

    r_norm5 = (r_norm3 )**(5/3)               # |r|**5
    m_dot_r = DX*m[0] + DY*m[1] + DZ*m[2]     # m . r
    Fact1 = 1./(4*np.pi)*3*m_dot_r/r_norm5
    Fact2 = 1./(4*np.pi)/r_norm3

    ui = np.asarray(Fact1*DX - Fact2*m[0])
    vi = np.asarray(Fact1*DY - Fact2*m[1])
    wi = np.asarray(Fact1*DZ - Fact2*m[2])

    # --- Singularity
    ui[bSing]=0
    vi[bSing]=0
    wi[bSing]=0
    
    return ui,vi,wi

def doublet_u_polar(rcp,zcp,m_z,z0=0): 
    """
    Velocity field induced by one doublet located at z0 on the z axis with vector intensity (0,0,mz)
         ur = 3 m_z /(4pi) r (z-z0) / |r|**5
         uz =   m_z /(4pi) (3 (z-z0)**2 / |r|**5 -1/|r|**3)
    Control points defined by polar coordinates `rcp` and `zcp`.
    """
    if np.any(rcp<0):
        raise Exception('Script meant for positive r')
    rcp=np.asarray(rcp)
    zcp=np.asarray(zcp)

    DZ = zcp-z0

    r_norm3 = (rcp**2 + DZ**2 )**(3/2) # |r|**3
    r_norm3=np.asarray(r_norm3)

    # --- Avoiding singularity by introducing a temporarily fake norm
    bSing= r_norm3<1e-16
    r_norm3[bSing]=1 # Temporary hack, replaced at the end

    r_norm5 = (r_norm3 )**(5/3)               # |r|**5

    ur = np.asarray( 3*rcp*m_z / (4*np.pi) * DZ / r_norm5                      )
    uz = np.asarray(       m_z / (4*np.pi) * (3 * DZ**2 / r_norm5 - 1/r_norm3) )

    # --- Singularity
    ur[bSing]= 0
    uz[bSing]= 0
    
    return ur,uz


def doublet_line_polar_u_num(rcp,zcp,dmz_dz,z0=0,zmax=1000,nQuad=100):
    """
    Velocity field induced by a doublet line (on the z axis) of intensity `dmz_dz`.
    Control points defined by polar coordinates `rcp` and `zcp`.
    The line goes from `z0` to `zmax`.
    Numerical integration is used with `nQuad` quadrature points.

    """
    rcp=np.asarray(rcp)
    zcp=np.asarray(zcp)

    zq = np.linspace(z0,zmax,nQuad)

    # --- Summation
    dz = zq[1]-zq[0]
    mz = dmz_dz * dz
    ur =  np.zeros(rcp.shape)
    uz =  np.zeros(rcp.shape)
    for z0 in zq: 
        dur, duz = doublet_u_polar(rcp,zcp,mz,z0=z0)
        ur += dur
        uz += duz

    return ur, uz

def doublet_line_polar_u(rcp,zcp,dmz_dz):
    """
    Velocity field induced by a semi-infinite doublet line (on the z axis) of intensity `dmz_dz`
    Control points defined by polar coordinates `rcp` and `zcp`.

     \int  1/(r^2 + (z-x)^2 )^(3/2) dx
     \int  1/(r^2 + (z-x)^2 )^(5/2) dx
    """
    if np.any(rcp<0):
        raise Exception('Script meant for positive r')
    r=np.asarray(rcp)
    z=np.asarray(zcp)

    # Vectorial "if" statements to isolate singular regions of the domain
    bZ0    = np.abs(z)<1e-16
    bR0    = np.abs(r)<1e-16
    bZ0R0  = np.logical_and(bZ0,bR0)
    bZ0Rp  = np.logical_and(bZ0, np.abs(r)>1e-16)
    bR0Zp  = np.logical_and(bR0, z>1e-16)
    bR0Zm  = np.logical_and(bR0, z<-1e-16)
    bOK    = np.logical_and(~bZ0,~bR0)

    uz=np.zeros(r.shape)
    ur=np.zeros(r.shape)

    norm2 = r**2+z**2
    uz[bOK]  = dmz_dz/(4*np.pi) * 1/r[bOK]**2 * ( z[bOK]**3/(norm2[bOK])**(3/2) - z[bOK]/(norm2[bOK])**(1/2) )
    uz[bZ0Rp] = 0
    uz[bR0Zm] = dmz_dz/(4*np.pi) * 1/norm2[bR0Zm]
    uz[bR0Zp] = dmz_dz/(4*np.pi) * 1/norm2[bR0Zp]
    ur[bOK]   =-dmz_dz/(4*np.pi) * r[bOK]          *  1/(norm2[bOK]  )**(3/2)
    ur[bZ0Rp] =-dmz_dz/(4*np.pi) * r[bZ0Rp]        *  1/(norm2[bZ0Rp])**(3/2)
    ur[bR0Zm] = 0
    ur[bR0Zp] = 0

    ur[bZ0R0] = 0
    uz[bZ0R0] = 0


    return ur, uz

def doublet_line_u(Xcp,Ycp,Zcp,dmz_dz,m=0):
    """ 
    Velocity field induced by a double line, of intensity [0,0,dmz/dz]
    The line is defined by the coordinates (x=mz,y=0,z)
    INPUTS:
       Xcp,Ycp,Zcp: vector or matrix of control points Cartesian Coordinates
       m          : tangent of the skew angle (0 for a straight line along z)
    """
#     if m==0:
#         vr, vpsi = np.sqrt(Xcp**2+Ycp**2), np.arctan2(Ycp,Xcp) # polar coords
#         ur,uz=doublet_line_polar_u(vr,Zcp,dmz_dz)
#         ux=ur*np.cos(vpsi)
#         uy=ur*np.sin(vpsi)
#     else:
    if True:
        ux =  np.zeros(Xcp.shape)
        uy =  np.zeros(Xcp.shape)
        uz =  np.zeros(Xcp.shape)
        #
        a = np.sqrt(Xcp**2+Ycp**2+Zcp**2)
        b = m*Xcp+Zcp
        c = np.sqrt(m**2+1)
        bAxis = np.abs(a*c-b)<1e-16
        bOK   = np.logical_not(bAxis)
        ux[bAxis]=0
        uy[bAxis]=0
        uz[bAxis]=0
        a=a[bOK]
        b=b[bOK]
        Xcp=Xcp[bOK]
        Ycp=Ycp[bOK]
        Zcp=Zcp[bOK]
        print('>>>')
        i30 = I30(a,b,c)
        i50 = I50(a,b,c)
        i51 = I51(a,b,c)
        i52 = I52(a,b,c)

        uz[bOK] =   dmz_dz/(4*np.pi)  * (   3 * Zcp**2 * i50 + 3*i52 - 6*Zcp * i51 - i30 )
        ux[bOK] = 3*dmz_dz/(4*np.pi)  * ( Xcp * Zcp    * i50 + m*i52 - (m*Zcp+Xcp) * i51 )
        uy[bOK] = 3*dmz_dz/(4*np.pi)  * ( Ycp * Zcp    * i50         -        Ycp  * i51 )

    return ux,uy,uz 


def doublet_line_u_num(Xcp,Ycp,Zcp,dmz_dz,m=0,zmax=1000,nQuad=100):
    """ 
    Velocity field induced by a double line, of intensity [0,0,dmz/dz]
    The line is defined by the coordinates (x=mz,y=0,z)

    INPUTS:
       Xcp,Ycp,Zcp: vector or matrix of control points Cartesian Coordinates
       m          : tangent of the skew angle (0 for a straight line along z)
    """
    ## --- Summation
    zq = np.linspace(0,zmax,nQuad)
    dz = zq[1]-zq[0]
    mz = dmz_dz * dz
    ux =  np.zeros(Xcp.shape)
    uy =  np.zeros(Xcp.shape)
    uz =  np.zeros(Xcp.shape)
    for z0 in zq: 
        dux, duy, duz = doublet_u(Xcp,Ycp,Zcp,m=[0,0,mz],x0=[m*z0,0,z0]) 
        ux += dux
        uy += duy
        uz += duz
    return ux,uy,uz 

# --------------------------------------------------------------------------------}
# --- Integrals useful for vortex doublet
# --------------------------------------------------------------------------------{
# --- Main Results 
# a = sqrt(x^2 + y^2 + z^2)
# b = m x + z
# c = sqrt(m^2+1)
#  I30       = 1/(a (a c-b))   # <<< NOT TILDE
#  I50       = (2 a c - b)/(3 a^3 (a c - b)^2) # <<< NOT Tilde
#  I51       = 1/ (3 a (ac - b)^2)
#  I52       = 1/(3 c (a c - b)^2)

# --- NOTE: INTEGRALS BELOW ARE "TILDE" INTEGRALS (a->atilde, b-> btilde, c->1)
# atilde = a/c 
# btilde = b/c^2
# II30       = \int  1 / ( a^2  + z^2 - 2 z b )^(3/2) dz
# II30_indef = (-b + z)/((a^2 - b^2) sqrt(a^2 + z (-2 b + z))) 
# II30       = 1/(a (a-b))

# II50       = \int  1 / ( a^2  + z^2 - 2 z b )^(5/2) dz
# II50_indef = ((-b + z) (3 a^2 - b^2 - 4 b z + 2 z^2))/(3 (a^2 - b^2)^2 (a^2 - (2 b - z) z)^(3/2))
# II50_inf   = 2/(3 (a^2 - b^2)^2)
# II50_0     = b (-3 a^2 + b^2))/(3 a^3 (a^2 - b^2)^2)
# II50       = (2 a - b)/(3 a^3 (a - b)^2)

# II51       = \int  z / ( a^2  + z^2 - 2 z b )^(5/2) dz
# II51_indef = (-a^4 - a^2 b (b - 3 z) + b z (3 b^2 - 6 b z + 2 z^2))/(3 (a^2 - b^2)^2 (a^2 + z (-2 b + z))^(3/2))
# II51_inf   = (2 b)/(3 (a^2 - b^2)^2)
# II51_0     = (-a^2 - b^2)/(3 a (a^2 - b^2)^2)
# II51       = 1/ (3 a (a - b)^2)

# II52       = \int  z^2 / ( a^2  + z^2 - 2 z b )^(5/2) dz
# II52_indef = (-2 a^4 b - b^2 (3 b - z) z^2 + a^2 z (6 b^2 - 3 b z + z^2))/(3 (a^2 - b^2)^2 (a^2 - (2 b - z) z)^(3/2))
# II52_inf   = (a^2 + b^2)/(3 (a^2 - b^2)^2)
# II52_0     = (-2 a b)/(3 (a^2 - b^2)^2)
# II52       = 1/(3 (a - b)^2)

# --- Terms for  atilde=btilde
# I30_axis       = \int  1 / ( a - z )^(3) 
# I30_axis_indef = 1/(2 (a - z)^2)
# I30_axis       = -1/(2a^2)

# I50_axis       = \int [ 1 / ( a - z )^(5) 
# I50_axis_indef = 1/(4 (a - z)^4)
# I50_axis       = - 1/(4 (a)^4

# I51_axis       = \int [ z / ( a - z )^(5) 
# I51_axis_indef = -(a - 4 z)/(12 (a - z)^4)
# I51_axis       =  1/(12 a^3)

# I52_axis       = \int  z^2 / ( a - z )^(5) 
# I52_axis_indef = (a^2 - 4 a z + 6 z^2)/(12 (a - z)^4)
# I52_axis       =  1/(12 a^2)

def I30(a,b,c):
#     if a==b/c:
#         return -1/(2*c*a**2)
    return 1/(a*(a*c-b))

def I50(a,b,c):
#     if a==b/c:
#         return -1/(4*c*a**4)
    return (2*a*c-b)/(3*a**3*(a*c-b)**2)

def I51(a,b,c):
#     if a==b/c:
#         return 1/(12*c**2*a**3)
    return 1/(3*a*(a*c-b)**2)

def I52(a,b,c):
#     if a==b/c:
#         return -1/(12*c**3*a**2)
    return 1/(3*c*(a*c-b)**2)

def I30_num(x,y,z,m,nQuad=50000,zmax=5000):
    zp = np.linspace(0,zmax,nQuad)
    r2 = (x-m*zp)**2 + y**2 + (z-zp)**2
    dI = 1/ r2**(3/2)
    return np.trapz(dI, zp)

def I50_num(x,y,z,m,nQuad=50000,zmax=5000):
    zp = np.linspace(0,zmax,nQuad)
    r2 = (x-m*zp)**2 + y**2 + (z-zp)**2
    dI = 1/ r2**(5/2)
    return np.trapz(dI, zp)

def I51_num(x,y,z,m,nQuad=50000,zmax=5000):
    zp = np.linspace(0,zmax,nQuad)
    r2 = (x-m*zp)**2 + y**2 + (z-zp)**2
    dI = zp/ r2**(5/2)
    return np.trapz(dI, zp)

def I52_num(x,y,z,m,nQuad=50000,zmax=5000):
    zp = np.linspace(0,zmax,nQuad)
    r2 = (x-m*zp)**2 + y**2 + (z-zp)**2
    dI = zp**2/ r2**(5/2)
    return np.trapz(dI, zp)


class TestDoublet(unittest.TestCase):

    def test_Integrals(self):
        # --- random point, z>0
        x = 3
        y = 1
        z = 1.0
        m = 0.3
        a = np.sqrt(x**2+y**2+z**2)
        b = m*x+z
        c = np.sqrt(m**2+1)
        np.testing.assert_almost_equal(I30(a,b,c), I30_num(x,y,z,m), 5)
        np.testing.assert_almost_equal(I50(a,b,c), I50_num(x,y,z,m), 5)
        np.testing.assert_almost_equal(I51(a,b,c), I51_num(x,y,z,m), 5)
        np.testing.assert_almost_equal(I52(a,b,c), I52_num(x,y,z,m), 5)

        # --- random point, z<0
        x = 1
        y = 2
        z = -0.5
        m = 0.3
        a = np.sqrt(x**2+y**2+z**2)
        b = m*x+z
        c = np.sqrt(m**2+1)
        np.testing.assert_almost_equal(I30(a,b,c), I30_num(x,y,z,m), 5)
        np.testing.assert_almost_equal(I50(a,b,c), I50_num(x,y,z,m), 5)
        np.testing.assert_almost_equal(I51(a,b,c), I51_num(x,y,z,m), 5)
        np.testing.assert_almost_equal(I52(a,b,c), I52_num(x,y,z,m), 5)

        # --- On Axis, z<0 no need for singularity
        y = 0
        z = -1
        m = 0.3
        x = z*m
        a = np.sqrt(x**2+y**2+z**2)
        b = m*x+z
        c = np.sqrt(m**2+1)
        np.testing.assert_almost_equal(I30(a,b,c), I30_num(x,y,z,m), 2)
        np.testing.assert_almost_equal(I50(a,b,c), I50_num(x,y,z,m), 2)
        np.testing.assert_almost_equal(I51(a,b,c), I51_num(x,y,z,m), 2)
        np.testing.assert_almost_equal(I52(a,b,c), I52_num(x,y,z,m), 2)

    def test_Doublet_example(self):
        # ---- Singularity check
        u=doublet_u(0,0,0,[0,0,100])
        np.testing.assert_almost_equal(u,(0,0,0))
        # --- Random point check
        u     = doublet_u(1,2,3,[700,-800,900],x0=[-4,-5,6])
        u_ref = (-0.16496, -0.043617, -0.039940)
        np.testing.assert_almost_equal(u,u_ref,5)

    def test_Doublet_polar(self):
        # ---- Singularity check
        u=doublet_u_polar(0,0,0,100)
        np.testing.assert_almost_equal(u,(0,0))
        # ---- Random points check
        u     = doublet_u_polar(1,  3,     100,  z0=6)
        u_ref = doublet_u      (1,0,3,[0,0,100], x0=[0,0,6])
        np.testing.assert_almost_equal(u,(u_ref[0],u_ref[2]))
        u     = doublet_u_polar( 1,  -3,     100,  z0=6)
        u_ref = doublet_u      ( 1,0,-3,[0,0,100], x0=[0,0,6])
        np.testing.assert_almost_equal(u,(u_ref[0],u_ref[2]))

        u     = doublet_u_polar( 0.011,  -0.326,  100    ,  z0=0)
        u_ref = doublet_u      ( 0.011,0,-0.326,[0,0,100], x0=[0,0,0])
        np.testing.assert_almost_equal(u,(u_ref[0],u_ref[2]))

    def test_Doublet_line(self):
        gamma_t = -10
        R       = 10
        dmzdz   = np.pi * R**2 * gamma_t

        # --- Singularity 0,0 (should be infinity)
        u=doublet_line_polar_u  (0,0,dmzdz)
        np.testing.assert_almost_equal(u,(0,0))

        # --- Formula on the z axis from analytical and numerical integration
        zcp = np.linspace(-10,-1,50)*R
        rcp = zcp*0 
        urn,uzn=doublet_line_polar_u_num(rcp,zcp,dmzdz,0,1000,10000)
        urt,uzt=doublet_line_polar_u    (rcp,zcp,dmzdz)

        # ---- Plot
        #uzc = gamma_t/2*(1+zcp/np.sqrt(zcp**2 + R**2))
        #import matplotlib.pyplot as plt
        #fig,ax = plt.subplots(1,1)
        #ax.plot(zcp/R, urt    , label='urt')
        #ax.plot(zcp/R, urn ,'--'   , label='urn')
        #ax.plot(zcp/R, uzt    , label='uzt')
        #ax.plot(zcp/R, uzn ,'--'   , label='uzn')
        #print(zcp)
        #ax.plot(zcp/R, uzc ,'k.'   , label='uzn')
        #ax.set_xlabel('')
        #ax.set_ylabel('')
        #ax.legend()
        #plt.show()

        np.testing.assert_almost_equal(urn,urt,1)
        np.testing.assert_almost_equal(uzn,uzt,1)


if __name__ == "__main__":
#     TestDoublet().test_Doublet_polar()
    TestDoublet().test_Integrals()
#     unittest.main()
