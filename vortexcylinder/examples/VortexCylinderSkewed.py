"""
References:
    [1] E. Branlard - Wind Turbine Aerodynamics and Vorticity Based Method, Springer, 2017
    [2] E. Branlard, M. Gaunaa - Cylindrical vortex wake model: skewed cylinder, application to yawed or tilted rotors - Wind Energy, 2015

Coordinate systems
   c coordinate system used in see [2], rotor in plane z_c=0
   w wind coordinate system where z_w is the wind direction
   theta_yaw : yaw angle, positive around y, pointing upward

   x_c =  x_w cost + z_w sint
   y_c =  y_w
   z_c = -x_w sint + z_w cost

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
from vortexcylinder.VortexCylinderSkewed import skewedcylinder_tang_semi_inf_u
from vortexcylinder.VortexRing import ring_u
try:
    from pybra.colors import darkrainbow
    from pybra.colors import manual_colorbar
    from pybra.curves import streamQuiver
    from pybra.lic    import lic
    from pybra.tictoc import Timer
    cmap = darkrainbow
except:
    cmap = plt.cm.get_cmap("jet")
#     raise Exception('This script requires the package `pybra` from https://github.com/ebranlard/pybra/')
# for nD in [0,2,4]:
# for nD in [0,2,4]:





# --- Parameters
bWindCoord =True
bRootVortex =True
R         = 1
r_hub     = 0.1*R
CLIM=[0.3,1.1]
#     CLIM=[0.0,0.18]
LIM       = [-2,2] # 
nx        = 100    # Number of points for velocity evaluation
CT        = 0.6
theta_yaw = 30*np.pi/180   # rad
U0        = 1

# Number of cylinders per radius
n_radial  = 1 # 1: tip vortex, 2: tip&root, n: "continuous"

# --- Derived params
chi= theta_yaw*(1+0.3*(1-np.sqrt(1-CT))) # rad
if CT==0.4:
    gamma_t = -0.21341 # CT=0.4
elif CT==0.6:
    gamma_t = -0.40 # 
else:
    gamma_t = -0.60414 # CT=0.95
ny = nx
m  = np.tan(chi) 


for nD in [0,4]:
    z0_w      = nD*2*R #Plane

    def Tw2c(x_w,y_w,z_w):
        if bWindCoord:
            x_c =  x_w * np.cos(theta_yaw) + z_w * np.sin(theta_yaw)
            y_c =  y_w
            z_c = -x_w * np.sin(theta_yaw) + z_w * np.cos(theta_yaw)
        else:
            x_c,y_c,z_c = x_w,y_w,z_w
        return x_c,y_c,z_c
    def Tc2w(x_c,y_c,z_c):
        if bWindCoord:
            x_w =  x_c * np.cos(theta_yaw) - z_c * np.sin(theta_yaw)
            y_w =  y_c
            z_w =  x_c * np.sin(theta_yaw) + z_c * np.cos(theta_yaw)
        else:
            x_w,y_w,z_w = x_c,y_c,z_c
        return x_w, y_w, z_w

    # --- Flow field and speed
    x_w = np.linspace(LIM[0],LIM[1],nx)
    y_w = np.linspace(LIM[0],LIM[1],ny)
    [X_w,Y_w]=np.meshgrid(x_w,y_w)
    Z_w=X_w*0+z0_w
    X_c,Y_c,Z_c = Tw2c(X_w,Y_w,Z_w) 

    with Timer('Computing for D={} - cylinder'.format(nD)):
        ux_c,uy_c,uz_c,_,_,_,_=skewedcylinder_tang_semi_inf_u(X_c,Y_c,Z_c,gamma_t,R,m)
        if bRootVortex:
            ux_c_root,uy_c_root,uz_c_root,_,_,_,_=skewedcylinder_tang_semi_inf_u(X_c,Y_c,Z_c,-gamma_t,r_hub,m)
            ux_c += ux_c_root
            uy_c += uy_c_root
            uz_c += uz_c_root


        uz_c=uz_c+U0*np.cos(theta_yaw) # Adding free wind
        ux_c=ux_c+U0*np.sin(theta_yaw)
    ux,uy,uz = Tc2w(ux_c,uy_c,uz_c)

    # --- Flow field from many rings
    with Timer('Computing for D={} - rings'.format(nD)):
        NRings=500
        ZRings=16.1*R
        Zr_c = np.linspace(0,ZRings,NRings)
        dz   = Zr_c[1]-Zr_c[0]
        Gamma_Rings = gamma_t*dz
        Xr_c = m*Zr_c
        ux_r_c = np.zeros(ux_c.shape)
        uy_r_c = np.zeros(ux_c.shape)
        uz_r_c = np.zeros(ux_c.shape)
        for iR,(xr,zr) in enumerate(zip(Xr_c,Zr_c)):
            uxx,uyy,uzz = ring_u(X_c-xr,Y_c,Z_c-zr,Gamma_Rings,R,cartesianOut=True)
            ux_r_c +=  uxx
            uy_r_c +=  uyy
            uz_r_c +=  uzz 
        if bRootVortex:
            for iR,(xr,zr) in enumerate(zip(Xr_c,Zr_c)):
                uxx,uyy,uzz = ring_u(X_c-xr,Y_c,Z_c-zr,-Gamma_Rings,r_hub,cartesianOut=True)
                ux_r_c +=  uxx
                uy_r_c +=  uyy
                uz_r_c +=  uzz 
        uz_r_c=uz_r_c+U0*np.cos(theta_yaw) # Adding free wind
        ux_r_c=ux_r_c+U0*np.sin(theta_yaw)
    ux_r,uy_r,uz_r = Tc2w(ux_r_c,uy_r_c,uz_r_c)


    # --- Removing singularity
    # TODO
    # bTip  = abs(sqrt((X-m*z0).^2+Y.^2)-R)<epsilon;
    # bRoot = sqrt((X-m*z0).^2+Y.^2)<epsilon  ;
    # b=bTip | bRoot;
    # % b=bTip;

    def plot(ux,uy,uz,label=''):
        Speed=np.sqrt(uz**2)
        # Temporary HACK until singularity is removed
        Speed[Speed>CLIM[1]] = CLIM[1]
        Speed[Speed<CLIM[0]] = CLIM[0]

        # rotor projection
        vpsi=np.linspace(0,2*np.pi,50)
        xc_w=R*np.cos(vpsi)*np.cos(theta_yaw)
        yc_w=R*np.sin(vpsi)

        dpi=300
        fig=plt.figure()
        ax=fig.add_subplot(111)
        im=ax.contourf(X_w,Y_w,Speed,30,cmap=cmap)
        ax.plot(xc_w,yc_w,'k--')
        cb=fig.colorbar(im)
        cb.set_clim(CLIM)
        sp=ax.streamplot(x_w,y_w,ux,uy,color='k',linewidth=0.7,density=2)

        ax.set_xlim(LIM)
        ax.set_ylim(LIM)
        ax.set_xlabel('x/R [-]')
        ax.set_ylabel('y/R [-]')
        ax.set_title('z = {}D{}'.format(int(z0_w/(2*R)),label))
        fig.savefig("VC_yaw{:02d}_CT{:03d}_{:d}D{}.png".format(int(np.round(theta_yaw*180/np.pi)),int(CT*100),int(z0_w/(2*R)),label),dpi=dpi)


    plot(ux  ,uy  ,uz  ,' cylinder')
    plot(ux_r,uy_r,uz_r,' rings'   )

plt.show()


# Colorbar
# cax = divider.append_axes("right", size="5%", pad=0.20)
# manual_colorbar(fig,cmap,cax=cax)
# Streamlines
# yseed=np.linspace(-0.88,0.88,7)
# start=np.array([yseed*0,yseed])
# sp=ax.streamplot(x_w,y_w,ux,uy,color='k',linewidth=0.7,density=1)
# # sp=ax.streamplot(x_w,y_w,ux,uy,color='k',start_points=start.T,linewidth=0.7,density=30,arrowstyle='-')
# # qv=streamQuiver(ax,sp,spacing=0.8,scale=40,angles='xy')
# # qv=streamQuiver(ax,sp,n=[1,5,5,5,5,5,1],scale=40,angles='xy')
# ax.set_xlabel('x/R [-]')
# ax.set_ylabel('y/R [-]')
# 

# plt.show()


