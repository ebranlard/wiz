#--- Legacy python 2.7
from __future__ import division
from __future__ import print_function
# --- General
import unittest
import numpy as np
# --- Local
try:
    from .VortexCylinder import cylinder_tang_semi_inf_u, cylinders_tang_semi_inf_u
    from .VortexCylinderSkewed import svc_tang_u, svc_longi_u, svc_root_u
except:
    try:
        from vortexcylinder.VortexCylinder import cylinder_tang_semi_inf_u, cylinders_tang_semi_inf_u
        from vortexcylinder.VortexCylinderSkewed import svc_tang_u, svc_longi_u, svc_root_u
    except:
        from VortexCylinder import cylinder_tang_semi_inf_u, cylinders_tang_semi_inf_u
        from VortexCylinderSkewed import svc_tang_u, svc_longi_u, svc_root_u
vc_tang_u=cylinder_tang_semi_inf_u


# --------------------------------------------------------------------------------}
# --- Helper functions for geometry 
# --------------------------------------------------------------------------------{
def RotMat_AxisAngle(u,theta):
    """ Returns the rotation matrix for a rotation around an axis u, with an angle theta """
    R=np.zeros((3,3))
    ux,uy,uz=u
    c,s=np.cos(theta),np.sin(theta)
    R[0,0]=ux**2*(1-c)+c    ; R[0,1]=ux*uy*(1-c)-uz*s; R[0,2]=ux*uz*(1-c)+uy*s;
    R[1,0]=uy*ux*(1-c)+uz*s ; R[1,1]=uy**2*(1-c)+c   ; R[1,2]=uy*uz*(1-c)-ux*s
    R[2,0]=uz*ux*(1-c)-uy*s ; R[2,1]=uz*uy*(1-c)+ux*s; R[2,2]=uz**2*(1-c)+c;
    return R

def orth_vect(u):
    """ Given one vector, returns a 3x1 orthonormal vector to it"""
    u=(u/np.linalg.norm(u)).ravel()
    if abs(u[0])>=1/np.sqrt(3):
        v = np.array([[-u[1]],[u[0]] ,[0]]    )
    elif abs(u[1])>=1/np.sqrt(3):
        v = np.array([[0]    ,[-u[2]],[u[1]]] )
    elif abs(u[2])>=1/np.sqrt(3):
        v = np.array([[u[2]] ,[0]    ,[-u[0]]])
    else:
        raise Exception('Cannot find orthogonal vector to a zero vector: {} '.format(u))
    return v/np.linalg.norm(v)

def transform_T(T_a2b,Xb,Yb,Zb):
    Xa=T_a2b[0,0]*Xb+T_a2b[1,0]*Yb+T_a2b[2,0]*Zb
    Ya=T_a2b[0,1]*Xb+T_a2b[1,1]*Yb+T_a2b[2,1]*Zb
    Za=T_a2b[0,2]*Xb+T_a2b[1,2]*Yb+T_a2b[2,2]*Zb
    return Xa,Ya,Za
def transform(T_a2b,Xa,Ya,Za):
    Xb=T_a2b[0,0]*Xa+T_a2b[0,1]*Ya+T_a2b[0,2]*Za
    Yb=T_a2b[1,0]*Xa+T_a2b[1,1]*Ya+T_a2b[1,2]*Za
    Zb=T_a2b[2,0]*Xa+T_a2b[2,1]*Ya+T_a2b[2,2]*Za
    return Xb,Yb,Zb


# --------------------------------------------------------------------------------}
# --- Main Class 
# --------------------------------------------------------------------------------{
class WindTurbine:
    def __init__(self,r_hub=[0,0,0],e_shaft_yaw0=[0,0,1],e_vert=[0,1,0],U0=[0,0,10],Ct=0.6,lambda_r=7,R=65,name=''):
        """ 
         - e_vert: verticla vector, about which yawing is done
        """
        self.set_yaw0_coord(e_shaft_yaw0,e_vert)
        self.update_position(r_hub)
        self.update_wind(U0)
        self.name=name
        self.R=R

    def set_yaw0_coord(self,e_shaft_yaw0,e_vert):
        self.e_shaft_g = np.asarray(e_shaft_yaw0).ravel().reshape(3,1)
        self.e_shaft_g0= self.e_shaft_g
        self.e_vert_g  = np.asarray(e_vert).ravel().reshape(3,1)
        v_horz0 = self.e_shaft_g -np.dot(self.e_shaft_g.T,self.e_vert_g)
        self.e_horz_g = v_horz0/np.linalg.norm(v_horz0)
        self.e_FAST_z = self.e_vert_g
        self.e_FAST_x = self.e_horz_g
        self.e_FAST_y = np.cross(self.e_FAST_z.T,self.e_FAST_x.T).T
        # Transformation matrix from global to FAST coordinate system
        self.T_F2g   = np.column_stack((self.e_FAST_x,self.e_FAST_y,self.e_FAST_z))
        # Transformation matrix from cylinder coordinate system to wind turbine
        # TODO: handle tilt
        e_c_x=np.cross(self.e_vert_g.T,self.e_shaft_g.T).T
        self.T_c2wt = np.column_stack((e_c_x,self.e_vert_g,self.e_shaft_g))
        self.update_yaw_pos(0)

    def update_yaw_pos(self,yaw):
        #self.r_hub=r_hub
        self.yaw_pos=yaw
        self.T_wt2g=RotMat_AxisAngle(self.e_vert_g,yaw)
        # Rotating the shaft vector so that its coordinate follow the new yaw position
        self.e_shaft_g=np.dot(self.T_wt2g , self.e_shaft_g0)

    def update_position(self,r_hub):
        self.r_hub=np.asarray(r_hub).ravel().reshape(3,1)

    def update_wind(self,U0_g):
        self.U0_g = np.asarray(U0_g).ravel().reshape(3,1)

    def update_loading(self,r=None,Ct=None,Gamma=None,Omega=None):
        U0=np.linalg.norm(self.U0_g)
        if Ct is not None:
            gamma_t,gamma_l,Gamma_r,misc=WakeVorticityFromCt(r,Ct,self.R,U0,Omega)
        elif Gamma is not None:
            gamma_t,gamma_l,Gamma_r,misc=WakeVorticityFromGamma(r,Ct,self.R,U0,Omega)
        else:
            raise Exception('Unknown loading spec')

    @property
    def yaw_wind(self):
        """ NOTE: this is wind angle not wind direction, measured with same convention as yaw:
            - around the axis e_vert
        """
        u_horz = self.U0_g - np.dot(self.U0_g.T,self.e_vert_g)*self.e_vert_g
        e_w    = u_horz/np.linalg.norm(u_horz)
        sign = np.sign  ( np.dot(np.cross(self.e_horz_g.T, self.U0_g.T),self.e_vert_g) )
        if sign==0:
            yaw_wind = np.arccos(np.dot(e_w.T,self.e_horz_g))
        else:
            yaw_wind = sign * np.arccos(np.dot(e_w.T,self.e_horz_g))
        return yaw_wind.ravel()[0]

    @property
    def yaw_error(self):
        v_horz = self.e_shaft_g -np.dot(self.e_shaft_g.T,self.e_vert_g)
        e_horz = v_horz/np.linalg.norm(v_horz)
        u_skew  = self.U0_g-np.dot(self.U0_g.T,e_horz)*e_horz-np.dot(self.U0_g.T,self.e_vert_g)*self.e_vert_g
        e_w=self.U0_g/np.linalg.norm(self.U0_g)
        yaw_sign=np.sign  ( np.dot(np.cross(e_w.T,self.e_shaft_g.T),self.e_vert_g) )
        yaw_error=yaw_sign * np.arccos(np.dot(e_w.T,self.e_shaft_g))
#         U0_wt = np.dot( self.T_wt2g.T , self.U0_g)
#         e_skew=u_skew/np.linalg.norm(u_skew)
#         print('e_skew',e_skew.T)
        return yaw_error.ravel()[0]

    #@property
    #def tilt(self):
    #    shaft_vert=np.dot(self.e_shaft.T,self.e_vert)
    #    return np.arcsin(shaft_vert)*180/np.pi

    #@property
    #def yaw(self):
    #    if self.tilt==0:
    #        u_skew=self.U0-np.dot(self.U0,self.e_shaft)
#   #          e_skew=self.

    #    else:
    #        raise Exception('Tilt and yaw not supported yet')
    #    return np.arcsin(shaft_vert)*180/np.pi

    def rotor_disk_points(self):
        nP=10
        points=np.zeros((3,nP))
        theta=np.linspace(0,2*np.pi,nP)
        e_r = self.R*orth_vect(self.e_shaft_g)
        for i,t in enumerate(theta):
            T=RotMat_AxisAngle(self.e_shaft_g,t)
            points[:,i]= np.dot(T,e_r).ravel()
        return points


    
    def compute_u(self,Xg,Yg,Zg,only_ind=False):
        # Transformtion from cylinder to global
        T_c2g=np.dot(self.T_wt2g,self.T_c2wt)
        Xc,Yc,Zc = transform_T(T_c2g, Xg,Yg,Zg)
        # Detecting whether our vertical convention match, and define chi
        e_vert_c = np.dot(T_c2g.T , self.e_vert_g)
        self.chi= np.sign(e_vert_c.ravel()[1])* (self.yaw_wind-self.yaw_pos)
        print(self.e_vert_g)
        print(e_vert_c)
        print('chi',self.chi*180/np.pi)
        gamma_t=-6 # TODO TODO
        if np.abs(self.chi)>1e-7:
            m=np.tan(self.chi)
            uxc,uyc,uzc = svc_tang_u(Xc,Yc,Zc,gamma_t=gamma_t,R=self.R,m=m,polar_out=False)
        else:
            uxc,uyc,uzc = vc_tang_u (Xc,Yc,Zc,gamma_t=gamma_t,R=self.R    ,polar_out=False)

        # Back to global
        uxg,uyg,uzg = transform(T_c2g, uxc, uyc, uzc)
        # Add free stream if requested
        if not only_ind:
            uxg += self.U0_g[0]
            uyg += self.U0_g[1]
            uzg += self.U0_g[2]
        return uxg,uyg,uzg


    def __repr__(self):
        s ='class WindTurbine({}), with attributes:\n'.format(self.name)
        s+=' - r_hub  : {}\n'.format(self.r_hub)
        s+=' - R      : {}\n'.format(self.R)
        s+=' - e_shaft: {}\n'.format(np.round(self.e_shaft_g.T,4))
        s+=' - U0     : {}\n'.format(self.U0_g.T)
        s+=' - wd     : {} deg\n'.format(self.yaw_wind*180/np.pi)
        s+=' - yaw_pos: {} deg\n'.format(self.yaw_pos  *180/np.pi)
        s+=' - yaw_err: {} deg\n'.format(self.yaw_error*180/np.pi)
        s+=' - yaw_err: {} deg\n'.format((-self.yaw_wind+self.yaw_pos)*180/np.pi)
        s+=' - T_F2g  : \n{}\n'.format(np.round(self.T_F2g ,4))
        s+=' - T_c2wt : \n{}\n'.format(np.round(self.T_c2wt,4))
        s+=' - T_wt2g : \n{}\n'.format(np.round(self.T_wt2g,4))
        s+=' - T_c2g  : \n{}\n'.format(np.round(np.dot(self.T_wt2g,self.T_c2wt),4))
        #s+=' - tilt   :  {}\n'.format(self.tilt)
        #s+=' - e_shaft: {}\n'.format(self.e_shaft)
        #s+=' - e_vert: {}\n'.format(self.e_vert)
        return s


# --------------------------------------------------------------------------------}
# --- TEST 
# --------------------------------------------------------------------------------{
class TestWindTurbine(unittest.TestCase):

    def test_WT_main(self):
        # TODO
        import matplotlib.pyplot as plt
        # --- Test in Cylinder coord
        R=65
        h_hub=0

        bSwirl    = True
        U0        = 10
        r_bar_cut = 0.11
        Lambda    = 6
        CT0       = 0.95
        nCyl      = 50
        Omega = Lambda * U0 / R

        if nCyl==1:
            vr_bar = np.array([0.995])
        else:
            vr_bar = np.linspace(0.005,0.995,nCyl)
        # --- Cutting CT close to the root
        Ct_AD = Ct_const_cutoff(CT0,r_bar_cut,vr_bar)

        WT=WindTurbine(R=R,e_shaft_yaw0=[0,0,1],e_vert=[0,1,0])
        #yaw=np.linspace(0,2*np.pi,9)



        #for y in yaw:
        #    WT.update_yaw_pos( y - np.pi/6)
        #    WT.update_wind([10*np.sin(y),0,10*np.cos(y)])
        #    WT.update_loading(r=vr_bar*R,Ct=Ct_AD)

        #    # --- Flow field and speed
        #    nx=40
        #    nz=41
        #    x = np.linspace(-4*R,4*R,nx)
        #    z = np.linspace(-4*R,4*R,nz)
        #    [X,Z]=np.meshgrid(x,z)
        #    Y=Z*0+h_hub
        #    ux,uy,uz = WT.compute_u(X,Y,Z)

        #    fig=plt.figure()
        #    ax=fig.add_subplot(111)
        #    Speed=np.sqrt(uz**2+ux**2)
        #    im=ax.contourf(Z/R,X/R,Speed,levels=30)
        #    Rotor=WT.rotor_disk_points()
        #    ax.plot(Rotor[2,:]/R,Rotor[0,:]/R,'k--')
        #    cb=fig.colorbar(im)
        #    sp=ax.streamplot(z/R,x/R,uz.T,ux.T,color='k',linewidth=0.7,density=2)
        #    ax.set_xlabel('z/R [-]')
        #    ax.set_ylabel('x/R [-]')
        #    deg=180/np.pi
        #    ax.set_title('yaw_pos = {:.1f} - yaw_wind={:.1f} - chi={:.1f} - yaw_err={:.1f}'.format(WT.yaw_pos*deg,WT.yaw_wind*deg,WT.chi*deg,WT.yaw_error*deg))


        #    plt.show()


        # --- Test in meteorological coord
#        R=65
#        h_hub=0
#        WT=WindTurbine(R=R, e_shaft_yaw0=[1,0,0],e_vert=[0,0,1])
#
#        yaw=np.linspace(-np.pi/2,np.pi/2,9)
#
#        WT.update_wind([10,0,0])
#        for y in yaw:
#            WT.update_yaw_pos(y)
#            # --- Flow field and speed
#            nx=40
#            ny=41
#            x = np.linspace(-4*R,4*R,nx)
#            y = np.linspace(-4*R,4*R,ny)
#            [X,Y]=np.meshgrid(x,y)
#            Z=Y*0+h_hub
#            ux,uy,uz = WT.compute_u(X,Y,Z)
#
#            fig=plt.figure()
#            ax=fig.add_subplot(111)
#            Speed=np.sqrt(uy**2+ux**2)
#            im=ax.contourf(X/R,Y/R,Speed,levels=30)
#            Rotor=WT.rotor_disk_points()
#            ax.plot(Rotor[0,:]/R,Rotor[1,:]/R,'k--')
#            cb=fig.colorbar(im)
#            sp=ax.streamplot(x/R,y/R,ux,uy,color='k',linewidth=0.7,density=2)
#            ax.set_xlabel('x/R [-]')
#            ax.set_ylabel('y/R [-]')
#            deg=180/np.pi
#            ax.set_title('yaw_pos = {:.1f} - yaw_wind={:.1f} - chi={:.1f} - yaw_err={:.1f}'.format(WT.yaw_pos*deg,WT.yaw_wind*deg,WT.chi*deg,WT.yaw_error*deg))
#            plt.axis('equal')
#
#
#            plt.show()
#

if __name__ == "__main__":
    unittest.main()
