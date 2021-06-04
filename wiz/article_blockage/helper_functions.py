import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.io
from scipy import interpolate
import scipy.interpolate as si
import pickle

from wiz.WindFarm  import WindFarm, gridLayout
from wiz.plotting_functions  import get_cmap
from wiz.Solver import *
from wiz.WindTurbine import *

# --- Local
try:
    from welib.tools.stats import rsquare
    from welib.tools.colors import *
    from welib.tools.curves import streamQuiver
    from welib.tools.clean_exceptions import *
except:
    def fColrs(i,n=3):
        cmap = matplotlib.cm.get_cmap('viridis')
        CMAP      = [(cmap(v)[0],cmap(v)[1],cmap(v)[2],1.0  ) for v in np.linspace(0,1,n+1)]
        CMAP.reverse()
        return CMAP[i]
    #print('This script requires the package `welib` from https://github.com/ebranlard/welib/')

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def getModelsCMAP(alpha=0.3):
    cmap = matplotlib.cm.get_cmap('viridis')
    n=4 # len(Models)
    CMAPTrans = [(cmap(v)[0],cmap(v)[1],cmap(v)[2],alpha) for v in np.linspace(0,1,n+1)]
    CMAP      = [(cmap(v)[0],cmap(v)[1],cmap(v)[2],1.0  ) for v in np.linspace(0,1,n+1)]

    Models = []
    Models.append({'ls':'-' , 'color':'k'    , 'label':'Actuator disk'  ,'name':'AD','mark':'.'})
    Models.append({'ls':'-' , 'color':CMAP[0], 'label':'Vortex cylinder'  ,'name':'VC', 'mark':'d'})

    return Models, CMAPTrans, CMAP


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx], idx


def interp2d_pairs(*args,**kwargs):
    """ Same interface as interp2d but the returned interpolant will evaluate its inputs as pairs of values.
    """
    # Internal function, that evaluates pairs of values, output has the same shape as input
    def interpolant(x,y,f):
        x,y = np.asarray(x), np.asarray(y)
        return (si.dfitpack.bispeu(f.tck[0], f.tck[1], f.tck[2], f.tck[3], f.tck[4], x.ravel(), y.ravel())[0]).reshape(x.shape)
    # Wrapping the scipy interp2 function to call out interpolant instead
    return lambda x,y: interpolant(x,y,si.interp2d(*args,**kwargs))

def gamma_CT_fun(x,*p):
    """ gamma/U0=f(CT), i.e. x=Ct
    INPUTS:
       x = CT value
       p : list of parameters of fitting function
    NOTE: returns gamma_t/U0! 
    """
    return -2*(p[0] * x + p[1] * x**2 + p[2] * x**3 + p[3] * x**4)

def relerrinduction(X,U1,U2,x1=-4.99, x2=-1):
    Xprobe  = np.linspace(x1,x2,100)
    U1probe = np.interp(Xprobe, X, U1)
    U2probe = np.interp(Xprobe, X, U2)
    U1probe=U1probe/U1[-1]
    U2probe=U2probe/U1[-1]
    relerr = np.mean(np.abs((U1probe-U2probe))/U1probe )*100
    return relerr

def r2induction(X,U1,U2):
    Xprobe  = np.linspace(-4.99,-1.0,100)
    U1probe = np.interp(Xprobe, X, U1)
    U2probe = np.interp(Xprobe, X, U2)
    R2,_=rsquare(U1probe, U2probe);
    return R2

def relerrinduction_mid(X,U1,U2):
    U1probe = np.interp(0, X, U1)
    U2probe = np.interp(0, X, U2)
    relerr = np.mean(np.abs((U1probe-U2probe))/U1probe )*100
    return relerr

def relerrinduction_25(X,U1,U2):
    U1probe = np.interp(-2.5, X, U1)
    U2probe = np.interp(-2.5, X, U2)
    relerr = np.mean(np.abs((U1probe-U2probe))/U1probe )*100
    return relerr
# --------------------------------------------------------------------------------}
# --- ProjCorr 
# --------------------------------------------------------------------------------{
class OutputProj():
    def __init__(self,CT,HH=9999,Lambda=9999,Horizontal=False,Yaw=None,suffix='_rev1_df01R_noGE'):
        filename='../data/ProjCorr/Layout1x1_CT0{:02d}{:s}.npz'.format(round(CT*100),suffix)
        print('>>> Filename',filename, 'HH',HH)

        # --- Plot params
        xrange = (-4,8)
        yrange = (-4.5,4.5)
        zrange = (0,2.5)
        spacing = 0.05

        d= np.load(filename)
        x1 = d['x1']
        y1 = d['y1']
        z1 = d['z1']
        u  = d['u']
        v  = d['v']
        w  = d['w']

        self.d=d

        if Horizontal:
            # Then we take a slice on z
            zslice = 0
            xx,yy = np.meshgrid(x1,y1,indexing='ij')
            z0=z1-HH
            k = np.argmin(np.abs(zslice-z0))
            self.x = np.sort(np.unique(np.round(yy.ravel(),3)))
            self.z = np.sort(np.unique(np.round(xx.ravel(),3)))
            self.U=w[:,:,k]
            self.V=v[:,:,k]
            self.W=u[:,:,k]
            self.X=yy
            self.Z=xx

        else:
            # Then we take a slice on y
            yslice = 0
            xx,zz = np.meshgrid(x1,(z1-HH),indexing='ij')
            j = np.argmin(np.abs(yslice-y1))
            self.y = np.sort(np.unique(np.round(zz.ravel(),3)))
            self.z = np.sort(np.unique(np.round(xx.ravel(),3)))
            self.U=w[:,j,:]
            self.V=v[:,j,:]
            self.W=u[:,j,:]
            self.Y=zz
            self.Z=xx
#         u = u[:,j,:]
#         w = w[:,j,:]

#         yslice = 0
#         xx,zz = np.meshgrid(2*x1,2*(z1-1),indexing='ij')
#         j = int((yslice-yrange[0])/spacing)

        self.O=self.U*0
        self.CT         = CT
        self.HH         = HH
        self.Lambda     = Lambda
        self.Horizontal = Horizontal
        self.Vertical   = not self.Horizontal
        self.Yaw        = Yaw

    def fVert(self,vy,vz):
        if self.Horizontal:
            return self.X,self.Z,self.U,self.W
        else:
            return self.Y,self.Z,self.V,self.W

    def setupInterpolants(self):
        print('Setting interpolants...')
        if self.Vertical:
            self._fU_Vert = interp2d_pairs(self.y, self.z, self.U, kind='cubic')
            self._fV_Vert = interp2d_pairs(self.y, self.z, self.V, kind='cubic')
            self._fW_Vert = interp2d_pairs(self.y, self.z, self.W, kind='cubic')
        else:
            self._fU_Horz = interp2d_pairs(self.x, self.z, self.U, kind='cubic')
            self._fV_Horz = interp2d_pairs(self.x, self.z, self.V, kind='cubic')
            self._fW_Horz = interp2d_pairs(self.x, self.z, self.W, kind='cubic')
        print('Done!')

    def fW(self,X,Y,Z):
        if self.Vertical:
            W=self._fW_Vert(Y,Z)
        else:
            W=self._fW_Horz(X,Z)
        U=W*0
        V=W*0
        return U,V,W



# --------------------------------------------------------------------------------}
# --- AD Lines For Farm
# --------------------------------------------------------------------------------{
class OutputADLines(dict):
    def __init__(self,filename=None):
        filename='../data/CFD_AD_Farm_lines_All'
#         filename='../data/CFD_AD_Farm_lines_2Spacing'
        d=load_obj(filename)
        super(OutputADLines, self).__init__(d)

    def getSim(self,S,Layout,CT,HH):
        if Layout=='1x1':
            S=0
        if int(S)!=S:
            raise Exception()
        base='d{:d}_{:s}_ct{:04.2f}_hh{:04.2f}'.format(int(S),Layout,CT,HH)
        shape=[int(x) for x in Layout.split('x')]
        ADFarm = self[base]
        WT_pos = ADFarm['xyz_wt']
        Cts    = ADFarm['ct_wt']
        Cts    = Cts.reshape(shape)
        X_mid  = ADFarm['xyz'][:,0,0]
        U_mid  = ADFarm['uvw'][:,0,0]
        if Layout=='1x1':
            pass
        else:
            X_top  = ADFarm['xyz'][:,0,1]
            U_top  = ADFarm['uvw'][:,0,1]
        return ADFarm, X_mid, U_mid

    def getSimTop(self,S,Layout,CT,HH):
        if Layout=='1x1':
            S=0
        if int(S)!=S:
            raise Exception()
        base='d{:d}_{:s}_ct{:04.2f}_hh{:04.2f}'.format(int(S),Layout,CT,HH)
        shape=[int(x) for x in Layout.split('x')]
        ADFarm = self[base]
        if Layout=='1x1':
            pass
        else:
            X_top  = ADFarm['xyz'][:,0,1]
            U_top  = ADFarm['uvw'][:,0,1]
        return ADFarm, X_top, U_top


    def get25(self,S,Layout,CT,HH):
        ADFarm, X_mid, U_mid = self.getSim(S,Layout,CT,HH)
        return np.interp(-2.5,X_mid,U_mid)

    def getSimCTs(self,S,Layout,CT,HH):
        if Layout=='1x1':
            S=0
        base='d{:d}_{:s}_ct{:04.2f}_hh{:04.2f}'.format(S,Layout,CT,HH)
        shape=[int(x) for x in Layout.split('x')]
        ADFarm = self[base]
        WT_pos = ADFarm['xyz_wt']
        Cts    = ADFarm['ct_wt'].reshape((shape[1]),shape[0])
        if Layout=='1x1':
            x=0
        else:
            x=np.unique(WT_pos[:,0])/S
        return  x, Cts


def ADFarm2VCFarm(ADFarm, name, gamma_t_Ct=None):
    # --- Parameters for this sript
    Model    = 'VC'         # Model used for flow computation ['VC','VCFF','VD','SS']
    no_wake  = False        # Removes wake - set to True when coupled to a wake model
    R        = 0.5          # Rotor radius [m]
    U0       = 1            # Free stream velocity [m/s]
    U0_g = [U0, 0, 0] 
    # AD variables
    WT_pos  = ADFarm['xyz_wt'] # All unints in diameters
    CTs     = ADFarm['ct_wt']
    # --- Creating wind farm (list of WT)
    WF = WindFarm(name=name)
    for r_hub in WT_pos:
        WF.append( WindTurbine(R=R, e_shaft_yaw0=[1,0,0], e_vert=[0,0,1], r_hub=np.array(r_hub) ) )
    #print(WF)
    Ground   = r_hub[2]<10         # Add ground effect

    # --- Setting WT specific values
    for WT,CT in zip(WF,CTs):
        WT.update_yaw_pos(0)   # Wind turbine yaw in [rad]
        WT.update_wind(U0_g)       # Free stream vector, local to that turbine [m/s]
        if gamma_t_Ct is not None:
            WT.update_loading(Ct=CT, gamma_t_Ct=gamma_t_Ct)  # Thrust coefficient, based on turbine local free stream [-]
        else:
            WT.update_loading(Ct=CT)  # Thrust coefficient, based on turbine local free stream [-]
    X_mid = ADFarm['xyz'][:,0,0]
    Y_mid = ADFarm['xyz'][:,1,0]
    Z_mid = ADFarm['xyz'][:,2,0]
    U_mid, _, _ = WF.velocity_field(X_mid, Y_mid, Z_mid, no_wake=no_wake, U0=U0_g, ground=Ground, Model=Model)
    if len(CTs)>1:
        X_top = ADFarm['xyz'][:,0,1]
        Y_top = ADFarm['xyz'][:,1,1]
        Z_top = ADFarm['xyz'][:,2,1]
        U_top, _, _ = WF.velocity_field(X_top, Y_top, Z_top, no_wake=no_wake, U0=U0_g, ground=Ground, Model=Model)
    else:
        U_top=U_mid*0

    return U_mid, U_top

def VCFarmFromFloris(CTs, HH, S, gamma_t_Ct=None,Model='VC'):
    # --- Parameters for this sript
    no_wake  = False        # Removes wake - set to True when coupled to a wake model
    R        = 0.5          # Rotor radius [m]
    U0       = 1            # Free stream velocity [m/s]
    U0_g = [U0, 0, 0] 
    # AD variables
    shape=CTs.shape
    nx=shape[0]
    ny=shape[1]
    yWT,zWT,xWT= gridLayout(nx, S, ny, S, hub_height=HH)
    Ground   = zWT[0]<10 # Add ground effect
    # --- Creating wind farm (list of WT)
    WF = WindFarm(name='')
    for (x,y,z) in zip(xWT,yWT,zWT):
        WF.append( WindTurbine(R=R, e_shaft_yaw0=[1,0,0], e_vert=[0,0,1], r_hub=np.array([x,y,z]), Ground=Ground, Model=Model ) )
    # --- Setting WT specific values
    for WT,CT in zip(WF,CTs.T.ravel()):
        WT.update_yaw_pos(0)   # Wind turbine yaw in [rad]
        WT.update_wind(U0_g)       # Free stream vector, local to that turbine [m/s]
        if gamma_t_Ct is not None:
            WT.update_loading(Ct=CT, gamma_t_Ct=gamma_t_Ct)  # Thrust coefficient, based on turbine local free stream [-]
        else:
            WT.update_loading(Ct=CT)  # Thrust coefficient, based on turbine local free stream [-]
    return WF



# --------------------------------------------------------------------------------}
# --- AD 
# --------------------------------------------------------------------------------{
class OutputAD():
    def setupInterpolants(self):
        if self.Vertical:
            self._fU_Vert = interp2d_pairs(self.y, self.z, self.U, kind='cubic')
            self._fV_Vert = interp2d_pairs(self.y, self.z, self.V, kind='cubic')
            self._fW_Vert = interp2d_pairs(self.y, self.z, self.W, kind='cubic')
        else:
            self._fU_Horz = interp2d_pairs(self.x, self.z, self.U, kind='cubic')
            self._fV_Horz = interp2d_pairs(self.x, self.z, self.V, kind='cubic')
            self._fW_Horz = interp2d_pairs(self.x, self.z, self.W, kind='cubic')

    def fU(self,X,Y,Z):
        if self.Vertical:
            W=self._fW_Vert(Y,Z)
            V=self._fV_Vert(Y,Z)
            U=W*0
        else:
            U=self._fU_Horz(X,Z)
            W=self._fW_Horz(X,Z)
            V=W*0
        return U,V,W

    def fW(self,X,Y,Z):
        if self.Vertical:
            W=self._fW_Vert(Y,Z)
        else:
            W=self._fW_Horz(X,Z)
        U=W*0
        V=W*0
        return U,V,W

    def fHorz(self,vx,vz):
        if self.Vertical:
            if self.HH>9:
                return self.Y,self.Z,self.V,self.W
            else:
                raise Exception('cannot compute fHorz on Vertical plane')
        return self.X,self.Z,self.U,self.W

    def fVert(self,vy,vz):
        if self.Horizontal:
            if self.HH>9:
                return self.X,self.Z,self.U,self.W
            else:
                raise Exception('cannot compute fVert on horizontal plane')
        return self.Y,self.Z,self.V,self.W

    def fCenter(self):
        if self.Horizontal:
            v,ix0 = find_nearest(self.x,0)
            v,iz0 = find_nearest(self.z,0)
            W0 = self.W[iz0,ix0]
        else:
            v,iy0 = find_nearest(self.y,0)
            v,iz0 = find_nearest(self.z,0)
            W0 = self.W[iz0,iy0]
        return W0

    def fAxial(self,z):
        if self.Horizontal:
            v,ix0 = find_nearest(self.x,0)
            z0 = self.Z[:,ix0]
            U0 = self.U[:,ix0]
            V0 = self.V[:,ix0]
            W0 = self.W[:,ix0]
        else:
            v,iy0 = find_nearest(self.y,0)
            z0 = self.Z[:,iy0]
            U0 = self.U[:,iy0]
            V0 = self.V[:,iy0]
            W0 = self.W[:,iy0]
        U=np.interp(z,z0,U0)
        V=np.interp(z,z0,V0)
        W=np.interp(z,z0,W0)
        return U,V,W

    def fRadial(self,r,z0=0):
        v,iz0 = find_nearest(self.z,z0)
        if np.abs(v-z0)>0.001:
            print('Cannot match z value',v,z0)
        if self.Horizontal:
            r0 = self.X[iz0,:]
            U0 = self.U[iz0,:]
            V0 = self.V[iz0,:]
            W0 = self.W[iz0,:]
        else:
            r0 = self.Y[iz0,:]
            U0 = self.U[iz0,:]
            V0 = self.V[iz0,:]
            W0 = self.W[iz0,:]
        U=np.interp(r,r0,U0)
        V=np.interp(r,r0,V0)
        W=np.interp(r,r0,W0)
        return U,V,W

class OutputVC():

    def fU(self,X,Y,Z):
        Y0=Y+self.h_hub
        U,V,W=self.WT.compute_u(X,Y0,Z,longi=False,root=False)
        return U,V,W

    def fHorz(self,vx,vz, doublet_far_field=False):
        X,Z = np.meshgrid(vx,vz) 
        Y=X*0+self.h_hub
        U,V,W=self.WT.compute_u(X,Y,Z,longi=False,root=False)
        return X,Z,U,W

    def fVert(self,vy,vz):
        Y,Z = np.meshgrid(vy,vz) 
        Y0=Y+self.h_hub
        X=Z*0
        U,V,W=self.WT.compute_u(X,Y0,Z,longi=False,root=False)
        return Y,Z,V,W

    def fAxial(self,z):
        x=z*0
        y=z*0+self.h_hub
        U,V,W=self.WT.compute_u(x,y,z,longi=False,root=False)
        return U,V,W

    def fRadial(self,r,z0=0):
        if self.Horizontal:
            x=r
            z=r*0+z0
            y=r*0 + self.h_hub
            U,V,W=self.WT.compute_u(x,y,z,longi=False,root=False)
        else:
            y=r+self.h_hub
            z=r*0+z0
            x=r*0
            U,V,W=self.WT.compute_u(x,y,z,longi=False,root=False)
        return U,V,W

def loadPlane(CT,HH,Lambda,Horizontal,Yaw):
    out=OutputAD()

    if Lambda>20:
        Lambda=99
    if HH>9:
        HH=9.9

    base='CT0{:d}_HH{:02d}_Lambda{:02d}_Yaw{:02d}'.format(int(CT*100),int(HH*10),Lambda,-Yaw)

    if HH>9:
        # Handling unavailable data
        if not Horizontal:
            print('>>> Case not existing',base)
            return None
        if np.abs(Yaw)>0 and Lambda!=99:
            print('>>> Case not existing',base)
            return None

        CFDDataDir ='../data/'
        filename='../data/AD_'+base+'.mat'
        print(filename)
        M=scipy.io.loadmat(filename)
        out.X=M['x']
        out.Z=M['z']
        out.Y=out.X*0
        out.x=np.sort(np.unique(np.round(M['x'].ravel(),3)))
        out.z=np.sort(np.unique(np.round(M['z'].ravel(),3)))
        if len(out.x)!=out.X.shape[1]:
            raise Exception('Wrong x shape')
        if len(out.z)!=out.X.shape[0]:
            raise Exception('Wrong z shape')

        out.y=0
        out.U=M['u']
        out.V=M['v']
        out.W=M['w']
        out.O=out.U*0

        sDir='Horizontal'
    else:
        # Handling unavailable data
        if Yaw!=0:
            print('>>> Case not existing',base)
            return None
        if Lambda not in [6, 99]:
            print('>>> Case not existing',base)
            return None

        CFDDataDir ='../data/Ground/'

        if Horizontal:
            out.x = np.genfromtxt(os.path.join(CFDDataDir,'Horizontal_X.csv'),delimiter=','  )
            out.z = np.genfromtxt(os.path.join(CFDDataDir,'Horizontal_Z.csv'),delimiter=','  )
            out.y = 0
            out.X,out.Z = np.meshgrid(out.x,out.z)
            out.Y=out.X*0
            sDir='Horizontal'
        else:
            out.x =0
            out.z = np.genfromtxt(os.path.join(CFDDataDir,'Vertical_Z.csv'),delimiter=','  )
            if HH==2.0:
                out.y = np.genfromtxt(os.path.join(CFDDataDir,'Vertical_Y_HH20.csv'),delimiter=','  )
            else:
                out.y = np.genfromtxt(os.path.join(CFDDataDir,'Vertical_Y_HH15.csv'),delimiter=','  )
            out.Y,out.Z = np.meshgrid(out.y,out.z)
            out.X=out.Y*0
            sDir='Vertical'
            pass
        base2=base+'_'+sDir
        out.U = np.genfromtxt(os.path.join(CFDDataDir,'{}_U.csv'.format(base2)),delimiter=','  )
        out.V = np.genfromtxt(os.path.join(CFDDataDir,'{}_V.csv'.format(base2)),delimiter=','  )
        out.W = np.genfromtxt(os.path.join(CFDDataDir,'{}_W.csv'.format(base2)),delimiter=','  )
        out.O = np.genfromtxt(os.path.join(CFDDataDir,'{}_Om.csv'.format(base2)),delimiter=','  )

    out.base=(base+'_'+sDir).replace('_',' ')
    out.CT        = CT
    out.HH        = HH
    out.Lambda    = Lambda
    out.Horizontal = Horizontal
    out.Vertical   = not Horizontal
    out.Yaw       = Yaw
    print('>>> ',out.base)

    return out

def setupModel(AD=None,CT=None,HH=None,Lambda=None,Horizontal=None,Yaw=None,MatchMidVel=None,U0=1,gamma_t=None, Model ='VC'):
    VC=OutputVC()

    R  = 1 
    if AD is not None:
        VC.CT         = AD.CT
        VC.HH         = AD.HH
        VC.Lambda     = AD.Lambda
        VC.Yaw        = AD.Yaw
        VC.Horizontal = AD.Horizontal
    else:
        VC.CT         = CT
        VC.HH         = HH
        VC.Lambda     = Lambda
        VC.Yaw        = Yaw
        VC.Horizontal = Horizontal
    Ground=VC.HH<9
    VC.h_hub=VC.HH*R

    if VC.Lambda<30:
        r_bar_cut = 0.10
        nCyl      = 30
    else:
        # Constant CT
        r_bar_cut = 0.0
        nCyl      = 1

    # Loading
    vr_bar = np.linspace(0,1,100)
    Ct_AD  = Ct_const_cutoff(VC.CT,r_bar_cut,vr_bar)

    chi= VC.Yaw*(1+0.3*(1-np.sqrt(1-VC.CT))); # deg
    chi= chi
    print('Chi',chi,VC.CT)
    chi=chi*np.pi/180

    #  WT
    WT=WindTurbine(R=R,e_shaft_yaw0=[0,0,1],e_vert=[0,1,0],r_hub=[0,VC.h_hub,0],Ground=Ground, Model=Model)
    WT.update_wind([U0*np.sin(VC.Yaw*np.pi/180),0,U0*np.cos(VC.Yaw*np.pi/180)])
    WT.set_chi(chi)
    WT.update_loading(r=vr_bar*R, Ct=Ct_AD, Lambda=VC.Lambda,nCyl=nCyl)
    VC.WT=WT
    if MatchMidVel==True:
        #_,_,u_center=AD.fAxial(0)
        u_center=AD.fCenter()
        #print('u_center',u_center)
        #print('u_center',-(1-u_center)*2)
        print('gamma_t',WT.gamma_t)
        WT.gamma_t=np.array([-(U0*np.cos(VC.Yaw*np.pi/180)-u_center)*2])
        print('gamma_t',WT.gamma_t)
        if gamma_t is not None:
            WT.gamma_t=np.array([gamma_t])
            print('gamma_t',WT.gamma_t)
    if gamma_t is not None:
        WT.gamma_t=np.array([gamma_t])
        print('gamma_t',WT.gamma_t)

    return VC


# def computeVC(CT,HH,Lambda,Horizontal,Yaw):
#     def default_WT(self,h_hub=0,Ground=False):
#         # --- Test in Cylinder coord
#         R         = 65
#         U0        = 10
#         r_bar_cut = 0.21
#         r_bar_tip = 0.81
#         Lambda    = 10
#         CT0       = 0.85
#         nCyl      = 20
# 
#         # --- Cutting CT close to the root
#         vr_bar = np.linspace(0,1,100)
#         Ct_AD  = Ct_const_cutoff(CT0,r_bar_cut,vr_bar,r_bar_tip)
# 
#         WTref=WindTurbine(R=R,e_shaft_yaw0=[0,0,1],e_vert=[0,1,0],r_hub=[0,h_hub,0],Ground=Ground)
#         WTref.update_wind([0,0,10])
#         WTref.update_loading(r=vr_bar*R, Ct=Ct_AD, Lambda=Lambda,nCyl=nCyl)
#         return WTref
# 
#     def test_WT_ground(self):
#         root  = False
#         longi = False
#         tang  = True
#         R         = 65
#         # --- Flow field on axis
#         z = np.linspace(-4*R,4*R,10)
#         x = 0*z
#         y = 0*z
# 
#         # --- Testing that Ground effect with h_hub=0 returns twice the induction
#         h_hub     = 0
#         WT =  self.default_WT(h_hub=h_hub)
#         ux_ref,uy_ref,uz_ref = WT.compute_u(x,y,z,root=root,longi=longi,tang=tang,only_ind=True,ground=False)



def plotPlane(data,X=None,Y=None,Z=None,ax=None,minSpeed=None,maxSpeed=None, cmap='coolwarm', colors='k', linewidths=0.8, alpha=0.3,nStreamlines=0,levelsContour=None,levelsLines=None,component='Axial',Horizontal=True, ls='solid', axequal=True):
    if Horizontal:
        x=data.Z
        y=data.X
        u=data.W
        v=data.U
        U=u.T
        V=v.T
    else:
        x=data.Z
        y=data.Y
        u=data.W
        v=data.V
        U=u.T
        V=v.T
    vy=np.unique(y.ravel())
    vx=np.unique(x.ravel())

    if component=='Axial':
        Speed=u
    elif component=='Vorticity':
        Speed=data.O
    else:
        Speed=np.sqrt(u**2+v**2)
    if minSpeed is None:
        minSpeed = Speed.min()
        maxSpeed = Speed.max()

    if not ax:
        fig, ax = plt.subplots()
    else:
        fig =None
    Z = np.ma.masked_where(np.isnan(Speed), Speed)

    # Plot the cut-through
    if levelsContour is None:
        im = ax.pcolormesh(x, y, Z, cmap=cmap, vmin=minSpeed, vmax=maxSpeed)
    elif len(levelsContour)==0:
        im=None
    else:
        im = ax.contourf  (x, y, Z, levels=levelsContour, cmap=cmap, vmin=minSpeed, vmax=maxSpeed)

    if levelsLines is not None:
        rcParams['contour.negative_linestyle'] = 'solid'
        cs=ax.contour(x, y, Z, levels=levelsLines, colors=colors, linewidths=linewidths, alpha=alpha, linestyles=ls)
#         ax.clabel(cs,list(levelsLines))

    if nStreamlines>0:
        if Horizontal:
            yseed=np.linspace(min(vy)*0.9,max(vy)*0.9,nStreamlines)
            start=np.array([yseed*0-4,yseed])
        else:
#             yseed=np.linspace(0,1,nStreamlines)
#             start=np.array([yseed*0-1,yseed])
            start=np.array([[-4,-4,-4,-4],[-0.5,0,0.5,1.5]])
        sp=ax.streamplot(vx,vy,U,V,color='k',start_points=start.T,linewidth=0.7,density=30,arrowstyle='-')
        qv=streamQuiver(ax,sp,n=5,scale=40,angles='xy')
#         sp=ax.streamplot(x,y,u,v,color='k',linewidth=0.7,density=2)

    if Horizontal:
        ax.set_ylabel('y/R [-]')
        ax.set_xlabel('x/R [-]')
    else:
        ax.set_ylabel('z/R [-]')
        ax.set_xlabel('x/R [-]')
    if axequal:
        ax.set_aspect('equal')
    ax.tick_params(direction='in')
    return im,ax,fig


def plotPlaneVert(Y,Z,V,W,ax=None,minSpeed=None,maxSpeed=None, cmap='coolwarm', colors='k', linewidths=0.8, alpha=0.3,levelsContour=None,levelsLines=None,component='Axial',ls='solid', axequal=True):
    vy=np.unique(Y.ravel())
    vz=np.unique(Z.ravel())

    if component=='Axial':
        Speed=W
    else:
        Speed=np.sqrt(V**2+W**2)
    if minSpeed is None:
        minSpeed = Speed.min()
        maxSpeed = Speed.max()

    if not ax:
        fig, ax = plt.subplots()
    else:
        fig =None
    Z = np.ma.masked_where(np.isnan(Speed), Speed)

    # Plot the cut-through
    if levelsContour is None:
        im = ax.pcolormesh(vz, vy, Z.T, cmap=cmap, vmin=minSpeed, vmax=maxSpeed)
    elif len(levelsContour)==0:
        im=None
    else:
        im = ax.contourf  (vz, vy, Z.T, levels=levelsContour, cmap=cmap, vmin=minSpeed, vmax=maxSpeed)

    if levelsLines is not None:
        rcParams['contour.negative_linestyle'] = 'solid'
        cs=ax.contour(vz, vy, Z.T, levels=levelsLines, colors=colors, linewidths=linewidths, alpha=alpha, linestyles=ls)
#         ax.clabel(cs,list(levelsLines))

    ax.set_ylabel('z/R [-]')
    ax.set_xlabel('x/R [-]')
    if axequal:
        ax.set_aspect('equal')
    ax.tick_params(direction='in')
    return im,ax,fig


def getModelsRelErr(angles, Roffsets, rho, UseRho, CT, Models, MatchMidVel, U0=1, R=1, HH=9999, Lambda=np.inf, Yaw=0, Horizontal=True , suffix_GP='', HH_GP=999):
    """
    Helper function to compute relerror of models compared to AD
    """
    ModelsRE=[np.zeros((len(angles),len(rho))) for m in Models]
    ModelsMRE=[np.zeros((1,len(rho))) for m in Models]
    
    AD  = loadPlane(CT = CT, HH = HH, Lambda = Lambda, Horizontal = Horizontal, Yaw=Yaw)
    AD.setupInterpolants()
    if MatchMidVel:
        u_center=AD.fCenter()
        gamma_t =-(U0-u_center)*2
    else:
        fact=1
        gamma_t = -(1-np.sqrt(1-fact*CT))# <<< NOTE gamma factor here!!!
    dmz_dz = gamma_t * R**2 * np.pi # doublet intensity per length

    #---
    GP  = OutputProj(CT = CT, HH = HH_GP, Lambda = Lambda, suffix=suffix_GP, Horizontal=Horizontal)
    GP.setupInterpolants()

    # --- Computing rele error for different directions
    for itheta,(theta,roff) in enumerate(zip(angles,Roffsets)):
        if Horizontal:
            if UseRho:
                Z= rho*np.cos(theta)
                X=-rho*np.sin(theta)
            else:
                Z= rho
                X= rho*0 + roff
            Y=X*0
        else:
            if UseRho:
                Z= rho*np.cos(theta)
                Y=-rho*np.sin(theta)
            else:
                Z= rho
                Y= rho*0 + roff
            X=Y*0
        _,_,ADW = AD.fW(X,Y,Z)
        for iModel, Model in enumerate(Models):

            if Model['name'] in ['VC','VD','SS']:
                VCV  = setupModel(AD,MatchMidVel=MatchMidVel, Model=Model['name'])
                _, _,VCW = VCV.fU(X,Y,Z)
            elif Model['name']=='AD':
                VCW =ADW
            elif Model['name']=='GP':
                #_,_,_,VCW = VCV.fVert()
                _,_,VCW = GP.fW(X,Y,Z)
            else:
                raise Exception('Model not supported')
            ModelsRE[iModel][itheta,:]=(VCW-ADW)/ADW*100

    for iModel, Model in enumerate(Models):
        ModelsMRE[iModel] = np.mean(ModelsRE[iModel],axis=0).ravel()

    return ModelsRE, ModelsMRE

