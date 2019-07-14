"""
Reference:
    [1] E. Branlard - Wind Turbine Aerodynamics and Vorticity Based Method, Springer, 2017
"""
#--- Legacy python 2.7
from __future__ import division
from __future__ import print_function
# --- General
import unittest
import numpy as np
import numpy.matlib

# --------------------------------------------------------------------------------}
# --- Raw/core functions, polar coordinates
# --------------------------------------------------------------------------------{
def vl_semiinf_u(xa,ya,za,xe,ye,ze,Gamma,visc_model,t):
    """ Induced velocity at point A, from a semi infinite vortex line starting at point 0, and directed along e
    See fUi_VortexlineSemiInfinite
    TODO vectorize 
    """
    norm_a      = np.sqrt(xa * xa + ya * ya + za * za)
    denominator = norm_a * (norm_a - xa * xe + ya * ye + za * ze)
    crossprod   = np.array([[ye * za - ze * ya],[ze * xa - xe * za],[xe * ya - ye * xa]])
    # check for singularity */
    if (denominator < 1e-12 or norm_a < 1e-05):
        return np.array([[0],[0],[0]])
    # viscous model */
    Kv = 1.0
    if visc_model==0:
        # No vortex core model 
        Kv = 1.0
    elif visc_model==1:
        # Rankine  - t<=>rc 
        norm_r0 = np.sqrt((xa - xb) * (xa - xb) + (ya - yb) * (ya - yb) + (za - zb) * (za - zb))
        h = np.sqrt(crossprod[0]**2 + crossprod[1]**2+ crossprod[2]**2) / norm_r0
        if (h < t):
            Kv = h * h / t / t
        else:
            Kv = 1.0
    elif visc_model==2:
        # Lamb-Oseen  - t<=>rc
        norm_r0 = np.sqrt((xa - xb) * (xa - xb) + (ya - yb) * (ya - yb) + (za - zb) * (za - zb))
        h = np.sqrt(crossprod[0]**2 + crossprod[1]**2+ crossprod[2]**2) / norm_r0
        Kv = 1.0 - np.exp(- 1.25643 * h * h / t / t)
    elif visc_model==3:
        # Vatistas n=2  - t<=>rc */
        norm_r0 = np.sqrt((xa - xb) * (xa - xb) + (ya - yb) * (ya - yb) + (za - zb) * (za - zb))
        h = np.sqrt(crossprod[0]**2 + crossprod[1]**2+ crossprod[2]**2) / norm_r0
        Kv = h * h / np.sqrt(t ** 4 + h ** 4)
    elif visc_model==4:
        # Cut-off radius no dimensions - delta*norm(r0)^2 - t<=>delta^2
        norm_r0 = np.sqrt((xa - xb) * (xa - xb) + (ya - yb) * (ya - yb) + (za - zb) * (za - zb))
        denominator = denominator + t * norm_r0
        Kv = 1.0
    elif visc_model==5:
        # Cut-off radius dimension (delta l_0)^2 - t<=>(delta l)^2 */
        Kv = 1.0
        denominator = denominator + t
    Kv = Gamma*Kv /4.0/ np.pi / denominator
    Uout = Kv * crossprod
    return Uout
