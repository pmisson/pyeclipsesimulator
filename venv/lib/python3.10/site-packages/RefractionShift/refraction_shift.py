# -*- coding: us-ascii -*-
"""
This code implements the calculation of the lateral shift due to atmospheric 
refraction. It is based on the resolution of an ODE system using a fourth-order
Runge Kutta. The theoretical model of the atmosphere is a two-layer one with a 
fixed temperature gradient. (ODE stands for Ordinary Differential Equation)

Author       : Hanae Labriji
Institute    : Office National d'Etudes et de Recherches Aerospatiales ONERA
Last modified: 2021-04-22
Dependencies : This code requires the python module refractivityModels found 
               in the library AstroAtmosphere 
"""

from __future__ import division
import numpy as np
from AstroAtmosphere import refractivityModels
from math import exp

class integration:
       def simpson(f, x0, xf, n):
        """
        Implements Simpson integration algorithm of a one-variable function.
        Inputs : 
            f       : function handle
            x0      : start point
            xf      : end point
            n       : number of samples used
        Outputs : 
            h/6*res : integral of f between x0 and xf
        """
        h = (xf-x0)/n # Integration step
        res = -f(x0) + f(xf)
        for i in range(0,n):
            res = res + 2*f(x0+i*h) + 4*f(x0+(i+1/2)*h)
            
        return h/6*res
    
    
class refraction:

    R_t         = 6378120 # Earth radius (m)
    R_equ       = 6378137 # Earth equatorial radius (m)
    R_pol       = 6356752 # Earth polar radius
    g_0         = 9.78841733 # 
    R           = 8.314472    #J mol-1 K-1 # universal gas constant
    h_max       = 80000       #m #maximum altitude of the atmosphere
    M_d         = 28.966*1e-3 #kgmol-1 # Molar mass of dry air
    h_tropo     = 11000       #m #Height of the troposphere
    h_0         = 0           #m # Altitude at the observer level

    alpha       = -6.5*1e-3  #Kelvin/m #temperature gradient in the troposphere
    gamma       = -g_0*M_d/R/alpha 

    
    
    def __init__(self, T0, P0, h0):
        """
        Initializes the atmospheric class with different temperature and 
        pressure at the observer, along with a chosen altitude.
        Inputs : 
            T0 : temperature at the observer level (K)
            P0 : pressure at the observer level (Pa)
            h0 : altitude at the observer level (m)
        """
        if T0 < 100 or h0 <0 or h0 > self.h_tropo:
            raise  Exception('The values of the input parameters are out of\
                             range.')
            
        self.T_0 = T0
        self.P_0 = P0
        self.h_0 = h0
        self.T_tropo = self.T_0 + self.alpha*(self.h_tropo - h0) 
        self.kappa = -self.g_0*self.M_d/self.R/self.T_tropo
    
    
    
    def refraction_index(self, h,lmbda):
        """
        Computes the refraction index of air. 
        Inputs: 
            h     : altitude (m)
            lmbda : wavelength (m)
        Outputs: 
            n     : refractive index at the height h and wavelength lmbda
            dn    : derivative of n with respect to h
        """        
        
        n= 1 + self.A_dry(lmbda)*self.pression_model(h)/ \
           self.temperature_model(h)
            
        if h <= self.h_tropo:
            dn = self.alpha * (self.gamma-1)*(n-1)/ \
                 self.temperature_model(h)
        else:
            dn = self.kappa *(n-1)
   
        return n, dn
    
    
    def A_dry(self, lmbda):
        """
        Computes the refractivity of air for a given wavelength. Up to 1.3 um,
        we use Ciddor (1996) : 
            https://doi.org/10.1364/AO.35.001566
        Above 1.3 um and up to 24 um, we use Mathar (2007) : 
            https://doi.org/10.1088/1464-4258/9/5/008
        
        Input: 
            lmbda : wavelength (m)
        """
        lmbda_um = 1e+6*lmbda
        if lmbda_um>24 :
            raise  Exception('lmbda should not exceed 24 um. The value of \
                             lmbda was: {}'.format(lmbda_um))
        if lmbda_um<0.2 :
            raise  Exception('lmbda should not be under 0.2 um. The value of \
                             lmbda was: {}'.format(lmbda_um))
        
        if lmbda_um < 1.3:
            A = 1e-10*(28760.4+162.88/(lmbda_um**2)+1.36/(lmbda_um**4))*\
                273.15/1013.25
        elif lmbda_um < 24 :
            A = (refractivityModels.Mathar(lmbda_um,self.T_0,self.P_0 )-1)*\
                (self.T_0)/self.P_0
        else : 
            print("The given wavelength is out of the requested range.")
        return A
    
    
    def temperature_model(self, h):
        """
        Computes temperature at a given altitude.
        Input: 
            h : altitude (m)
        Outputs:
            T : temperature (K)
        """
        if h < self.h_tropo:
            T = self.T_0 + self.alpha*(h-self.h_0)
        else:
            T = self.T_tropo
        return T
    
    
    def pression_model(self, h):
        """
        Computes pressure at a given altitude.
        Input: 
            h : altitude (m)
        Outputs:
            P : pressure (Pa)
        """
        if h <= self.h_tropo:
            P = self.P_0*(self.temperature_model(h)/ \
                self.T_0)**self.gamma
        else:
            P = self.P_0*(self.T_tropo/self.T_0)**self.gamma \
                *exp(self.kappa*(h-self.h_tropo))    
 
        return P
    
    
    def density_model(self, h):
        """
        Computes air density at a given altitude.
        Input: 
            h : altitude (m)
        Outputs:
            d : density (kg/m3)
        """
        d = self.M_d/self.R*self.pression_model(h)/ \
            self.temperature_model(h)  
        return d

    
    def ode_AngularShift(self, X, lmbda): 
        """
        Implements the left hand side of refraction ODE. 
        Inputs : 
            X     : state variables [h, z, theta, zeta]
            lmbda : wavelength (m)
        Outputs : 
            dx_ds : derivative of X with respect to the length along the ray 
            path (s)
        """
        h = X[0]
        #z = X[1] # the value of z is not needed
        #theta = X[2] # the value of theta is not needed
        zeta = X[3]
        
        n, dn = self.refraction_index(h,lmbda)
        
        dx_ds = np.zeros((4,1))
        
        dx_ds[0] = np.cos(zeta)
        dx_ds[1] = - np.sin(zeta)/n * dn
        dx_ds[2] = np.sin(zeta)/(self.R_t + h)
        dx_ds[3] = - np.sin(zeta)/(self.R_t + h) - np.sin(zeta)/n * dn
        
        return np.transpose(dx_ds)[0]
    
    def solve_ode_AngularShift(self, z0, lmbda, ds, smax):
        """
        Solves refraction ODE using a 4th order Runge-Kutta method. 
        Inputs : 
            z0    : apparent zenith angle at the observer (rad)
            lmbda : wavelength (m)
            ds    : integration step (m), we recommend 100m
            smax     : maximum length along the ray path (m)
        Outputs : 
            X     : state variable along the ray path
            s     : true maximum length L
        """
        
        if ds < 0:
            raise Exception('The integration step should be a positive number.\
                            The value of ds was: {}'.format(ds))
            
        h0 = self.h_0
        X0 = [h0,z0,0,z0]
        n = int((smax-h0)/ds)
        X = np.zeros((n,) + np.shape(X0))
        X[0] = X0
        flag = 0
        h = ds/2
        s = 0
        
        for i in range(n-1):
            k1 = self.ode_AngularShift(X[i],lmbda)
            k2 = self.ode_AngularShift(X[i]+h*k1,lmbda)
            k3 = self.ode_AngularShift(X[i]+h*k2,lmbda)
            k4 = self.ode_AngularShift(X[i]+2*h*k3,lmbda)
            X[i+1] = X[i]+h/3*( k1+2*k2+2*k3+k4)
            
            s = s+ds
            if X[i+1,0]>self.h_tropo and flag == 0:
                flag = 1
                deltaH = X[i+1,0]-self.h_tropo
                deltaS = ds - deltaH/np.abs(np.cos(X[i+1,3]))
                hTropopause = deltaS/2
                k1Tropopause = self.ode_AngularShift(X[i],lmbda)
                k2Tropopause = self.ode_AngularShift(X[i]+hTropopause*\
                               k1Tropopause,lmbda)
                k3Tropopause = self.ode_AngularShift(X[i]+hTropopause*\
                               k2Tropopause,lmbda)
                k4Tropopause = self.ode_AngularShift(X[i]+2*hTropopause*\
                               k3Tropopause,lmbda)
                X[i+1] = X[i]+hTropopause/3*( k1Tropopause+2*k2Tropopause+2*\
                         k3Tropopause+k4Tropopause)

        return X, s
    
    
    def ode_LateralShift(self, X, lmbda, Zinfty): 
        """
        Implements the left hand side of the lateral shift ODE. 
        Inputs : 
            X     : state variables [h, z, theta, zeta, b]
            lmbda : wavelength (m)
        Outputs : 
            dx_ds : derivative of X with respect to the length along the ray 
            path (s)
        """
        h = X[0]
        z = X[1] 
        #theta = X[2] # the value of theta is not needed
        zeta = X[3]
        
        n, dn = self.refraction_index(h,lmbda)
        
        dx_ds = np.zeros((5,1))
        
        dx_ds[0] = np.cos(zeta)
        dx_ds[1] = - np.sin(zeta)/n * dn
        dx_ds[2] = np.sin(zeta)/(self.R_t + h)
        dx_ds[3] = - np.sin(zeta)/(self.R_t + h) - np.sin(zeta)/n * dn
        dx_ds[4] = np.sin(z-Zinfty)
        return np.transpose(dx_ds)[0]
    
    
    def solve_ode_LateralShift(self, z0, zinf, ds, smax, lmbda):
        """
        Solves lateral shift ODE using a 4th order Runge-Kutta method. 
        Inputs : 
            z0    : apparent zenith angle at the observer (rad)
            zinf  : true zenith angle (rad)
            lmbda : wavelength (m)
            ds    : integration step (m), we recommend 100m
            smax    : maximum length along the ray path (m)
        Outputs : 
            X     : state variable along the ray path
            s     : true maximum length L
        """
        if ds < 0:
            raise Exception('The integration step should be a positive number.\
                            The value of ds was: {}'.format(ds))
            
        h0 = self.h_0
        n = int((smax-h0)/ds)
        X0 = [h0,z0,0,z0,0]
        X = np.zeros((n,)+np.shape(X0))
        X[0] = X0
        flag = 0
        h = ds/2
        
        s = 0
        for i in range(n-1):
            k1 = self.ode_LateralShift(X[i],lmbda, zinf)
            k2 = self.ode_LateralShift(X[i]+h*k1,lmbda, zinf)
            k3 = self.ode_LateralShift(X[i]+h*k2,lmbda, zinf)
            k4 = self.ode_LateralShift(X[i]+2*h*k3,lmbda, zinf)
            X[i+1] = X[i]+h/3*( k1+2*k2+2*k3+k4)
            
            s = s+ds
            
            if X[i+1,0]>self.h_tropo and flag == 0:
                flag = 1
                deltaH = X[i+1,0]-self.h_tropo
                deltaS = ds - deltaH/np.abs(np.cos(X[i+1,3]))
                hTropopause = deltaS/2
                k1Tropopause = self.ode_LateralShift(X[i],lmbda, zinf)
                k2Tropopause = self.ode_LateralShift(X[i]+hTropopause*\
                               k1Tropopause,lmbda, zinf)
                k3Tropopause = self.ode_LateralShift(X[i]+hTropopause*\
                               k2Tropopause,lmbda, zinf)
                k4Tropopause = self.ode_LateralShift(X[i]+2*hTropopause*\
                               k3Tropopause,lmbda, zinf)
                X[i+1] = X[i]+hTropopause/3*( k1Tropopause+2*k2Tropopause+2*\
                         k3Tropopause+k4Tropopause)
                
    
        return X
    
    
    def get_AngularShift(self, z0, lmbda, ds = 100, *args):
        """
        Computes refraction angle Ra using the previously defined methods. 
        Inputs : 
            z0    : apparent zenith angle at the observer (rad)
            lmbda : wavelength (m)
            ds    : integration step (m), we recommend 100m
            smax  : maximum length along the ray path (m)
        Outputs : 
            Ra    : refraction angle Ra (rad)
        """
        if len(args) == 0:
            if abs(z0) < np.pi/2:
                smax = self.h_max/np.cos(z0)
            else :
                smax = self.h_max/np.cos(np.deg2rad(85))
        elif (len(args) == 1):
            smax = args[0]
        else:
            raise  Exception('Too many input parameters.')
            
        sol, s = self.solve_ode_AngularShift(z0, lmbda, ds, smax)
        return sol[-1, 1] - sol[0,1]
    
    
    
    def get_LateralShift(self, z0, lmbda, ds = 100, *args):
        """
        Computes the lateral shift using the previously defined methods. 
        Inputs : 
            z0    : apparent zenith angle at the observer (rad)
            zinf  : true zenith angle (rad)
            lmbda : wavelength (m)
            ds    : integration step (m), we recommend 100m (optional)
            smax  : maximum length along the ray path (m)  (optional)
        Outputs : 
            Ra    : refraction angle Ra (rad)
        """
        if len(args) == 0:
            if abs(z0) < np.pi/2:
                smax = self.h_max/np.cos(z0)
            else :
                smax = self.h_max/np.cos(np.deg2rad(85))
        elif (len(args) == 1):
            smax = args[0]
        else:
            raise  Exception('Too many input parameters.')
            
        zinf = self.get_AngularShift(z0, lmbda, ds, smax) + z0
        sol  = self.solve_ode_LateralShift(z0, zinf, ds, smax, lmbda)
        return -sol[-1, 4]
    
    
    
    def get_approx1_LateralShift(self, z0, lmbda):
        """
        Computes the first order approximation of the lateral shift.
        Inputs : 
            z0    : apparent zenith angle (rad)
            lmbda : wavelength (m)
        Output : 
            b0    : lateral shift (m)
        """
        b0 = self.A_dry(lmbda)*self.R/self.M_d/self.g_0*np.tan(z0)/np.cos(z0)\
             *self.P_0
        return b0
    
    
        
    def get_approx2_LateralShift(self, z0, lmbda):
        """
        Computes the second order approximation of the lateral shift.
        Inputs : 
            z0    : apparent zenith angle (rad)
            lmbda : wavelength (m)
        Output : 
            b0    : lateral shift (m)
        """
        hrho = lambda x : x*self.density_model(x)
        rho_carree = lambda x : self.density_model(x)**2
        
        alpha_rhos = self.A_dry(lmbda)*self.R/self.M_d
        
        int_hrho = integration.simpson(hrho, self.h_0, self.h_max, 1000)
        int_rho = integration.simpson(self.density_model, self.h_0, self.h_max,
                  1000)
        int_rho_carree = integration.simpson(rho_carree, self.h_0, self.h_max,
                         1000)
        
        A_shift = int_rho-2/self.R_t*int_hrho + alpha_rhos*\
                  self.density_model(0)*int_rho - alpha_rhos*int_rho_carree
        
        B_shift = 3/self.R_t*int_hrho-2*alpha_rhos*self.density_model(0)*\
                  int_rho + 3/2 * alpha_rhos * int_rho_carree
        
        b2 = alpha_rhos*(np.tan(z0)/np.cos(z0)*A_shift - np.tan(z0)**3\
             /np.cos(z0)*B_shift)
        
        return b2
    
    def get_approx32_LateralShift(self, z0, lmbda):
        """
        Computes an approximation of the lateral shift, of order 3/2. This 
        approximation presents the advantage of being separable in lambda and 
        z0 (respectively the observation wavelength and the zenith angle).
        Inputs : 
            z0    : apparent zenith angle (rad)
            lmbda : wavelength (m)
        Output : 
            b0    : lateral shift (m)
        """
        
        hrho = lambda x : x*self.density_model(x)
        
        alpha_rhos = self.A_dry(lmbda)*self.R/self.M_d
        
        int_hrho = integration.simpson(hrho, self.h_0, self.h_max, 1000)
        int_rho = integration.simpson(self.density_model, self.h_0, self.h_max,
                  1000)
        
        A_shift_32 = int_rho - 2/self.R_t*int_hrho
        
        B_shift_32 = 3/self.R_t*int_hrho 
        
        b32 = alpha_rhos*(np.tan(z0)/np.cos(z0)*A_shift_32 - \
              np.tan(z0)**3/np.cos(z0)*B_shift_32)
        
        return b32
    
###############################################################################