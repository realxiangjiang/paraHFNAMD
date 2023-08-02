import numpy as np
from scipy.fft import fft, ifft
from scipy.special import loggamma
from math import factorial as fac

import matplotlib as mpl
mpl.use('agg')
import matplotlib.pyplot as plt

mpl.rcParams['axes.unicode_minus'] = False
mpl.rcParams['axes.linewidth'] = 1

def Feven(fx, h, s, x0, k0):
    TNP1 = fx.shape[0] # 2 * N + 1
    assert TNP1 % 2 == 1
    fhat = np.zeros(TNP1, dtype = complex)
    fhat[::2] = fx[::2]
    fhat[0] *= 0.5
    fhat[-1] *= 0.5
    dft_fhat = TNP1 * ifft( np.exp(1j * k0 * np.arange(TNP1) * h) * fhat )
    return np.exp(1j * ( k0 + (2 * np.pi / h) * (np.arange(s * TNP1) / TNP1) ) * x0) * np.tile(dft_fhat, s)

def Fodd(fx, h, s, x0, k0):
    TNP1 = fx.shape[0] # 2 * N + 1
    assert TNP1 % 2 == 1
    ftilde = np.zeros(TNP1, dtype = complex)
    ftilde[1::2] = fx[1::2]
    dft_ftilde = TNP1 * ifft( np.exp(1j * k0 * np.arange(TNP1) * h) * ftilde )
    return np.exp(1j * ( k0 + (2 * np.pi / h) * (np.arange(s * TNP1) / TNP1) ) * x0) * np.tile(dft_ftilde, s)

def abg(k, h): # alpha, beta, gamma
    TNP1 = k.shape[0]
    theta = k * h
    theta3 = theta**3
    sint = np.sin(theta)
    cost = np.cos(theta)
    alpha1 = theta**2 + theta * sint * cost - 2 * sint**2
    beta1  = 2 * (theta * (1 + cost**2) - 2 * sint * cost)
    gamma1 = 4 * (sint - theta * cost)
    small_theta_idx = np.arange(TNP1)[np.abs(theta) < (1.0 / 6.0 / 10**2)] # recommended is 1 / 6
    stheta = theta[small_theta_idx]
    alpha1[small_theta_idx]  = 2.0 / 45.0 * stheta**3 - 2.0 / 315.0 * stheta**5 + 2.0 / 4725.0 * stheta**7
    beta1[small_theta_idx]   = 2.0 / 3.0 + 2.0 / 15.0 * stheta**2 - 4.0 / 105.0 * stheta**4 + 2.0 / 567.0 * stheta**6 - 4.0 / 22275 * stheta**8
    gamma1[small_theta_idx]  = 4.0 / 3.0 - 2.0 / 15.0 * stheta**2 + 1.0 / 210.0 * stheta**4 - 1.0 / 11340.0 * stheta**6
    theta3[small_theta_idx]  = 1.0

    return alpha1 / theta3, beta1 / theta3, gamma1 / theta3


def FourierIntegralFilon(fx, h, s = 1, x0 = 0.0, k0 = 0.0):
    assert h > 0.0
    assert fx.shape[0] >= 3
    if fx.shape[0] % 2 == 0:
        fx = np.append(fx, fx[-1] + h * np.tan(2.0 * np.arctan( (fx[-1] - fx[-2]) / h )
                                                   - np.arctan( (fx[-2] - fx[-3]) / h )))
    TNP1 = fx.shape[0]
    k = k0 + (2 * np.pi / h) * (np.arange(s * TNP1) / TNP1)
    alpha, beta, gamma = abg(k, h)
    return h * ( -1j * alpha * ( fx[-1] * np.exp(1j * k * (x0 + (TNP1 - 1) * h))
                               - fx[0]  * np.exp(1j * k * x0) )
                     + beta  * Feven(fx, h, s, x0, k0) 
                     + gamma * Fodd(fx, h, s, x0, k0) )

def FItest():
    r'''
    the test function is f(x) = 1/(2pi) * e^{-x^2/2} and e^{-|x|}
    the analytic result is f(k) = e^{-k^2/2} and 2 / (1 + k^2)
    '''
    x0 = -10; xm1 = 10.1; N = 51; h = (xm1 - x0) / (N - 1)
    x = x0 + np.arange(N) * h
    s = 3
    k0 = - (2.0 * np.pi / h) * (s * N // 2 / N)
    k = k0 + (2.0 * np.pi / h) * (np.arange(s * N) / N)
    fx = 1.0 / np.sqrt(2.0 * np.pi) * np.exp(-x**2 / 2)
    fx2 = np.exp(- np.abs(x))
    res = [FourierIntegralFilon(fx, h, s, x0, k0),
           FourierIntegralFilon(fx2, h, s, x0, k0)]
    anares = [np.exp(-k**2 / 2.0),
              2.0 / (1.0 + k**2)]
    
    fig, ax = plt.subplots(nrows = 2, ncols = 1,
                           figsize = (4.4, 3.3 * 2))
    center_kidx = np.logical_and(k < 10.0, k > -10.0)
    for irow in range(2):
        ax[irow].plot(k[center_kidx], np.abs(res[irow])[center_kidx], '-', label='Filon')
        ax[irow].plot(k[center_kidx], anares[irow][center_kidx], ':', label='Analytic')
        ax[irow].text(0.85, 0.70, "N = %i" % N,
                      ha='center', va='center',
                      transform=ax[irow].transAxes)
        ax[irow].legend()
    fig.savefig('FItest.png', dpi = 300, bbox_inches = 'tight')

class FastTransformLog:
    def __init__(self, r0=1e-1, H=1e-3):
        print('Fast Transform of Fourier/sine/cosine, spherical Bessel, and Bessel on logarithmical discrete points')
        self.r0 = r0
        self.H = H
        self.rho0 = np.log(self.r0)
        self.TPoH = 2.0 * np.pi / H

    def TPIphi(self, fr, alpha, s = 1): 
        r'''

          phi( t = (2pi / H)(m / N), alpha >= 1/2 )
        
        = (1 / 2pi) Int_{-oo}^{+oo} e^{it x \rho} e^{(\alpha\rho} x fhat(\rho) d\rho
               
        = ...

                      e^{(\alpha + it)(\rho_0 + H)} - e^{(\alpha + it)\rho_0}
        = (1 / 2pi) x -------------------------------------------------------
                                          \alpha + it

                    x \sum_{n=0}^{N-1} e^{\alpha nH}f(r0 e^{nH}) x e^{2\pi i nm/N}

        
        r = r0 x exp(nH), n = 0, 1, ..., N - 1
        r = exp(rho), rho = rho0 + nH, rho0 = ln(r0)
        t = (2pi / H) x (m / N),  m1 <= m < m2
        if s is odd,  s = 2S + 1, S = 0, 1, ..., m1 = -(N+1)/2 + 1 - SN, m2 = N/2 + 1 + SN
        if s is even, s = 2S    , S = 1, 2, ..., m1 = -(SN - 1),         m2 = SN + 1

        for sin/cosin,         alpha = 1 / 2
        for spherical Bessel,  alpha = m + 3 / 2
        for Bessel,            alpha = 1 + mu
        
        '''
        assert s >= 1
        self.s = s
        N = fr.shape[0]
        self.N = N
        
        res = np.zeros(s * N, dtype=complex)
        #expfhat = np.exp( alpha * np.arange(N) * self.H ) * fr + 0.0 * 1j
        expfhat = np.exp( alpha * (self.rho0 + np.arange(N) * self.H) ) * fr + 0.0 * 1j
        dft_expfhat = N * ifft(expfhat)
        
        S = s // 2
        if s % 2:
            self.m1 = - ( (N + 1) // 2 ) + 1 - S * N
            self.m2 = N // 2 + 1 + S * N
            dft_expfhat = np.append(dft_expfhat[N // 2 + 1:],
                                    dft_expfhat[:N // 2 + 1])
        else:
            self.m1 = - (S * N - 1)
            self.m2 = S * N + 1
            dft_expfhat = np.append(dft_expfhat[1:], dft_expfhat[:1])

        for nb in range(s):
            t = 2 * np.pi / self.H / N * np.arange(self.m1 + nb * N, self.m1 + (nb + 1) * N)
            #res[nb * N:(nb + 1) * N] = (  np.exp((self.rho0 + self.H) * (alpha + 1j * t)) 
            #                            - np.exp( self.rho0           * (alpha + 1j * t))  ) / (alpha + 1j * t) * dft_expfhat
            res[nb * N:(nb + 1) * N] = (np.exp((alpha + 1j * t) * self.H) - 1) \
                                     * np.exp(1j * t * self.rho0) / (alpha + 1j * t) * dft_expfhat
        self.tpphit = res
        print( np.vstack(( np.arange(self.m1, self.m2) * 2 * np.pi / self.H / N,
                           np.abs(res) )).transpose()[N//2 - 10:N//2 + 10])

    def TPIphi_Filon(self, fr, alpha, s = 1):
        self.s = s
        N = fr.shape[0]
        rho = self.rho0 + np.arange(N) * self.H
        N += (N + 1) % 2
        self.N = N
        self.t0 = - self.TPoH * (s * N // 2 / N)
        self.tpphit = FourierIntegralFilon(np.exp(alpha * rho) * fr, self.H, s, self.rho0, self.t0)
        t = self.t0 + self.TPoH * (np.arange(s * N) / N)
        print( np.vstack(( t, np.abs(self.tpphit) )).transpose()[s * N // 2 - 10:s * N // 2 + 10])

    def Phi1(self, t):
        return np.imag(loggamma(0.5 - 1j * t))

    def Phi2(self, t):
        return np.arctan(np.tanh(0.5 * np.pi * t))

    def Msin(self, t, l=None, m=None):
        return (8.0 * np.pi)**(-0.5) * np.exp( 1j * (self.Phi1(t) - self.Phi2(t)) )
    
    def Mcos(self, t, l=None, m=None):
        return (8.0 * np.pi)**(-0.5) * np.exp( 1j * (self.Phi1(t) + self.Phi2(t)) )

    def Mlm(self, t, l, m):
        p = l - m
        prod1 = 1.0
        prod2 = 1.0
        if p != 0:
            for j in np.arange(1, p + 1):
                prod1 *= j - 0.5 - 1j * t
        if l != 0:
            for j in np.arange(1, l + 1):
                prod2 /= 2 * j - l + m - 0.5 + 1j * t
        
        return (8.0 * np.pi)**(-0.5) * prod1 * prod2 * (
                np.cos(p * np.pi / 2) * np.exp( 1j * (self.Phi1(t) - self.Phi2(t)) ) +
                np.sin(p * np.pi / 2) * np.exp( 1j * (self.Phi1(t) + self.Phi2(t)) ) )
    
    def Qnmu(self, t, n, mu):
        assert (mu > - 0.5) and (mu < n + 1)
        assert (n - mu) % 2 == 0
        prod1 = 1.0
        prod2 = 1.0
        if n - mu > 0:
            for j in np.arange(1, n - mu, 2):
                prod1 *= (j - 1j * t)
        if n + mu > 0:
            for j in np.arange(1, n + mu, 2):
                prod2 *= (j + 1j * t)
 
        return 1.0 / (2.0 * np.pi) * 2**(- 1j * t) * (prod1 / prod2) * np.exp(1j * 2.0 * self.Phi1(t / 2.0))


    def FTlog(self, lorn, mormu, funM, kappa0, sp = 1):
        r'''
         ghat(\kappa = \kappa_0 + n x (H / s))

        = ...
           
          e^{\beta\kappa}(e^{i\kappa x (2pi / H) / N} - 1)
        = ------------------------------------------------
                            i\kappa

        
        x sum_{m = m1}^{m2 - 1} 
          
          e^{i\kappa_0(2pi/H)(m/N)} x 2pi\phi( (2pi/H)(m/N) )M( (2pi/H)(m/N) )

                                    x e^{2pi i nm / (sN)}
        '''
        assert sp >= 1
        if funM == self.Msin or funM == self.Mcos:
            beta = - 0.5
        elif funM == self.Mlm:
            beta = mormu - 1.5
        elif funM == self.Qnmu:
            beta = mormu - 1.0

        m1tom2 = np.arange(self.m1, self.m2)
        TPIoverNHm =  (2.0 * np.pi / self.H / self.N) * m1tom2
        data = np.exp( 1j * kappa0 * TPIoverNHm ) * self.tpphit * funM(TPIoverNHm, lorn, mormu)
        half_sN = (self.s * self.N + 1) // 2 - 1
        data = np.append(data[half_sN:], data[:half_sN])
        dft_data = self.s * self.N * ifft(data)

        kappa = kappa0 + np.arange(sp * self.s * self.N) * (self.H / self.s)
        close_zero_idx = np.arange(sp * self.s * self.N)[np.isclose(kappa, 0.0)]
        numerator = np.exp(1j * kappa * (2.0 * np.pi / self.H) / self.N) - 1.0
        denominator = kappa * 1j
        numerator[close_zero_idx] = 2.0 * np.pi / self.H / self.N
        denominator[close_zero_idx] = 1.0
        
        return kappa, np.exp(beta * kappa) * (numerator / denominator) * dft_data

    def FTlog_Filon(self, lorn, mormu, funM, kappa0, sp = 1):
        assert sp >= 1
        if funM == self.Msin or funM == self.Mcos:
            beta = - 0.5
        elif funM == self.Mlm:
            beta = mormu - 1.5
        elif funM == self.Qnmu:
            beta = mormu - 1.0

        delta_t = self.TPoH / self.N
        delta_kappa = self.H / self.s
        t = self.t0 + np.arange(self.s * self.N) * delta_t
        sN = self.s * self.N
        sN += (sN + 1) % 2
        kappa = kappa0 + np.arange(sp * sN) * delta_kappa
        return kappa, np.exp(beta * kappa) * FourierIntegralFilon(self.tpphit * funM(t, lorn, mormu), delta_t,
                                                                  sp, self.t0, kappa0)
    def SphericalBesselTransrom(self, fr, l, kappa0, s = 1, sp = 1):
        NN = fr.shape[0]
        r = self.r0 * np.exp( np.arange(NN) * self.H )
        k = np.exp(kappa0 + np.arange( sp * s * (NN + (NN + 1) % 2) ) * (self.H / s))
        res = 0.0
        for m in range(l + 1):
            if (l + 1 - m) % 4 == 0:
                pm1 = 1.0
                funM = self.Mcos
            elif (l + 1 - m) % 4 == 1:
                pm1 = 1.0
                funM = self.Msin
            elif (l + 1 - m) % 4 == 2:
                pm1 = -1.0
                funM = self.Mcos
            else:
                pm1 = -1.0
                funM = self.Msin

            ffr = fr * r**(1 - m)
            self.TPIphi_Filon(ffr, 1.0 / 2.0, s)
            res += pm1 * self.FTlog_Filon(0, 0, funM, kappa0, sp)[1] \
                 * (fac(l + m) / fac(l - m)) / (fac(m) * 2**m) / k**(m + 1)
        return np.log(k), res

def TestSBT(): # test for spherical Bessel transfrom
    #f(r) = e^(-r)
    l = 1; m = 0; alpha = m + 3.0 / 2.0
    N = 64; H = 0.2; r0 = np.exp(-10.0) # \delta\rho   = 0.2
    s = 1; kappa0 = -3.0; sp = 1         # \delta\kappa = 0.2
    r = r0 * np.exp(np.arange(N) * H)
    fr = np.exp(-r)
    sbtLog = FastTransformLog(r0, H)
    #sbtLog.TPIphi(fr, alpha, s)
    #kappa, res = sbtLog.FTlog(l, m, sbtLog.Mlm, kappa0, sp)
    #sbtLog.TPIphi_Filon(fr, alpha, s)
    #kappa, res = sbtLog.FTlog_Filon(l, m, sbtLog.Mlm, kappa0, sp)
    kappa, res = sbtLog.SphericalBesselTransrom(fr, l, kappa0, s, sp)
    print(np.vstack(( kappa, np.abs(res) )).transpose()[kappa > -3.00001][:10])

if __name__ == '__main__':
    #TestSBT()
    FItest()

