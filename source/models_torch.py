import numpy as np
import torch
import torch.nn as nn
import cmath
from torchquad import Trapezoid, Simpson
from torch.distributions.multivariate_normal import MultivariateNormal
import torchaudio


'''
class Heston1(nn.Module):
    def __init__(self, r, kappa, theta, sigma, rho):
        
        
        - s0, v0: initial parameters for asset and variance
        - rho   : correlation between asset returns and variance
        - kappa : rate of mean reversion in variance process
        - theta : long-term mean of variance process
        - sigma : vol of vol / volatility of variance process
        - r     : risk free rate
        - lambd : risk premium of variance
        
        super(Heston1, self).__init__()

        # Trainable params
        self.kappa = nn.Parameter(torch.tensor(float(kappa)))
        self.theta = nn.Parameter(torch.tensor(float(theta)))
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        self.rho = nn.Parameter(torch.tensor(float(rho)))
        self.r = nn.Parameter(torch.tensor(float(r)))


    def initialize(self, s0, v0, lambd):
        self.s0 = torch.tensor(s0)
        self.v0 = torch.tensor(v0) 
        self.lambd = torch.tensor(lambd) # risk premium of variance

    # forward method
    def forward(self, strike, expire): 

        # Rectangle integral resolution
        P, umax, N = 0, 100, 1000
        dphi=umax / N #dphi is width
        phis =  dphi * (2 * torch.arange(1, N) + 1) / 2
        numerators = map(lambda phi: torch.exp(self.r*expire)*self.heston_charfunc(phi-1j, expire) - strike * self.heston_charfunc(phi, expire), phis)
        denominators = map(lambda phi: 1j*phi*strike**(1j*phi), phis)

        numerators = torch.vstack(list(numerators))
        denominators = torch.vstack(list(denominators))
        P = dphi * (numerators / denominators).sum(0)
        
        
        for i in range(1, N):
            phi = dphi * (2*i + 1)/2 # midpoint to calculate height

            numerator = torch.exp(self.r*expire)*self.heston_charfunc(phi-1j, expire) - strike * self.heston_charfunc(phi, expire)
            denominator = 1j*phi*strike**(1j*phi)
            P += dphi * numerator/denominator
        
       
        return torch.real((self.s0 - strike*torch.exp(-self.r*expire))/2 + P/np.pi)
    
    
    def heston_charfunc(self, phi, expire):
    
        # constants
        a = self.kappa*self.theta
        b = self.kappa+self.lambd
    
        # common terms w.r.t phi
        rspi = self.rho*self.sigma*phi*1j
    
        # define d parameter given phi and b
        d = torch.sqrt( (self.rho*self.sigma*phi*1j - b)**2 + (phi*1j+phi**2)*self.sigma**2 )
    
        # define g parameter given phi, b and d
        g = (b-rspi+d)/(b-rspi-d)
    
        # calculate characteristic function by components
        exp1 = torch.exp(self.r*phi*1j*expire)
        term2 = self.s0**(phi*1j) * ( (1-g*torch.exp(d*expire))/(1-g) )**(-2*a/self.sigma**2)
        exp2 = torch.exp(a*expire*(b-rspi+d)/self.sigma**2 + self.v0*(b-rspi+d)*( (1-torch.exp(d*expire))/(1-g*torch.exp(d*expire)) )/self.sigma**2)
        return exp1*term2*exp2
'''


class Heston(nn.Module):
    def __init__(self, r, kappa, theta, sigma, rho):
        
        '''
        - s0, v0: initial parameters for asset and variance
        - rho   : correlation between asset returns and variance
        - kappa : rate of mean reversion in variance process
        - theta : long-term mean of variance process
        - sigma : vol of vol / volatility of variance process
        - r     : risk free rate
        - lambd : risk premium of variance
        '''
        super(Heston, self).__init__()

        # Trainable params
        self.kappa = nn.Parameter(torch.tensor(float(kappa)))
        self.theta = nn.Parameter(torch.tensor(float(theta)))
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        self.rho = nn.Parameter(torch.tensor(float(rho)))
        self.r = nn.Parameter(torch.tensor(float(r)))
        self.quad = Simpson()

    def initialize(self, s0, v0, lambd):
        self.s0 = torch.tensor(float(s0))
        self.v0 = torch.tensor(float(v0))
        self.lambd = torch.tensor(float(lambd)) # risk premium of variance

    # forward method
    def forward(self, strike, expire): 
        a = self.s0*self.Heston_P_Value(strike, expire, 1)
        b = strike*torch.exp(-self.r*expire)*self.Heston_P_Value(strike, expire,2)
        return a-b

    def Heston_P_Value(self, strike, expire, typ):
        return 0.5+(1./np.pi) * self.quad.integrate(lambda xi: self.Int_Function_1(xi, strike, expire, typ), 
                                                    dim=1, 
                                                    N=300, 
                                                    integration_domain=[[1e-8, 500.]])
       
    def Int_Function_1(self, xi, strike, expire, typ):
        return (cmath.e**(-1j*xi*torch.log(strike)) * self.Int_Function_2(xi, expire, typ)/(1j*xi)).real

    def Int_Function_2(self, xi, expire, typ):
        if typ == 1:
            w = 1.
            b = self.kappa - self.rho*self.sigma
        else:
            w = -1.
            b = self.kappa
        ixi = 1j*xi
        #return (rho*sigma*ixi-b)*(rho*sigma*ixi-b) - sigma*sigma*(w*ixi-xi*xi)
        d = torch.sqrt((self.rho*self.sigma*ixi-b)*(self.rho*self.sigma*ixi-b) - self.sigma*self.sigma*(w*ixi-xi*xi))
        g = (b-self.rho*self.sigma*ixi-d) / (b-self.rho*self.sigma*ixi+d)
        #print(d)
        ee = cmath.e**(-d*expire)
        C = self.r*ixi*expire + self.kappa*self.theta/(self.sigma*self.sigma)*((b-self.rho*self.sigma*ixi-d)*expire - 2.*torch.log((1.0-g*ee)/(1.-g)))
        D = ((b-self.rho*self.sigma*ixi-d)/(self.sigma*self.sigma))*(1.-ee)/(1.-g*ee)
        return cmath.e**(C + D*self.v0 + ixi*torch.log(self.s0))

    def simulate(self, expire, steps, num_chains):
        dt = expire / steps
        mu = torch.tensor([0.0, 0.0])
        cov = torch.tensor([[1.0, self.rho],
                            [self.rho, 1.0]])
        S, v = torch.zeros(steps + 1, num_chains), torch.zeros(steps + 1, num_chains)
        S[0] = self.s0.repeat(num_chains)
        v[0] = self.v0.repeat(num_chains)
        # sampling correlated brownian motions under risk-neutral measure
        Z_dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=cov)
        Z = Z_dist.sample([steps, num_chains])
        for i in range(1, steps+1):
            S[i] = S[i-1] * torch.exp( (self.r - 0.5*v[i-1])*dt + torch.sqrt(v[i-1] * dt) * Z[i-1,:,0] )
            v[i] = torch.maximum(v[i-1] + self.kappa*(self.theta-v[i-1])*dt + self.sigma*torch.sqrt(v[i-1]*dt)*Z[i-1,:,1], torch.tensor(0))
    
        return S, v



def phi(x): ## Gaussian density
    return np.exp(-x*x/2.)/np.sqrt(2*np.pi)

      
class BlackScholes(nn.Module):
    def __init__(self, s0, v0, r, d=0., CallPutFlag=True):
        
        '''
        - s0, v0: initial parameters for asset and variance
        - rho   : correlation between asset returns and variance
        - kappa : rate of mean reversion in variance process
        - theta : long-term mean of variance process
        - sigma : vol of vol / volatility of variance process
        - r     : risk free rate
        - lambd : risk premium of variance
        '''
        super(BlackScholes, self).__init__()

        # Trainable params
        self.s0 = torch.tensor(float(s0))
        self.v0 = torch.tensor(float(v0))
        self.d = torch.tensor(float(d))
        self.r = torch.tensor(float(r))
        self.CallPutFlag = CallPutFlag
        self.norm = torch.distributions.normal.Normal(loc=0.0, scale=1.0)


    def BlackScholesCore_(self, strike, expire):
        ## DF: discount factor
        ## F: Forward
        vsqrt=self.v0*torch.sqrt(expire)
        DF = torch.exp(-self.r * expire)
        F = torch.exp((self.r - self.d)*expire) * self.s0
        d1 = (torch.log(F/strike)+(vsqrt*vsqrt/2.))/vsqrt
        d2 = d1-vsqrt
        if self.CallPutFlag:
            return DF*(F*self.norm.cdf(d1)-strike*self.norm.cdf(d2))
        else:
            return DF*(strike*self.norm.cdf(-d2)-F*self.norm.cdf(-d1))

    def forward(self, strike, expire):
        ## r, d: continuous interest rate and dividend
        return self.BlackScholesCore_(strike, expire)



class roughHeston(nn.Module):
    def __init__(self, r, kappa, theta, sigma, rho):
        
        '''
        - s0, v0: initial parameters for asset and variance
        - rho   : correlation between asset returns and variance
        - kappa : rate of mean reversion in variance process
        - theta : long-term mean of variance process
        - sigma : vol of vol / volatility of variance process
        - r     : risk free rate
        - lambd : risk premium of variance
        '''
        super(roughHeston, self).__init__()

        # Trainable params
        self.kappa = nn.Parameter(torch.tensor(float(kappa)))
        self.theta = nn.Parameter(torch.tensor(float(theta)))
        self.sigma = nn.Parameter(torch.tensor(float(sigma)))
        self.rho = nn.Parameter(torch.tensor(float(rho)))
        self.r = nn.Parameter(torch.tensor(float(r)))
        self.quad = Simpson()


def g(x, a):
    """
    TBSS kernel applicable to the rBergomi variance process.
    """
    return x**a

def b(k, a):
    """
    Optimal discretisation of TBSS process for minimising hybrid scheme error.
    """
    return ((k**(a+1.0)-(k-1.0)**(a+1.0))/(a+1.0))**(1.0/a)

class roughBergomi(nn.Module):
    def __init__(self, r, a, rho, xi, eta, steps=100, num_chains=50000):

        '''
        - s0, v0: initial parameters for asset and variance
        - rho   : correlation between asset returns and variance
        - steps : steps per year
        - num_chains: number of paths for time series
        '''
        super(roughBergomi, self).__init__()

        # Trainable params
        self.r = nn.Parameter(torch.tensor(float(r)))
        self.a = nn.Parameter(torch.tensor(float(a)))
        self.rho = nn.Parameter(torch.tensor(float(rho)))
        self.xi = nn.Parameter(torch.tensor(float(xi)))
        self.eta = nn.Parameter(torch.tensor(float(eta)))
        self.steps = steps
        self.num_chains = num_chains
        
        

    def initialize(self, s0):
        self.s0 = s0

    def simulate(self, expire):

        dt = 1.0 / self.steps
        total_steps = int(expire * self.steps)
        # Backpropogate a ?
        covariance_matrix = self.cov(self.a, self.steps)
        #print(covariance_matrix.dtype)
        dist = MultivariateNormal(loc=torch.tensor([0.0, 0.0]), 
                                  covariance_matrix=covariance_matrix)



        dw1 = dist.sample([self.num_chains, total_steps])
        dw2 = torch.randn([self.num_chains, total_steps]) * np.sqrt(dt)
        dB = self.rho * dw1[:, :, 0] + torch.sqrt(1 - self.rho**2) * dw2
        Y = self.Volterra_process(dw1, total_steps=total_steps)
        t = torch.linspace(0, expire, 1 + total_steps)[np.newaxis,:]
        V = self.xi * torch.exp(self.eta * Y - 0.5 * self.eta**2 * t**(2*self.a +1))
        # Construct non-anticipative Riemann increments
        increments = torch.sqrt(V[:,:-1]) * dB +(self.r- 0.5 * V[:,:-1]) * dt

        # Cumsum is a little slower than Python loop.
        integral = torch.cumsum(increments, axis = 1)

        S = torch.zeros_like(V)
        S[:,0] = self.s0
        S[:,1:] = self.s0 * torch.exp(integral)
        return S


    def Volterra_process(self, dW, total_steps):
        """
        Constructs Volterra process from appropriately
        correlated 2d Brownian increments.
        """
        Y1 = torch.zeros([self.num_chains, 1 + total_steps]) # Exact integrals
        Y2 = torch.zeros([self.num_chains, 1 + total_steps]) # Riemann sums

        # Construct Y1 through exact integral
        for i in np.arange(1, 1 + total_steps, 1):

            Y1[:,i] = dW[:,i-1,1] # Assumes kappa = 1

        # Construct arrays for convolution
        G = torch.zeros(1 + total_steps) # Gamma
        for k in torch.arange(2, 1 + total_steps, 1):
            G[k] = g(b(k, self.a)/self.steps, self.a)

        X = dW[:,:,0] # Xi

        # Initialise convolution result, GX
        GX = torch.zeros((self.num_chains, len(X[0,:]) + len(G) - 1))

        # Compute convolution, FFT not used for small n
        # Possible to compute for all paths in C-layer?
        for i in range(self.num_chains):
            GX[i,:] = torchaudio.functional.convolve(G, X[i,:])

        # Extract appropriate part of convolution
        Y2 = GX[:,:1 + total_steps]

        # Finally contruct and return full process
        Y = torch.sqrt(2 * self.a + 1) * (Y1 + Y2)
        return Y
    
    def cov(self, a, n):
        """
        Covariance matrix for given alpha and n, assuming kappa = 1 for
        tractability.
        """
        cov = np.array([[0.,0.],[0.,0.]])
        cov[0,0] = 1./n
        cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
        cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
        cov[1,0] = cov[0,1]
        
        return torch.tensor(cov, dtype=torch.float)


