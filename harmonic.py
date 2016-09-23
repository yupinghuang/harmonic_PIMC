#!/usr/bin/python3
# MCMC code for a SHO
import numpy as np

# slices is the number of time lattices
slices=np.int_(80)
ntrials=np.int_(100000)
# start and end position
x0=np.float_(0.0)
xm=np.float_(0.0)
#The dimensionless inverse temperature 
zeta=10
# step size for transition
d=np.sqrt(2)

class harmonic:

    def __init__(self,zeta):
        self.zeta = np.float_(zeta)

    '''
    The likelihood/weight function for a given path
    xarray: x_1 to x_m-1 coordinates
    '''
    def logweight(self,xarray):
        assert xarray.shape[0]==slices+2, 'incorrect input array size for the weight function'
        brace1 = 0.5*(xarray[0]**2+xarray[-1]*2) + np.sum(np.square(xarray[1:-1]))
        # sum_{m=1}^{M} (x_m-x_{m-1})^2.
        brace2 = np.sum(np.square(
            xarray[1:]-xarray[:-1]))
        return -0.5*((self.zeta/slices)*brace1 + (slices/self.zeta)*brace2)

    def weightDif(self,x1Array,x2Array):
        return np.exp(self.logweight(x2Array)-self.logweight(x1Array))

    '''
    Generate the candidate for the next path
    '''
    def nextpathCandidate(self,xarray):
        # using a uniform between -d and d; d chosen to be sqrt[2] as suggested by Creutz and Freedman
        index = np.random.randint(1,xarray.shape[0]-1)
        deltax = np.random.uniform(-d,d)
        nextXarray = np.copy(xarray)
        nextXarray[index] = nextXarray[index]+deltax
        return nextXarray


    '''
    Generate the next path using accept/reject
    '''
    def nextpath(self,xarray):
        nextXarray = self.nextpathCandidate(xarray)
        # accept/reject
        ratio = self.weightDif(xarray,nextXarray) 
        u = np.random.uniform()
        if ratio>u:
            return nextXarray
        else:
            return xarray

    '''
    '''
    def generatePaths(self):
        result = np.zeros((ntrials,slices+2),np.dtype(np.float64))
        i=1
        result[0] = np.linspace(x0,xm,slices+2,endpoint=True,retstep=False,dtype=np.dtype(np.float64))
        while(i<ntrials):
            result[i]=self.nextpath(result[i-1])
            i+=1
        return result

