'''
Created on Mar 30, 2020

@author: vogler
'''
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
from numpy import linalg as LA
from scipy.optimize import minimize
from scipy.linalg import get_blas_funcs
import matplotlib.pyplot as plt

def lq(samplesX,ex):
    
    #ex samples evaluated at nodes
    
    #number samples and dimension of X and Y
    N,K = ex.shape

    
  
     
    #objective functional J to minimize (only take 180 samples)
    def J(alph):
        
        r=np.power(LA.norm(samplesX[:]-ex.dot(alph)),2)
        r=r/float(N)
        
   
        return r
    
    
    #Jacobian of functional J
    def Jjac(alph):
        
        gemn = get_blas_funcs("gemm", [np.transpose(ex),samplesX[:]-(ex.dot(alph))])
        jc=gemn(1, np.transpose(ex), samplesX[:]-(ex.dot(alph))) 

        jc=-2*jc/float(N)

        jc=jc.reshape([K])
        
        return jc
    
    #only 100 reference points
    x0=np.ones([K])
    res=minimize(J, x0, method='Newton-CG',jac=Jjac ,tol=0.0005)

  
    gemn = get_blas_funcs("gemm", [ex,res.x])
    #samples of conditional expectation
    sol=gemn(1, ex, res.x)  
    
    
    return sol



