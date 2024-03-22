''''
Created on Mar 24, 2020

@author: vogler
'''
import numpy as np
import matplotlib as mlp
mlp.use('Agg')
from lq import lq
from solutionFHN import FHN
import multiprocessing as mult
from numpy import linalg as LA
from numpy.linalg import solve
import matplotlib.pyplot as plt

def sadj(cT,wgate,ic,N,T,dt,ar,ad,Vrev,J,sigJ,la,Tmax,VT,a,b,c,X,yT,yR,I,sigex,delta,K,gamma):
  #N Numer of Monte-Carlo sim
    #T terminal time
    #dt 
    #X solution to the high dim particle system [N,M]
    #yT terminal ref profile (V,w,y)-|(V,w,y)-yT|
    #yR running ref profile (V,w,y)-|(V,w,y)-yR[:,t]|
    
    coupling=0
    
    if J>0:
        coupling=1
    
    
    M=int(T/dt) #number of time steps
    
    #mean membrane-potential and gating
    m0=np.zeros([M]) 
    m1=np.zeros([M]) 
 
    
    z=[]    #list for adjoint particles/samples

    #initialize adjoint state realizations via particles/samples
    for i in range(N):
        
        #initialize terminal condition for the ith particle/sample (adjoint particle i depends on X[i]!)
        dgT=np.zeros([3])
        dgT[0]=2*cT*gamma*(X[i][0,M-1]-yT[0]) #first component difference to desired terminal state (distance of its particle to the desired state...we want to control all particles)
        p=np.zeros([3,M])
        p[:,M-1]=dgT
        z.append(p)

       
    #initialize matrix coefficient for iteration
    A=np.zeros([3,3])
    A[0,1]=-1
    A[1,0]=c
    A[1,1]=-c*b
    
    #running cost derivative as coefficient for iteration
    df=np.zeros([3,1])  

    #mean for mean field drift term 1
    
    for k in range(N):
        m0=m0+X[k][0,:]
    m0=m0/float(N)
    
    for k in range(N):
        m1=m1+X[k][2,:]
    m1=m1/float(N)

    # for all time steps backwards
    for r in range(M-2,-1,-1):
        
        print(r)
        #mean for mean field term 2
        m2=0    
        for k in range(N):
            m2=m2-J*((X[k][0,r+1]-Vrev)*z[k][0,r+1])
        m2=m2/float(N)
        
        
    
        for i in range(N):
        
                    
            df[0,0]=2*gamma*(m0[r+1]-yR[0,r+1])    #only first component because we only want to control membrane potential

            #matrix coefficient
            A[0,0]=1-np.power(X[i][0,r],2)-J*m1[r+1]
            
            if coupling==1:
                A[2,0]=ar*((la*Tmax*np.exp(-la*(X[i][0,r]-VT)))/float(np.power(np.exp(-la*(X[i][0,r]-VT))+1,2)))*(1-X[i][2,r])
                A[2,2]=-ad-ar*(Tmax/float(1+np.exp(-la*(X[i][0,r]-VT))))
            
            
            #mean field drift term
            phi=np.zeros([3,1])
        
            phi[2]=m2
           
            #p1=z[i][0,r+1]
            #p2=z[i][1,r+1]
            #p3=z[i][2,r+1]
        
            #conditional expectation
            p=np.zeros([3,1])
            
            #conditional expectation
            p[0]=z[i][0,r+1]
            p[1]=z[i][1,r+1]
            p[2]=z[i][2,r+1]
      
            z[i][:,r]=(p+dt*np.transpose(A).dot(z[i][:,r+1]).reshape([3,1])+dt*df+dt*phi).reshape([3])
            #z[i][:,r]=solve(iden-dt*np.transpose(A),p+dt*df+dt*phi).reshape([3])

    

    return z
        