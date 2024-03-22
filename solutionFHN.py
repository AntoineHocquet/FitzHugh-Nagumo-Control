

import numpy as np
import matplotlib as mlp
mlp.use('Agg')
from numpy.linalg import solve
from scipy import optimize

def FHN(wgate,dt,T,N,a,b,c,I,sigex,J,sigJ,Vrev,ar,ad,Tmax,la,VT,x0,w,u):
    
    #T = final time
    #dt = discretization parameter for time
    #N = number of particles
    #V0,w0,y0 = mean vector for initial condititon for each particle (membrane potential, recovery, conductance). initial cond. will be chosen randomly normal dist. 
    #a,b,c,I,sigex = parameter of non coupled FitzHugh-Nagumo model
    #J,sigJ = synaptic weights
    #Vrev,ar,ad,Tmax,la,VT = parameter for synapse
    #Vrev = reversal potential for synaptic gates
    #ar,ad = transition rates for opening and closing of synaptic gates
    #la, VT =param. for sigmoid. correlation between potential and release of neurotransmitter
    
    #x0 initial condition sample
    #w noise sample
    
    #with coupling or without
    coupling=0
    
    #if coupling param>0
    if J>0:
        coupling=1

    
    #number of time steps
    M=int(T/dt) 

    #list of solutions for each particle
    z=[] 
    
    #Id matrix
    iden=np.identity(3)
    
        
    #initial conditions for each particle
    for i in range(N):      
        x=np.zeros([3,M])
        x[:,0]=x0[i]
        #x[:,0]=m
        z.append(x)
    
    
    #For each time r the state of the ith particle
    for r in range(1,M):   
      
        m0=0
        
        #mean-field interaction
        for k in range(N):
            m0=m0+z[k][2,r-1]

        m0=m0/float(N)
         
        for i in range(N):
            
                        
            
            def drift(y):
                
                #drift term with no interaction
                f=np.zeros([3,1])      
                f[0,0]=y[0]-(np.power(y[0],3)/float(3))-y[1]+I[0,r]
                f[1,0]=c*(y[0]+a-b*y[1])
                
                if coupling==1:
                    f[2,0]=ar*(Tmax/float(1+np.exp(-la*(y[0]-VT))))*(1-y[2])-ad*y[2]
    
    
                #mean-field drift term
                q=np.zeros([3,1])       
                
    
                q[0,0]=-J*(y[0]-Vrev)*m0
                
                return dt*f+dt*q
    
            #----------------------------------------------------------------------------------------------------------------------------------------------------
    
            def Jdrift(y):
                
                #jacobian of drift
                jac=np.zeros([3,3])
                
                jac[0,1]=-1
                jac[1,0]=c
                jac[1,1]=-c*b
                
                #matrix coefficient
                jac[0,0]=1-np.power(y[0],2)-J*m0
            
                if coupling==1:
                    jac[2,0]=ar*((la*Tmax*np.exp(-la*(y[0]-VT)))/float(np.power(np.exp(-la*(y[0]-VT))+1,2)))*(1-y[2])
                    jac[2,2]=-ad-ar*(Tmax/float(1+np.exp(-la*(y[0]-VT))))
                
                return iden-dt*jac
            
            #----------------------------------------------------------------------------------------------------------------------------------------------------
            
            #diffusion term with no interaction
            g=np.zeros([3,2])       
                
            g[0,0]=sigex
                
            #det parameter for turning coupling noise on or off
            if z[i][2,r-1] < 1 and z[i][2,r-1] > 0 and wgate==1:
                g[2,1]=(0.1*np.exp(-0.5/(1-np.power(2*z[i][2,r-1]-1,2))))*np.sqrt(ar*(Tmax/float(1+np.exp(-la*(z[i][0,r-1]-VT))))*(1-z[i][2,r-1])+ad*z[i][2,r-1])
                
            #----------------------------------------------------------------------------------------------------------------------------------------------------
                
            #mean-field diffusion term
            h=np.zeros([3,1])       
                
    
            h[0,0]=-sigJ*(z[i][0,r-1]-Vrev)*m0

            #----------------------------------------------------------------------------------------------------------------------------------------------------

            def funMin(y):
                
                func=y-((z[i][:,r-1]).reshape([3,1])+drift(y)+np.sqrt(dt)*g.dot(w[i][r-1])+np.sqrt(dt)*u[i][r-1]*h).reshape([3])
                
                return func
            
            #x0=np.ones([3])

            #sol=optimize.root(funMin, x0, jac=Jdrift, method='hybr')
            #z[i][:,r]=sol.x
            #z[i][:,r]=((z[i][:,r-1]).reshape([3,1])+drift(z[i][:,r-1])+np.sqrt(dt)*g.dot(w)+np.sqrt(dt)*u*h).reshape([3])
            z[i][:,r]=((z[i][:,r-1]).reshape([3,1])+drift(z[i][:,r-1])+np.sqrt(dt)*g.dot(w[i][r-1])+np.sqrt(dt)*u[i][r-1]*h).reshape([3])
       
    
    return z      