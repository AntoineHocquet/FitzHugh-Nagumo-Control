'''
Created on Apr 15, 2020

@author: vogler
'''

import numpy as np
from solutionFHN import FHN
import matplotlib.pyplot as plt
from GDstoch import coptCl
from solutionFHNdet import FHNdet

if __name__ == '__main__':

    #parameter for state modell
        
    #time discretization
    dt=0.1
            
    #final time
    T=200 
    M=int(T/dt) 
            
    V0=-0.8275021695916729
    w0=-0.1391607698173808
    y0=0.589165868968053
    
            
    sigex=0.04

            
    a=0.7
    b=0.8
    c=0.08
            
    #synapse i.e. parameter defining reversal potential and 
    #param for open channels for synaptic input
            
    #excitatory synapses since rev. potential higher then rest. potetial
    Vrev=1.2
            
    #fast excitatory conductance i.e. fast activation and deactivation 
    ar=1
    ad=0.3
    Tmax=1
    la=0.1
    
    #threshold for presyn. neuron for opening of synaptic gates to postsyn. neuron
    VT=2
            
    #Number of neurons in the network
    N=500
            
    #number of nodes for approximation of conditional expectation
    K=10
            
    #coupling strength
    J=0.46
            
    #turning diffusion approximation for synaptic gates on/off
    wgate=0
            
    #stoch. coupling
    sigJ=0
        
        
    #-------------------------------------------------------------------------------------------------------------------
    #initializing distribution on limit cycle
    
    
    V1=-1.19940804
    w1=-0.62426004
    y1=0.58373236
        
    #initial condition
    ic = np.zeros([3,M])
        
    ic[0,:] = V0*np.ones([M])
    ic[1,:] = w0*np.ones([M])
    ic[2,:] = y0*np.ones([M])
    
    ic2 = np.zeros([3,M])
        
    ic2[0,:] = V1*np.ones([M])
    ic2[1,:] = w1*np.ones([M])
    ic2[2,:] = y1*np.ones([M])
        
    I2=0*np.ones([1,int(T/dt)])
    
    I2[0,300]=0
        
    N2=1
        
    #limit cycle
    phase=FHNdet(wgate,ic,dt,T,N2,a,b,c,I2,sigex,0.46,sigJ,Vrev,ar,ad,Tmax,la,VT)
    phase2=FHNdet(wgate,ic2,dt,T,N2,a,b,c,I2,0,0.46,sigJ,Vrev,ar,ad,Tmax,la,VT)
        
    x = np.arange(0, 100,0.1)
    
    plt.plot(phase[0][0,:])
    plt.savefig('phase')
    plt.close()
    
    #-------------------------------------------------------------------------------------------------------------------
    #solution for initial condition on limit cycle
    
    x = np.arange(0, 200,0.1)
    
    #external current
    I3=0*np.ones([1,int(T/dt)])
    I3[0,0:70]=0.8
    #I3[0,0:10]=1
    #I3[0,200:220]=0.3
    #I3[0,300:320]=0.35
    #I3[0,400:420]=0.355
    #I3[0,500:520]=0.365
    #I3[0,600:620]=0.365
    #I3[0,700:720]=0.375
    #I3[0,800:820]=0.375
    #I3[0,900:920]=0.38

    #I3=0.5*np.ones([1,int(T/dt)])
    
    L=500
    
    I=0*np.ones([1,int(T/dt)])
    
    #parameters for initial distribution
    m=np.zeros([3])
    
    m[0]=ic[0,0]
    m[1]=ic[1,0]
    m[2]=ic[2,0]

    
    j,laenge = phase[0].shape
    
    #sample noise and initial condition
    #sample initial condition (initial conditions for each particle)
    
    x0=[]
    
    for i in range(L):      
        ini=phase[0][:,np.random.randint(0,laenge -1)]
        #x[:,0]=m
        x0.append(ini)
    
    #sample noise
    
    w=[]
    u=[]
    
    
    
    for i in range(L):
        #noise w
        k=[]
        #noise u
        l=[]
        for r in range(M):
            #external noise term
            y=np.random.normal(0,1,[2,1])    
              
            #synaptic noise 
            o=np.random.normal(0,1) 
            
            k.append(y)
            l.append(o)
        
        w.append(k)
        u.append(l)
        
        
    sol=FHN(wgate,dt,T,L,a,b,c,I3,sigex,0.46,sigJ,Vrev,ar,ad,Tmax,la,VT,x0,w,u)
    sol1=FHN(wgate,dt,T,L,a,b,c,I,0,0,sigJ,Vrev,ar,ad,Tmax,la,VT,x0,w,u)
    
    for k in range(L):
    
        plt.plot(x,sol1[k][0,:],'r')
    
    plt.ylim(-2,2)
    plt.savefig('networkFHN')
    plt.close()
    
    g0=np.zeros([M])
    
    for k in range(L):
    
        g0=g0+sol[k][2,:]
        plt.plot(x,sol1[k][2,:],'b')
        
    plt.savefig('gateFHN')
    plt.close()
    
    g0=g0/float(N)
    
    plt.plot(x,g0,'b')
    plt.savefig('agateFHN')
    plt.close()
    
    #local field potential
    m0=np.zeros([M])
    m1=np.zeros([M])
    
    for k in range(L):
        m0=m0+sol[k]
        m1=m1+sol1[k]
    m0=m0/float(L)
    m1=m1/float(L)
    
    plt.plot(x,m0[0,:],'r')
    plt.ylim(-2,2)
    plt.savefig('localFieldPotential.png')
    plt.close()
    
    plt.plot(x,m1[0,:],'r')
    plt.ylim(-2,2)
    plt.savefig('m1.png')
    plt.close()
    
    #for k in range(N):
    #    plt.plot(x,sol[k][0,:],'r')
    #    plt.plot(x,sol[k][1,:],'b')
    #    plt.plot(x,sol[k][2,:],'green')
    #plt.savefig('sol.png')
    #plt.close()
    
    delta=0.02
    eps=0.000000000001
    gamma=1
    la2=0.00
    
    Iex=-1*np.ones([3,M])
    
    ref=np.zeros([3,M])
    ref[:,0:int(M/float(2.05))]=m0[:,0:int(M/float(2.05))]
    #ref=ref1[0]
    ref[:,int(M/float(2.05)):M]=m1[:,int(M/float(2.05)):M]
    #ref[:,int(M/float(1.78)):M]=m1[:,int(M/float(1.78)):M]

 
    yR=ref
    yT=yR[:,M-1]
    
    cT=0
    
    x = np.arange(0, 200,0.1)
    plt.plot(x,yR[0,:],'r')

    plt.savefig('ref.png')
    plt.close()
    
    
    
    op=coptCl(cT,wgate,phase[0],N,T,dt,a,b,c,I,J,sigJ,Vrev,ar,ad,Tmax,VT,sigex,K,delta,yR,yT,eps,la,la2,gamma)