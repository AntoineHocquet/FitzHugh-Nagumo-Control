'''
Created on Jan 9, 2020

@author: vogler
'''
import os
import pickle
import numpy as np
import matplotlib as mlp
import time
mlp.use('Agg')
from numpy import linalg as LA
from adjFHNdet import sadj
import matplotlib.pyplot as plt
from solutionFHN import FHN
import multiprocessing as mult

def step(cT,M,s,g,dlast,glast,Iprev,wgate,ic,dt,T,N,a,b,c,sigex,J,sigJ,Vrev,ar,ad,Tmax,la,VT,gamma,la2,yR,yT,Jpre,w,u,x0):
    
    #x0 sample initial condition
    
    #w,u noise realizations
    
    #parameters for initial distribution
    m=np.zeros([3])
    
    m[0]=ic[0,0]
    m[1]=ic[1,0]
    m[2]=ic[2,0]

    
    j,laenge = ic.shape
    
    #calculate new control mid by direction
    Inew=np.zeros([1,M])
    
    #Fletcher reeves
    #beta=g.dot(g)/float(glast.dot(glast))
    beta=0
    #new direction
    d=-g+beta*dlast
    
    Inew[0,:]=Iprev[0,:]+s*d
    
    for r in range(M):
        if Inew[0,r]>0.3:
            Inew[0,r]=0.3
            
        if Inew[0,r]<-0.3:
            Inew[0,r]=-0.3
    
    
    #calculate solution for corresponding step size mod. control
    sampleX=FHN(wgate,dt,T,N,a,b,c,Inew,sigex,J,sigJ,Vrev,ar,ad,Tmax,la,VT,x0,w,u)

    
    #local field potential
    m0=np.zeros([M])
    
    #mean

    for k in range(N):
        m0=m0+sampleX[k][0,:]
    m0=m0/float(N)
            
    #new cost
    Jnew=0
    cost1=0
    cost2=0
    #calculate new cost
    for k in range(N):
        Jnew=Jnew +dt*gamma*np.power(LA.norm(m0[:]-yR[0,:]),2)+gamma*cT*np.power(LA.norm(sampleX[k][0,M-1]-yT[0]),2)+dt*la2*np.power(LA.norm(Inew[0,:]),2)
        cost1=cost1+dt*gamma*np.power(LA.norm(m0[:]-yR[0,:]),2)+gamma*cT*np.power(LA.norm(sampleX[k][0,M-1]-yT[0]),2)
        cost2=cost2+dt*la2*np.power(LA.norm(Inew[0,:]),2)
    Jnew = Jnew/float(N)
    cost1=cost1/float(N)
    cost2=cost2/float(N)
    
    print('cost1')
    print(cost1)
    print('cost2')
    print(cost2)
        
        
                
    #check if there is better step size
    if Jnew<Jpre or s<0.000001:
                    
        
        out=[]
                  
        out.append(Jnew)
        out.append(Inew)
        out.append(sampleX)
        out.append(d)
        
        
    else:
                
        out=[]
        
        out.append(-1)
        out.append(Jnew)
    
    return out

def coptCl(cT,wgate,ic,N,T,dt,a,b,c,Iinit,J,sigJ,Vrev,ar,ad,Tmax,VT,sigex,K,delta,yR,yT,eps,la,la2,gamma):
    
    #parameters for initial distribution
    m=np.zeros([3])
    
    m[0]=ic[0,0]
    m[1]=ic[1,0]
    m[2]=ic[2,0]

    
    j,laenge = ic.shape
    
    
    #count how many accept
    reset=0
    
    #after 10 successful iterations, reset step size 
    resetStep=0

    #lower step size by...
    ratea=1
    rateb=2

    #number of time steps
    M=int(T/dt) 

    #local field potential
    m0=np.zeros([M])

    #cost at each step
    cost=[]
    
    
    #initialize for plots
    count=0
    
    x = np.arange(0, 200,0.1) 
    x1 = np.arange(0, 200-0.2,0.1)   #time x-axis 

    
    #Failmarker if no step size was accepted
    fmark=-1

    #initial step size for gradient decent
    s=float(1)
    
    #gradient 
    g=np.zeros([M])
    
    print('calculating initial cost')
    #Iinit=0.2*np.ones([1,M])
    Iprev=Iinit

    #sample initial condition (initial conditions for each particle)
    
    x0=[]
    
    for i in range(N):      
        ini=ic[:,np.random.randint(0,laenge -1)]
        #x[:,0]=m
        x0.append(ini)
    
    #sample noise
    
    w=[]
    u=[]
    
    
    
    for i in range(N):
        #noise w
        k=[]
        #noise u
        l=[]
        for r in range(1,M):
            #external noise term
            y=np.random.normal(0,1,[2,1])    
              
            #synaptic noise 
            o=np.random.normal(0,1) 
            
            k.append(y)
            l.append(o)
        
        w.append(k)
        u.append(l)
    
    #initial samples of state eq. for intial control and 0 gradient 
    sampleX=FHN(wgate,dt,T,N,a,b,c,Iinit,sigex,J,sigJ,Vrev,ar,ad,Tmax,la,VT,x0,w,u)
    
    #mean
    for k in range(N):
        m0=m0+sampleX[k][0,:]
    m0=m0/float(N)
    
    #calculate initial cost
    Jin=0
    
    #Monte Carlo for initial cost over all particles(samples)
    for i in range(N):
        Jin=Jin+gamma*dt*np.power(LA.norm(m0[:]-yR[0,:]),2)+gamma*cT*np.power(LA.norm(sampleX[i][0,M-1]-yT[0]),2)+dt*la2*np.power(LA.norm(Iprev[0,:]),2)
    Jpre=Jin/float(N)
    
    print('initial cost')
    print(Jpre)
    
    #cost with initial control
    cost.append(Jpre)
    
    print('calculate initial gradient')
    
    #initial adjoint sampels
    p=sadj(cT,wgate,ic,N,T,dt,ar,ad,Vrev,J,sigJ,la,Tmax,VT,a,b,c,sampleX,yT,yR,Iprev,sigex,delta,K,gamma)
  
    #initial gradient
    for k in range(N):
        g[:]=g[:]+p[k][0,:]+2*la2*Iprev[0,:]
  
    g[:]=g[:]/float(N)

    
    #set last gradient and last direction (initial)
    glast=g
    dlast=np.zeros([M])
    
    print('complete')
    
    for i in range(N):
        plt.plot(x,sampleX[i][0,:],'r')
                
    plt.savefig('initstates.png')
    plt.close()
    
    plt.plot(x,sampleX[0][0,:],'r')
    plt.savefig('initialsingleState.png')
    plt.close()
    
    plt.plot(x,m0,'r')
    plt.ylim(-2,2)
    plt.savefig('localFieldPotential.png')
    plt.close()
    
    for i in range(N):
        plt.plot(x,p[i][0,:],'r')
    plt.savefig('initadjs.png')
    plt.close()
    
    
    
    #initial gnorm
    gnorm=dt*LA.norm(g)
    
    plt.plot(x,-g[:],'r')
    plt.savefig('initgrad.png')
    plt.close()

 
    #accepted
    acep=1
    
    
    cstep=s
    
    #sample initial and noise for first iteration
    
    #sample initial condition (initial conditions for each particle)
    
    x0=[]
    
    for i in range(N):      
        ini=ic[:,np.random.randint(0,laenge -1)]
        #x[:,0]=m
        x0.append(ini)
    
    #sample noise
    
    w=[]
    u=[]
    
    
    
    for i in range(N):
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
    
    #while norm is beyond threshold
    while gnorm>eps:
        
        if Jpre<20:
            N=1000
        
        print('current gradient norm')
        print(gnorm)
        Jdiff=-1
        
        while Jdiff<0:
        
    
            
            if acep==1 and not count==0:
                acep=0
                plt.title('current step size: ' + str(cstep) + ', current cost: ' + str(Jpre), fontdict=None, loc='center')
                plt.plot(x,-g[:],'r')
                plt.savefig( str(count-1) + 'grad.png')
                plt.close()

            
            
            new=step(cT,M,s,g,dlast,glast,Iprev,wgate,ic,dt,T,N,a,b,c,sigex,J,sigJ,Vrev,ar,ad,Tmax,la,VT,gamma,la2,yR,yT,Jpre,w,u,x0)
            
           
            if new[0]<Jpre and not new[0]==-1 or s<0.0000001:
                
                    
                Jpre=new[0]
                Iprev=new[1]
                sampleX=new[2]
                fmark=1
                    
                #update last direction
                dlast=new[3]
                    
                print('step size accepted:' + str(s))
                    
            else:
                    
                if new[0]==-1:
                    print('larger then prev value')
                    print(new[1])
                 
       
            #if new cost was accepted at some step size  
            if fmark==1:
                
                #control successful accepted
                resetStep=resetStep+1
                
                if resetStep>10 or  s<0.000001:
                    
                    #reset step size
                    s=1
                    
                    resetStep=0
                
                #reset res counter
                reset=0
                
                #save last gradient
                glast=g
                
                #add current new cost
                cost.append(Jpre)
                
                #plot the result    
                acep=1  
    
                print('calculating new gradient')
                
                #calculate new gradient
                g=np.zeros([M])
                
                p=sadj(cT,wgate,ic,N,T,dt,ar,ad,Vrev,J,sigJ,la,Tmax,VT,a,b,c,sampleX,yT,yR,Iprev,sigex,delta,K,gamma)
                    
                for i in range(N):
                    plt.plot(x,p[i][0,:],'r')
                plt.savefig( str(count) + 'adjs.png')
                plt.close()
           
                    
                for k in range(N):
                    g[:]=g[:]+p[k][0,:]+2*la2*Iprev[0,:]
                            
                g[:]=g[:]/float(N)
           
                
                    
                print('complete')
                
                    
                gnorm=dt*LA.norm(g)
                                    
                Jdiff=0
                
                #plot current control
                plt.plot(x1,Iprev[0,0:M-2],'blue')
                plt.savefig( str(count) + 'control.png')
                plt.close()
                
                #plot current states
                for i in range(N):
                    plt.plot(x,sampleX[i][0,:],'r')
                plt.savefig( str(count) + 'states.png')
                plt.close()
                
                
                plt.plot(x,sampleX[0][0,:],'r')
                plt.savefig( str(count) + 'singleState.png')
                plt.close()
                
                
                #local field potential
                m0=np.zeros([M])
                
                for k in range(N):
                    m0=m0+sampleX[k][0,:]
                m0=m0/float(N)
                
                plt.plot(x,m0,'r')
                plt.savefig( str(count) + 'localFieldPotential.png')
                plt.close()
                
                count=count+1
    
             
                #reset
                fmark=-1
                
                print('current cost')
                print(Jpre)
                print('current step size')
                print(s)
                
                #sample for next iteration
                #sample initial condition (initial conditions for each particle)
    
                x0=[]
                
                for i in range(N):      
                    ini=ic[:,np.random.randint(0,laenge -1)]
                    #x[:,0]=m
                    x0.append(ini)
                
                #sample noise
    
                w=[]
                u=[]
                
                
                
                for i in range(N):
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
                
            else:
            
                reset=reset+1
                #lower step size
                s=(ratea/float(rateb))*s
                
                print(s)

    print(gnorm)
    return Iprev

