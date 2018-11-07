#!/usr/bin/env python
from scipy import *
from tb_solver import *
from pylab import *

lat=[[sqrt(3),0],[sqrt(3)/2.,3/2.]]
orb=[[0,0],[-1/3.,2/3.]]
site_ene=[-1,1]
t=1.
t2=1.0*exp(1.j*pi/2.)
t2c=t2.conjugate()

kpath=[[0,0],[2./3,1./3],[1./3,2./3],[0,0]]; Nk=20

mymodel=tb_model(2,2,lat,orb,1)
mymodel.set_onsite(site_ene)

mymodel.set_hop(-t,0,1,[-1/3.,2/3.])
mymodel.set_hop(-t,0,1,[-1/3.,-1/3.])
mymodel.set_hop(-t,0,1,[2/3.,-1/3.])
mymodel.set_hop(t2,0,0,[1,0])
mymodel.set_hop(t2c,0,0,[0,1])
mymodel.set_hop(t2,0,0,[-1,1])
mymodel.set_hop(t2c,1,1,[1,0])
mymodel.set_hop(t2,1,1,[0,1])
mymodel.set_hop(t2c,1,1,[-1,1])
mymodel.display()

(kline,Evals)=mymodel.solve_kpath(kpath,Nk,True)
om,dos=mymodel.tetra_dos(4,400,100,0.02,True)

Nk=20
wf=wf_array(mymodel,Nk)
wf.wf_onmesh()
wf.chern_number(0)
show()

