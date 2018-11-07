#!/usr/bin/env python
from scipy import *
from tb_solver import *
from pylab import *

lat=[[sqrt(3),0],[sqrt(3)/2.,3/2.]]
orb=[[0,0],[-1/3.,2/3.]]
t=1.
kpath=[[0,0],[2./3,1./3],[1./3,2./3],[0.5,0.5],[0,0]]; Nk=20

mymodel=tb_model(2,2,lat,orb,1)
mymodel.set_hop(-t,0,1,[-1/3.,2/3.])
mymodel.set_hop(-t,0,1,[-1/3.,-1/3.])
mymodel.set_hop(-t,0,1,[2/3.,-1/3.])
mymodel.display()
(kline,Evals)=mymodel.solve_kpath(kpath,Nk,True)
om,dos=mymodel.tetra_dos(4,400,100,0.02,True)

show()

