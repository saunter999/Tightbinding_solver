#!/usr/bin/env python
from scipy import *
from tb_solver import *
from pylab import *


lat=[[1,0,0],[0,1,0],[0,0,1]]
orb=[[0,0,0]]
t=1.0
kpath=[[0,0,0],[0.5,0,0],[0.5,0.5,0.5],[0,0,0]]; Nk=10
mymodel=tb_model(3,3,lat,orb,1)
mymodel.set_hop(-t,0,0,[1,0,0])
mymodel.set_hop(-t,0,0,[0,1,0])
mymodel.set_hop(-t,0,0,[0,0,1])
mymodel.display()
(kline,Evals)=mymodel.solve_kpath(kpath,Nk,True)
om,dos=mymodel.tetra_dos(6,100,20,0.01,True)
show()
