#!/usr/bin/env python
from scipy import *
from tb_solver import *
from pylab import *
lat=[[1.0,0],[0.,1.]]
orb=[[0,0]]
t=0.25
kpath=[[0,0],[0.5,0],[0.5,0.5],[0,0]]; Nk=20

mymodel=tb_model(2,2,lat,orb,1)
mymodel.set_hop(-t,0,0,[1,0])
mymodel.set_hop(-t,0,0,[0,1])
mymodel.display()
(kline,Evals)=mymodel.solve_kpath(kpath,Nk,True)
om,dos=mymodel.tetra_dos(2,400,100,0.01,True)


show()

