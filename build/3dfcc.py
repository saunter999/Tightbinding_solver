#!/usr/bin/env python
from scipy import *
from tb_solver import *
from pylab import *


lat=[[0.0,0.5,0.5],[0.5,0.0,0.5],[0.5,0.5,0]]
orb=[[0,0,0]]
t=1./(4.*sqrt(3.))


kpath=[[0,0,0],[0.5,0,0],[0.5,0.5,0.5],[0,0,0]]; Nk=10
mymodel=tb_model(3,3,lat,orb,1)
mymodel.set_hop(-t,0,0,[1,0,0]) ## there are 12 nearest neighbors so we have 6 inequivalent hoppings
mymodel.set_hop(-t,0,0,[0,1,-1])
mymodel.set_hop(-t,0,0,[0,1,0])
mymodel.set_hop(-t,0,0,[1,0,-1])
mymodel.set_hop(-t,0,0,[0,0,1])
mymodel.set_hop(-t,0,0,[1,-1,0])


mymodel.display()

(kline,Evals)=mymodel.solve_kpath(kpath,Nk,True)
om,dos=mymodel.tetra_dos(4,400,30,0.03,True)
show()

