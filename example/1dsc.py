#!/usr/bin/env python
from scipy import *
from tb_solver import *
from pylab import *
def Eknn(k):
	return -2*t*cos(k)

lat=[[1.0]]
orb=[[0]]
t=1.0
kpath=[[0],[1]]; Nk=50

mymodel=tb_model(1,1,lat,orb,1)
mymodel.set_hop(-t,0,0,[1])
mymodel.set_hop(-t,0,0,[2])
mymodel.set_hop(-t,0,0,[3])
mymodel.display()

(kline,Evals)=mymodel.solve_kpath(kpath,Nk,True)
om,dos=mymodel.tetra_dos(8,400,1000,0.01,True)
##Ekana=array([-2*t*cos(k[0]) for k in kline]) ##nn hopping
#Ekana=array([-2*t*cos(k[0])-2*t*cos(2*k[0]) for k in kline]) ##nn+nnn hopping
#Ekana=array([-2*t*cos(k[0])-2*t*cos(2*k[0])-2*t*cos(3*k[0]) for k in kline]) ##nn+nnn+nnnn hopping
show()

