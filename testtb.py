#!/usr/bin/env python
from scipy import *
from tb_solver import *
lat=[[1,0],[0,2]]
orb=[[0,0],[0.5,0.5]]
#lat=[[1]]
#orb=[[0]]
mymodel=tb_model(2,2,lat,orb,1)
#mymodel=tb_model(1,1,lat,orb,1)
mymodel.display()
##mymodel=tb_model(1,1)

