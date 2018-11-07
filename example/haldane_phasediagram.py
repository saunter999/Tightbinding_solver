#!/usr/bin/env python
from scipy import *
from tb_solver import *
from pylab import *

lat=[[sqrt(3),0],[sqrt(3)/2.,3/2.]]
orb=[[0,0],[-1/3.,2/3.]]
mymodel=tb_model(2,2,lat,orb,1)
t=1.


Nph=50;Nm=100
phils=linspace(-pi,pi,Nph);
mb=6.
mosls=linspace(-mb,mb,Nm)*t
cn=zeros((Nph,Nm))
Nk=20

for iph,phi in enumerate(phils):
    for im,mos in enumerate(mosls):
	print "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	print "m=",mos,'phi=',phi
	t2=1.0*exp(1.j*phi)
	t2c=t2.conjugate()
        site_ene=[-mos,mos]
	mymodel.set_onsite(site_ene)
	mymodel.set_hop(-t,0,1,[-1/3.,2/3.],'reset')
	mymodel.set_hop(-t,0,1,[-1/3.,-1/3.],'set')
	mymodel.set_hop(-t,0,1,[2/3.,-1/3.],'set')
	mymodel.set_hop(t2,0,0,[1,0],'set')
	mymodel.set_hop(t2c,0,0,[0,1],'set')
	mymodel.set_hop(t2,0,0,[-1,1],'set')
	mymodel.set_hop(t2c,1,1,[1,0],'set')
	mymodel.set_hop(t2,1,1,[0,1],'set')
	mymodel.set_hop(t2c,1,1,[-1,1],'set')
#	mymodel.display() 

	wf=wf_array(mymodel,Nk)
	wf.wf_onmesh()
	cn[iph,im]=wf.chern_number(0)
imshow(cn.transpose(), interpolation='nearest',origin='lower',extent=[-pi,pi,-mb,mb],aspect='equal')
colorbar()
plot(phils,3*sqrt(3)*sin(phils),'k--',lw=2)
plot(phils,-3*sqrt(3)*sin(phils),'k--',lw=2)
tight_layout()
savefig("hald_chernum.png")
show()

