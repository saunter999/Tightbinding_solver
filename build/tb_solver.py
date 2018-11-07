#!/usr/bin/env python
from scipy import *
import numpy as np
from pylab import *
import copy
from numpy import linalg as LA

"""
This TB_solver contain classes:
1) "tb_model"--main class describing tight-binding model.Inside the class, we also evaluate physical
quantities related to its eigenvalues only such as band dispersion in k space and density of states by 
diagonalzing matrix [H(k)]_orb*orb. Visualization of those quantites are also supported. 
2) "wf"--class used to compute physical quantites realated to its eigenvectors such as chern number 
of the bands (limited to 2D case only now).

only support nspin=1 now
"""

class tb_model:
	def __init__(self,dim_r,dim_k,lat=None,orb=None,nspin=1):

	    ##set dimension of lattice,we consider only dim_r in the range of [1,3]  
	    if type(dim_r).__name__!='int':
		raise Exception("\nFirst argument of tb_model class--dim_r should be an integer")
	    if dim_r < 1 or dim_r > 3:
            	raise Exception("\nFirst argument of tb_model class--dim_r out of range. Must be dim_r>0 and dim_r<4.")
	    self._dim_r=dim_r

	    ##set dimension of k space, dim_k<=dim_r 
	    if type(dim_k).__name__!='int':
		raise Exception("\nSecond argument of tb_model class--dim_k should be an integer")
	    if dim_k < 0 or dim_k > dim_r:
            	raise Exception("\nSecond argument of tb_model class--dim_k out of range. Must be dim_k>=0 and dim_k<=dim_r.")
	    self._dim_k=dim_k

	    ##set primitive lattice vectors of the bravias lattice in cartesian coordinate system
	    ##self._lat takes the form of array of dim_r*dim_r
	    ##format is (lat_vec_index,cartesian_index)
	    if lat == None:
		raise Exception("\nThird argument of tb_model class--lattice vectors are not given")
	    if type(lat).__name__ not in ['list','ndarray']:
		raise Exception("\nThird argument of tb_model class--lattice vectors not in the form of list or ndarray") 
	    self._lat=np.array(lat,dtype=float)
	    if self._lat.shape != (dim_r,dim_r):
		raise Exception("\nWrong format of input for the lattice vectors")

            ##check that volume is not zero and that have right handed system
            if dim_r>0:
                if np.abs(np.linalg.det(self._lat))<1.0E-6:
                   raise Exception("\nLattice vectors length/area/volume too close to zero, or zero.")
                if np.linalg.det(self._lat)<0.0:
                   raise Exception("\nLattice vectors need to form right handed system.")

	    ##set the positions of inequivalent atoms in primtive unit cell in reduced coordinate,i.e. in terms of lattice vectors.
	    ##self._orb takes the form of array of Norb*dimr
	    ##format is _orb(orb_index,lat_vec_index)
	    ##also we initialize self._norb = number of basis orbitals per primitve unit cell
	    if orb == None:
		raise Exception("\nFourth argument of tb_model class--positions of orbitals are not given")
	    if type(orb).__name__ not in ['list','ndarray']:
		raise Exception("\nFourth argument of tb_model class--positions of orbitals not in the form of list or ndarray") 
	    self._orb=np.array(orb,dtype=float)
#	    print self._orb.shape
	    if len(self._orb.shape) != 2:
		raise Exception("\nWrong orb array rank")
	    if self._orb.shape[1]!= dim_r:
                raise Exception("\nWrong orb array dimensions")
	    self._norb=self._orb.shape[0]
#	    print self._norb

	    ##set spin component
	    if nspin not in [1,2]:
                raise Exception("\nWrong value of nspin, must be 1 or 2!")
            self._nspin=int(nspin)

	    ##set additional quantities to be used by functions 
	    ##compute number of electronic states at each k-point or number of bands
            self._nsta=self._norb*self._nspin
		
	    ##set onsite energy
            if self._nspin==1:
	       self._site_enes=zeros((self._norb),dtype=float)
	    elif self._nspin==2:
	       "For nspin==2,onsite energy is not implemented yet!"
	
	    ##Initializing hopping to empty list
	    self._hoppings=[]
	    # The onsite energies and hoppings are not specified
            # when creating a 'tb_model' object.  They are speficied
            # subsequently by separate function calls defined below.

	    ##Calculating reciprocal lattice vector
	    self._klat=copy.copy(self._lat)
	    if dim_r==1:
		self._klat[0]=[2*pi/self._lat[0]] 
	    if dim_r==2:
		self._klat=np.dot(2*pi*np.identity(2),LA.inv( (self._lat).transpose() ) )
	    if dim_r==3:
		a2ca3=np.cross(self._lat[1],self._lat[2])	
		a3ca1=np.cross(self._lat[2],self._lat[0])	
		a1ca2=np.cross(self._lat[0],self._lat[1])	
		v=np.dot(self._lat[0],a2ca3)
		self._klat[0]=2*pi*a2ca3/v
		self._klat[1]=2*pi*a3ca1/v
		self._klat[2]=2*pi*a1ca2/v

	def set_onsite(self,site_enes):
	    if type(site_enes).__name__ not in ['list','ndarray']:
	       raise Exception("\nSite_energies have to be ginve in form of list of ndarray")
	    if len(site_enes)!=self._norb:
	       raise Exception("\nNumber of site energies is not equal to number of orbtials")
	    self._site_enes=site_enes

	def set_hop(self,hop_amp,orb1,orb2,hop_dir=None,mode='set',allow_conjugate_hop=False):
	    if allow_conjugate_hop == False:
#	        print "In present implementation,we only need to specify hopping in one direction, the conjugate hopping in opposite direction is takin into account implicitly."
		pass
	    else:
		raise Exception("\nUsing convention of conjugating hopping is Not supported ")
	    if hop_dir== None:
		raise Exception("\nHopping directions are not given")
	    if type(hop_dir).__name__ not in ['list','ndarray']:
		raise Exception("\nHopping directions are not in the form of list or ndarray") 
	    hop_dir=np.array(hop_dir,dtype=float)
#	    if len(hop_dir.shape) != 2:
#		raise Exception("\nWrong hop_dir array rank")
	    if hop_dir.shape[0]!= self._dim_r:
                raise Exception("\nWrong hop_dir array dimensions")

	    if type(hop_amp).__name__ not in ['float','float64','complex','complex128']:
		raise Exception("\nHopping amplitude must be the type of float or complex")
	    if type(orb1).__name__!='int' or type(orb2).__name__!='int':
		raise Exception("\nOrb index must be an integer")
	    if orb1>self._norb-1 or orb1<0:
		raise Exception("\n Orb index must between [0,norb-1]")
	    if orb2>self._norb-1 or orb2<0:
		raise Exception("\n Orb index must between [0,norb-1]")

	    if type(hop_amp).__name__=='float':
		print "real hopping amplitude in direction", hop_dir,'from','orb-',orb1,'to','orb-',orb2 
	    if type(hop_amp).__name__=='complex128':
		print "complex hopping amplitude in direction",hop_dir,'from','orb-',orb1,'to','orb-',orb2  

	    if mode == 'set':
	       self._hoppings.append([hop_amp,orb1,orb2,hop_dir]) 
	    if mode == 'reset':
	       self._hoppings=[]
	       self._hoppings.append([hop_amp,orb1,orb2,hop_dir]) 

	def solve_ham_onek(self,kpt,eig_vectors=False):
	    """
		Obtain the Hamiltonian matrix in k space at one k pint and then diagonalize the matrix to obtain its eigenvalue and eigenvectors(if eig_vectors ==True).
	    """
	    hamk=zeros((self._nsta,self._nsta),dtype=complex)
	    eigk=zeros(self._nsta,dtype=complex)
#	    print self._nsta
	    kpt=np.array(kpt,dtype=float)
	    if kpt.shape[0]!=self._dim_k:
		raise Exception("\nWrong dimension of k point")

	    for hop in self._hoppings:
		hopr=0.0
		for k in range(self._dim_r):
		    hopr += hop[-1][k]*self._lat[k] ##hopping vector in cartesian coordinate
	        kdotr=np.dot(kpt,hopr)
		
	        if self._nsta==1:
		   hamk[hop[1]][hop[2]] += ( hop[0]*exp(1j*kdotr) )+( hop[0]*exp(1j*kdotr)  ).conjugate()
	        else:
		   if hop[1]==hop[2]:
		       hamk[hop[1]][hop[2]] += ( hop[0]*exp(1j*kdotr)  )+ ( hop[0]*exp(1j*kdotr) ).conjugate()
		   else:	
		       hamk[hop[1]][hop[2]] += hop[0]* exp(1j*kdotr) 

            ##make hamk a hermittian matrix using upper triangle part of Hamk
            for i in range(self._nsta):
	          for j in range(i):
	              hamk[i][j]=hamk[j][i].conjugate()

	    ## Adding on_site energis into the diagonal part of Hamk
	    for i in range(self._nsta):
	        hamk[i][i] += self._site_enes[i]

	    if eig_vectors == False:
	       eigk=LA.eigvalsh(hamk)
	    elif eig_vectors == True:
	       eigk,v=LA.eigh(hamk)
	    for i in range(self._nsta):
	        if abs(eigk[i].imag)> 1e-6:
		   raise Exception("\nEigenvaule is not real,make sure that Hamiltonian is hermitian.")

	    if eig_vectors == False:
	        return eigk.real
	    elif eig_vectors == True:
	        idx=eigk.argsort()
                eigk=eigk[idx]
		v=v[:,idx]
	        return eigk.real,v

	def solve_kpath(self,kpath,Nk,visualize=False):
	    if type(kpath).__name__ not in ['list','ndarray']: 
	       raise Exception("\nKpath are not in the form of list or ndarray ")
	    kpath=np.array(kpath,dtype=float)
	    if kpath.shape[1]!=self._dim_r:
               raise Exception("\nWrong kpath array dimensions")

	    Nhymk=len(kpath) 
	    kline=[]
	    for i in range(Nhymk-1):
		inex=(i+1) % Nhymk
		Dir=kpath[inex]-kpath[i]
		for j in range(Nk):
		    kptred=kpath[i]+Dir*float(j)/Nk
		    kpt=0.0
		    for k in range(self._dim_r):
		        kpt+=kptred[k]*self._klat[k]
		    kline.append(kpt)
	    kpt=0.0
            for k in range(self._dim_r):
	        kpt+=kpath[-1][k]*self._klat[k] ##adding last point on the k path
	    kline.append(kpt)

	    Evals=array([self.solve_ham_onek(k) for k in kline])

	    Etop=max(Evals.transpose()[-1]) ##band top
	    Ebot=min(Evals.transpose()[0]) ##band bottom
	    if visualize == True:
	        clf()
		plot(range(len(kline)),Evals) ##plotting in python supports ploting of (x,y) in which x is (n,) array and y is (n,p) array
	        xticks(visible=False)
		for i in range(len(kpath)):
		   for j in range(self._dim_r):
		       kpath[i][j]=round(kpath[i][j],2) ##suppress decimal digits for plotting
		for i in range(len(kpath)):
		    axvline(x=i*Nk,ls='--',c='k')
		    text(i*Nk,Ebot-0.05*(Etop-Ebot),str(kpath[i]),size='large')
		ylabel('Ek',size='large')
	      
	    savefig("band_kpath.pdf")
	    return (kline,Evals)
	    
		
	def tetra_dos(self,L,Nw,Nk,dlt,visualize=False):
	    L=float(L);Nw=int(Nw);Nk=int(Nk);dlt=float(dlt)
	    om=linspace(-L,L,Nw)
	    dim=self._dim_r
	    epsk=zeros((self._nsta,Nk**dim),dtype=float)
	    g=zeros((self._nsta,Nw),dtype=complex)
	    ii=0
	    if dim==1:
	      for i in range(Nk):
		  kpt=i*self._klat[0]/(Nk+0.0)
		  epsk[:,ii]=self.solve_ham_onek(kpt)
		  ii +=1
	    if dim==2:
	      for i in range(Nk):
		  for j in range(Nk):
		      kpt=(i*self._klat[0]+j*self._klat[1])/(Nk+0.0)
		      epsk[:,ii]=self.solve_ham_onek(kpt)
		      ii +=1
	    if dim==3:
	      for i in range(Nk):
		  for j in range(Nk):
		      for k in range(Nk):
		           kpt=(i*self._klat[0]+j*self._klat[1]+k*self._klat[2])/(Nk+0.0)
		           epsk[:,ii]=self.solve_ham_onek(kpt)
		           ii +=1

	    for k in range(self._nsta):
   	        for ek in epsk[k]:
	          g[k] += 1./(om-ek+1j*dlt)
	    g=g/Nk**dim
	    g=g.transpose() 
	    if visualize == True:
	       clf()
	       plot(om,g.imag*(-1./pi))
	       xlabel(r"$\omega$",size='large')	 
	       ylabel("DOS",size='large')	 
	    savefig("tetra_dos.pdf")

	    return (om,g.imag*(-1./pi))

	def display(self):
	    r"""
	    Prints on the screen some information about this tight-binding
	    model. This function doesn't take any parameters.
	    """
	    print '---------------------------------------'
	    print 'report of tight-binding model' 
	    print '---------------------------------------' 
	    print 'r-space dimension           =',self._dim_r 
	    print 'k-space dimension           =',self._dim_k 
	    print 'number of spin components   =',self._nspin 
	    print 'number of orbitals          =',self._norb 
	    print 'number of bands(or of electronic states per k point)=',self._nsta 
	    print 'lattice vectors:' 
	    for i,o in enumerate(self._lat):
	        print " #", i ," ===> ",o
	    print 'reciprocal lattice vectors:' 
	    for i,o in enumerate(self._klat):
	        print " #", i ," ===> ",o
	    print 'positions of orbitals in reduced coordinate:'
	    for i,o in enumerate(self._orb):
	        print " #", i ," ===> ",o
	    print "site energies are:"
	    print self._site_enes
	    print "hoppings are:"
	    print self._hoppings


class wf_array:
	def __init__(self,model,Nk): 
	    self._model=model
	    self._Nk=Nk
	    self._kmesh=zeros((Nk,Nk,self._model._dim_k),dtype=float)
	    self._wf=zeros((Nk,Nk,self._model._nsta,self._model._nsta),dtype=complex)

	def wf_onmesh(self):
	    if self._model._dim_k==2:
		for i in range(self._Nk):
		    for j in range(self._Nk):
			k = (i*self._model._klat[0]+j*self._model._klat[1])/(self._Nk+0.0)
	                self._kmesh[i,j,:] = k
		        eigk,self._wf[i,j,:,:]=self._model.solve_ham_onek(k,True) 

	def chern_number(self,nband):
	    if type(nband).__name__ != 'int':
	       raise Exception("\nBand index has to be an integer")
	    if nband not in range(self._model._nsta):
	       raise Exception("\nBand index is out of range")
	    Nk=self._Nk
	    fs=zeros((Nk,Nk),dtype=complex)
	    cn=0.0
	    for i in range(Nk):
               ir=(i+1) % Nk
	       for j in range(Nk):
		  ju=(j+1) % Nk
		  inprd=np.vdot(self._wf[i,j,:,nband],self._wf[ir,j,:,nband])
		  U1x=inprd/abs(inprd)
		  inprd=np.vdot(self._wf[ir,j,:,nband],self._wf[ir,ju,:,nband])
		  U2y=inprd/abs(inprd)
		  inprd=np.vdot(self._wf[ir,ju,:,nband],self._wf[i,ju,:,nband])
		  U3x=inprd/abs(inprd)
		  inprd=np.vdot(self._wf[i,ju,:,nband],self._wf[i,j,:,nband])
		  U4y=inprd/abs(inprd)
		  fs[i,j]=log(U1x*U2y*U3x*U4y)
		  cn += fs[i,j].imag
	    cn *= 1./(2*pi)
	    return cn

