import scipy.optimize as opt
import numpy as np

hbar = 197.326 # MeV fm when speed of ligth is 1

d = np.array([0,0,1])
L_lattice_size = 10.0 # fermis
L = L_lattice_size/hbar # MeV^-1
m1 = 146.0 #MeV
d2 = float(np.dot(d,d))
mu2 = ((L*m1)/(2*np.pi))**2

n = [0,0,1]

# Dot product of vector d with n triplets
if d2 == 0:
	d_n_dot = np.dot(d,n)
else:
	d_n_dot = np.dot(d,n)/d2

# Parallel component of the n [(d dot n) / (d2) * d]

n_par = d_n_dot * d

def f(x2):
	gamma = 0.5 * np.sqrt(4+d2/(x2+mu2))
	rn = 1/gamma*(n_par-0.5*d)+n-n_par
	rn2 = np.dot(rn,rn)
	a = rn2-x2
	return a

print('%.15f' % opt.fsolve(f,1)) 
