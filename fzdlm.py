# -----------------
# Import packages
import pickle
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
from scipy import special
from scipy import integrate

# -----------------
# Config file:
config_file = str(sys.argv[1])

with open(config_file, 'r') as f:
	for line in f:
		if line[0:6] == 'folder':
			folder = line[7:-1]
		if line[0:9] == 'cube num ':
			cube_num = int(line[9:])
		if line[0] == 'L':
			L_lattice_size = float(line[2:])
		if line[0] == 'l':
			l = int(line[2:])
		if line[0] == 'm':
			m = int(line[2:])
		if line[0:2] == 'dx':
			dx = int(line[2:])
		if line[0:2] == 'dy':
			dy = int(line[2:])
		if line[0:2] == 'dz':
			dz = int(line[2:])
		if line[0:5] == 'x2min':
			x2min = float(line[6:])	
		if line[0:5] == 'x2max':
			x2max = float(line[6:])
		if line[0:6] == 'points':
			points = int(line[7:])

d = [dx,dy,dz]

print 'F function with (l,m) = (' + str(l) + ','+ str(m) +')\n d = ' + str(d) 
print 'From '+ str(x2min)+ ' to ' +str(x2max)+ ' ' +str(points)+' points'

# -----------------
# Location to get triplets from:
trip_folder = 'felipe_results/'

# -----------------
# Define constants
I = complex(0.0,1.0)
hbar = 197.326 # MeV fm when speed of ligth is 1
eps = 2.22044604925e-16 # machine epsilon

# -----------------
# Define variables of the Z function
# d= [1.0,1.0,0.0]
#L_lattice_size = 10.0 # fermis
L = L_lattice_size/hbar # MeV^-1
lab_moment = (2*np.pi/L)*np.array(d)
lab_moment2 = np.dot(lab_moment,lab_moment)
m1 = 146.0 # MeV
m2 = 146.0 # MeV

# -----------------
# Define integrands functions for second and third term calculation
def integr_second_term(t,x2):
	return np.exp(t*x2)/np.sqrt(t)

def integr_third_term_r(t,l,gw2,x2,coef):
	integr = np.power(np.pi/t,3.0/2.0+l)*np.exp(t*x2-np.square(np.pi)*gw2/t)
	return np.real(sum(coef*integr))

def integr_third_term_i(t,l,gw2,x2,coef):
	integr = np.power(np.pi/t,3.0/2.0+l)*np.exp(t*x2-np.square(np.pi)*gw2/t)
	return np.imag(sum(coef*integr))

# -----------------
# Calculate triplets used in the sums of the first and third term

# ---- First term triplets -----
# Get the array of n triplets
filename = '../' + trip_folder + 'triplets/n_list_r<' + str(cube_num) + '.txt'
f = open(filename, 'r')
n_list = pickle.load(f)
n_arr = np.array(n_list)

# Create an array of d vector copies, and calculate its magnitude
d_arr = np.repeat([np.array(d)],len(n_arr),axis=0)
d2 = float(d[0]**2+d[1]**2+d[2]**2)

# Dot product of vector d with n triplets
if d2 == 0:
	d_n_dot = np.inner(d,n_arr)
else:
	d_n_dot = np.inner(d,n_arr)/d2

# Create array of the paralel component of the n triplets [(d dot n) * d]
n_par_arr = np.transpose(np.append([d_n_dot],[d_n_dot,d_n_dot],axis=0)) * d_arr

# Create array of the perpendicular component
n_perp_arr = n_arr - n_par_arr

# ---- Third term triplets -----
# Get the array of w triplets
filename = '../' + trip_folder + 'triplets/w_list_r<' + str(cube_num) + '.txt'
f = open(filename, 'r')
w_list = pickle.load(f)
w_arr = np.array(w_list)

# Take out one element, summation not in w=0
d_arr_for_w = np.delete(d_arr, 0,0)

# Dot product / d2, of vector d with w triplets
if d2 == 0:
	d_w_dot = np.inner(d,w_arr)
else:
	d_w_dot = np.inner(d,w_arr)/d2

# Create array of the paralel component of the w triplets [(d dot w) / (d2) * d]
w_par_arr = np.transpose(np.append([d_w_dot],[d_w_dot,d_w_dot],axis=0)) * d_arr_for_w

# Create array of the perpendicular component
w_perp_arr = w_arr - w_par_arr

# Get the poles of Z (identical particles)
mu2 = np.power(L*m1/(2*np.pi),2.)
def get_pole(n):
	first_part = -mu2 + np.dot(n,n)-np.dot(n,d)
	radicand = (mu2+np.dot(d-n,d-n))*(mu2+np.dot(n,n))
	pp = 0.5*(first_part+np.sqrt(radicand))
	return pp

raw_poles = np.zeros(4*4*4)

count = 0
for nz in xrange(4):
	for ny in xrange(4):
		for nx in xrange(4):
			raw_poles[count] = get_pole(np.array([nx,ny,nz]))
			count += 1

poles_sorted = np.sort(raw_poles)

poles = np.array([])
num_poles = 0
index1 = 0
while index1 < len(poles_sorted)-1:
	if poles_sorted[index1] > x2max + 1:
		break
	poles = np.append(poles,poles_sorted[index1])
	num_poles  += 1
	index2 = index1+1
	while np.abs(poles_sorted[index1]-poles_sorted[index2])< eps:
		index2 +=1
	index1 = index2

print poles
# -----------------
# Calculate the F function


def ff(l,m,x2,num_poles):
	if l < m:
		print 'Error:\nabs(m) must be equal or less than l'
		return None
	else:
		# Kinematics
		q2 = ((2*np.pi/L)**2)*x2
		cm_energy = np.sqrt(q2 + m1**2) + np.sqrt(q2 + m2**2)
		lab_energy = np.sqrt(cm_energy**2+lab_moment2)
		alpha = 1.0/2.0*(1.0+(np.square(m1)-np.square(m2))/np.square(cm_energy))
		gamma = lab_energy/cm_energy
		
		# -----------------
		# Finish triplets calculation (gamma dependent part)
		# rn triplets
		# Calculate array of r vectors
		r_arr = (1.0/gamma)*(n_par_arr-alpha*d_arr)+n_perp_arr

		# Calculate array with square of the magnitude of r, the azimuthal and polar angle
		r2_array = np.sum(r_arr*r_arr, axis = 1)
		r_azimuthal = np.arctan2(r_arr[:,1],r_arr[:,0])
		r_polar = np.arctan2(np.sqrt(r_arr[:,0]**2 + r_arr[:,1]**2),r_arr[:,2])

		#gw triplets
		# Calculate array of gw vectors
		gw_arr = gamma*w_par_arr+w_perp_arr

		# Calculate array with square of the magnitude of gw, the azimuthal and polar angle
		gw2_array = np.sum(gw_arr*gw_arr, axis = 1)
		gw_azimuthal = np.arctan2(gw_arr[:,1],gw_arr[:,0])
		gw_polar = np.arctan2(np.sqrt(gw_arr[:,0]**2 + gw_arr[:,1]**2),gw_arr[:,2])

		first_term = sum(np.exp(-r2_array+x2)*(1.0/(r2_array-x2))*(r2_array**(l/2.0))*special.sph_harm(m,l,r_azimuthal,r_polar))

		# Calculate the second term
		if l == 0:
			second_term = special.sph_harm(0,0,0.0,0.0)*gamma*np.power(np.pi,3.0/2.0)
			second_term_bracket = 2 * x2 * integrate.quad(integr_second_term,0,1, args = (x2))[0] - 2 * np.exp(x2)
			second_term *= second_term_bracket

		else:
			second_term = 0

		# Calculate the third term
		wd = np.inner(w_arr,d)
		t1 = np.exp(I*2*np.pi*alpha*wd)
		t2 = np.power(gw2_array,l/2.0)
		t3 = special.sph_harm(m,l,gw_azimuthal,gw_polar)
		coef_third_term = t1*t2*t3

		third_term_r = integrate.quad(integr_third_term_r,0,1, args = (l,gw2_array,x2,coef_third_term))[0]

		third_term_i = integrate.quad(integr_third_term_i,0,1, args = (l,gw2_array,x2,coef_third_term))[0]

		third_term = gamma*np.power(I,l)* (third_term_r + I*third_term_i)

		factor = x2-poles[0:num_poles]
	 	gggg= np.prod(factor)

		total = gggg*(first_term + second_term + third_term)

		return total

# ---------
# Several point calculation		

x = np.zeros(points)
y = np.zeros(points) + I*np.zeros(points)

# for l in xrange(0,1):
# 	for m in xrange(-l,l+1):

for x2 in xrange(0,points):
	if x2 % 10 == 0:
		print x2
	x[x2] = x2/(points-1.0)*(x2max - x2min) + x2min
	y[x2] = ff(l,m,x[x2],num_poles)


filename = 'F_' + str(l) +'_' + str(m)+ '_d(' + str(d) +') x2max=' + str(x2max)

folder = '../'+folder+'F_values_' + str(dt.datetime.now())[:-16] +'/'

if not os.path.exists(folder):
	os.makedirs(folder)

np.save(folder+filename,y)

filename = 'x2-'+filename 

np.save(folder+filename,x)

filename = 'poles-'+ filename[3:]

np.save(folder+filename,poles[0:num_poles])

# Other ideas:

## to put inside ff function
#-------
		# Calculate the term with the pole in the first summation
		# factor = x2-poles[0:num_poles]
	 # 	factor_mat = np.repeat([factor],len(factor),axis=0)
	 # 	np.fill_diagonal(factor_mat,1)
	 # 	values = np.prod(factor_mat,axis=1)
	 # 	mult_dict = dict(zip(poles,values))
	 # 	mult_dict['tot']= np.prod(factor)

	 	# a_n = np.zeros(len(r2_array))
	 	# for x in xrange(len(r2_array)):
	 	# 	# if r2_array[x] < poles[num_poles-1]: 
	 		# 	a_n[x] = -mult_dict[r2_array[x]]
	 		# else:
	 		#	a_n[x]= mult_dict['tot']/(r2_array[x]-x2)	

	 	# Calculate the first term of the Z function
	 	# Denominator substitution by a_n term

#-------



# for i in xrange(len(x2_forplot)):
# 	y_forplot[i] = ff(x2_forplot[i],0,0,num_poles)
# 	print i

# plt.plot(x2_forplot,y_forplot)
# plt.show()

# def pru(x2):
# 	factor = x2-r2_array
# 	mul_mat = np.array(factor,ndmin =2)
# 	for x in xrange(1,r2_array.shape[0]): mul_mat=np.append(mul_mat,[factor],axis=0)
# 	np.fill_diagonal(mul_mat,1)
# 	mul_arr = np.prod(mul_mat,axis=1)
# 	print np.exp(-r2_array+x2)*mul_arr
# 	val = sum(np.exp(-r2_array+x2)*mul_arr)

# Get the poles of Z
# poles = np.sort(list(set(r2_array)))

# def g_f(x2):
# 	val = 1
# 	for pole in poles[0:3]: val *= x2-pole
# 	return val

# print g_f
# x2_forplot =np.linspace(0,1,num=21)
# y_forplot = np.zeros(len(x2_forplot))
# for i in xrange(len(x2_forplot)):
# 	y_forplot[i] = np.where((0.25-x2_forplot[i]) != 0, g_f(x2_forplot[i])/(0.25-x2_forplot[i]), 0)

# print y_forplot
# plt.plot(x2_forplot,y_forplot)
# plt.show()

# -----------------
# Calculate the zeta function
# left here just in case

def zdlm(l,m,x2):
	if l < m:
		print 'Error:\nabs(m) must be equal or less than l'
		return None
	else:
		# Kinematics
		q2 = ((2*np.pi/L)**2)*x2
		cm_energy = np.sqrt(q2 + m1**2) + np.sqrt(q2 + m2**2)
		lab_energy = np.sqrt(cm_energy**2+lab_moment**2)
		alpha = 1.0/2.0*(1.0+(np.square(m1)-np.square(m2))/np.square(cm_energy))
		gamma = 1.0 # lab_energy/cm_energy

		first_term = sum(np.exp(-r2_array+x2)*(1.0/(r2_array-x2))*(r2_array**(l/2.0))*special.sph_harm(m,l,r_azimuthal,r_polar))

		print first_term

		# Calculate the second term
		if l == 0:
			second_term = special.sph_harm(0,0,0.0,0.0)*gamma*np.power(np.pi,3.0/2.0)
			second_term_bracket = 2 * x2 * integrate.quad(integr_second_term,0,1, args = (x2))[0] - 2 * np.exp(x2)
			second_term *= second_term_bracket
		else:
			second_term = 0

		print second_term


		# Calculate the third term
		wd = np.inner(w_arr,d)
		t1 = np.exp(I*2*np.pi*alpha*wd)
		t2 = np.power(gw2_array,l/2.0)
		t3 = special.sph_harm(m,l,gw_azimuthal,gw_polar)
		coef_third_term = t1*t2*t3

		third_term_r = integrate.quad(integr_third_term_r,0,1, args = (l,gw2_array,x2,coef_third_term))[0]

		third_term_i = integrate.quad(integr_third_term_i,0,1, args = (l,gw2_array,x2,coef_third_term))[0]

		third_term = gamma*np.power(I,l)* (third_term_r + I*third_term_i)

		print third_term

		total = first_term + second_term + third_term

		return total

# for i in xrange(len(x2_forplot)):
# 	y_forplot[i] = zdlm(0,0,x2_forplot[i])
# 	print i