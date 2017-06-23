import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

# Matplotlib param
plt.rc('text', usetex=True)
plt.rc('font', size=12)
plt.rc('font', family='serif')
plt.rc('axes.formatter', useoffset = False)

# -----------------
# Config file:
config_file = str(sys.argv[1])
sd = False

with open(config_file, 'r') as f:
	for line in f:
		if line[0:6] == 'folder':
			folder = line[7:-1]
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
		if line[0:2] == 'sd':
			sd = True
			speci_dat = str(line[3:-1])


d = [dx,dy,dz]

save_folder = '../'+folder+'scratch/'

# Filename of the calculated z function
filename = 'Z_' + str(l) +'_' + str(m)+ '_d(' + str(d) +')'

# For plotting a specific data set use the following format:
# speci_dat = 'F_values_'+'2017-mm-dd'+'/'
if sd == False:
	folder = '../'+folder+'Z_values_'+str(dt.datetime.now())[:-16]+'/'
else:
	folder = '../'+folder+'Z_values_'+speci_dat+'/'
	
zdlm = np.load(folder+filename+'.npy')

print zdlm[0]
 
filename = 'x2-'+filename 

x2 = np.load(folder+filename+'.npy')

x2max = x2[-1]

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

# Get the poles of Z (identical particles m1==m2)
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

# Pole annotation
vert_pos = .88

polestring = 'Poles at: \n $z_i \in \{$'
for ii in xrange(num_poles):
	if ii+1 == num_poles:
		polestring += '%.2f' % poles[ii] + '$\}$'
	elif ii < 5 and (ii+1)%5 == 0 :
		vert_pos -= 0.05
		polestring += '%.2f' % poles[ii] +',\n'
	elif (ii+1)%6  == 0 and ii > 5:
		vert_pos -= 0.05
		polestring += '%.2f' % poles[ii] +',\n'
	else:
		polestring += '%.2f' % poles[ii] +', '


# Calculate g 

x2_arr = np.repeat([x2],num_poles,axis=0)
pol_arr = np.transpose(np.repeat([poles[0:num_poles]],len(x2),axis=0))

factor = x2_arr-pol_arr
gggg= np.prod(factor,axis=0)

# Multiply g times Z to get f
fdlm = gggg*zdlm

# Plot

y_re = np.real(fdlm)
y_im = np.imag(fdlm)


fig = plt.figure()

plt.plot(x2, y_re,label=r'Real part')
plt.plot(x2, y_im,label=r'Imaginary part')

# tidy up the figure
plt.grid(True)
plt.legend(loc='upper right')
plt.title(r'$\displaystyle Z_{'+str(l)+str(m)+'}^{'+str(d)+'}[1;x^2]\prod_{i}(x^2-z_i)$')
plt.subplots_adjust(top=0.86)
plt.xlabel(r'$x^2$')

plt.annotate(polestring, xy=(0.05, vert_pos), xycoords="axes fraction",
			bbox=dict(boxstyle="round",fc="1"))


# Prepare to save plot

if not os.path.exists(save_folder):
	os.makedirs(save_folder)

filename = 'F_z_' + str(l) +'_' + str(m)+ '_d(' + str(d) +') x2max='+str(x2max)

figname = save_folder + filename + '--' + str(dt.datetime.now())[:-16]+'.pdf'

# show the plot on the screen

plt.show()

fig.savefig(figname)
