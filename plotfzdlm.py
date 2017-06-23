import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import datetime as dt

# Matplotlib param
plt.rc('text', usetex=True)
plt.rc('font', size=12)
plt.rc('font', family='serif')
plt.rc('axes.formatter', useoffset = False)

# -----------------
# Config file:
config_file = str(sys.argv[1])

with open(config_file, 'r') as f:
	for line in f:
		if line[0:6] == 'folder':
			folder = line[7:-1]
		if line[0] == 'l':
			l = int(line[2:])
		if line[0] == 'm':
			m = int(line[2:])
		if line[0:5] == 'x2max':
			x2max = float(line[6:])
		if line[0:2] == 'dx':
			dx = int(line[2:])
		if line[0:2] == 'dy':
			dy = int(line[2:])
		if line[0:2] == 'dz':
			dz = int(line[2:])


d = [dx,dy,dz]

save_folder = '../'+folder+'Plots/'

filename = 'F_' + str(l) +'_' + str(m)+ '_d(' + str(d) +') x2max='+ str(x2max)

# For plotting a specific data set use the following format:
# folder = 'F_values_'+'2017-mm-dd'+'/'

folder = '../'+folder+'F_values_'+str(dt.datetime.now())[:-16]+'/'

y = np.load(folder+filename+'.npy')

filename = 'x2-'+filename 

x = np.load(folder+filename+'.npy')

filename = 'poles-'+ filename[3:]

poles = np.load(folder+filename+'.npy')
num_poles = len(poles)

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

y_re = np.real(y)
y_im = np.imag(y)

fig = plt.figure()

plt.plot(x, y_re,label='Real part')
plt.plot(x, y_im, label='Imaginary part')

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

figname = save_folder + filename[6:] + '--' + str(dt.datetime.now())[:-16]+'.pdf'

# show the plot on the screen

plt.show()

fig.savefig(figname)
