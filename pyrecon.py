import os
import logging
import json
import scipy
from scipy.ndimage.filters import gaussian_filter
from scipy.fftpack import fftfreq
import pyfftw
from pymakesurvey import sky_to_cartesian,distance

class PyFFTWBoxReconstruction(object):

	logger = logging.getLogger('PyFFTWBoxReconstruction')

	def __init__(self,data,Position='Position',Position_rec='Position_rec',los='local',f=0.817,bias=2.3,beta=None,smooth=15.,nthreads=1,Nmesh=256,BoxSize=None,BoxCenter=0.,path_wisdom='wisdom.nmesh-{0[0]:d}-{0[1]:d}-{0[2]:d}.nthreads{1:d}'):

		if beta is None: beta = f/bias
		if bias is None: bias = f/beta
		self.attrs = {}
		self.attrs['f'] = f
		self.attrs['beta'] = beta
		self.attrs['bias'] = bias
		self.attrs['Position'] = Position
		self.attrs['Position_rec'] = Position_rec
		self.attrs['los'] = los
		self.attrs['smooth'] = smooth
		self.attrs['path_wisdom'] = path_wisdom
		self.attrs['nthreads'] = nthreads
		self.data = data

		self._set_box_(BoxSize,BoxCenter,Nmesh)
		for key in ['f','bias','smooth','los','nthreads']:
			self.logger.info('Using {} = {}.'.format(key,self.attrs[key]))
		for key in ['Nmesh','BoxSize']:
			self.logger.info('Using {} = {}.'.format(key,getattr(self,key)))

	def _set_box_(self,BoxSize,BoxCenter,Nmesh):
		self.BoxSize = scipy.empty(self.ndim,dtype='f8')
		self.BoxSize[:] = BoxSize
		self.BoxCenter = scipy.empty(self.ndim,dtype='f8')
		self.BoxCenter[:] = BoxCenter
		self.Nmesh = scipy.empty(self.ndim,dtype=int)
		self.Nmesh[:] = Nmesh
	
	def set_empty_aligned(self,dtype):
		for key in ['delta','deltak','psix','psiy','psiz']:
			tmp = pyfftw.empty_aligned(self.Nmesh,dtype=dtype)
			setattr(self,key,tmp)

	def load_wisdom(self,path_wisdom=None):
		if path_wisdom is not None:
			self.attrs['path_wisdom'] = path_wisdom
		path_wisdom = self.attrs['path_wisdom'].format(self.Nmesh,self.attrs['nthreads'])
		if os.path.isfile(path_wisdom):
			self.logger.info('Reading wisdom from {}.'.format(path_wisdom))
			wisdom = open(path_wisdom,'r')
                	pyfftw.import_wisdom(json.load(wisdom))
			wisdom.close()

	def save_wisdom(self,path_wisdom=None):
		if path_wisdom is not None:
			self.attrs['path_wisdom'] = path_wisdom
		path_wisdom = self.attrs['path_wisdom'].format(self.Nmesh,self.attrs['nthreads'])
		if not os.path.isfile(path_wisdom):
			self.logger.info('Saving wisdom to {}.'.format(path_wisdom))
			wisdom = pyfftw.export_wisdom()
			file = open(path_wisdom,'w')
			json.dump(wisdom,file)
			file.close()

	def sample_cic(self,position,weight=1.):
		delta = scipy.zeros(self.Nmesh,dtype=position.dtype)
		findex = ((position-self.offset)/self.CellSize)
		index = findex.astype(int)
		dindex = findex-index
		#edges = [scipy.linspace(0,n,n+1) for n in self.Nmesh]
		ishifts = scipy.array(scipy.meshgrid(*([[0,1]]*self.ndim),indexing='ij')).reshape((self.ndim,-1)).T
		delta[...] = 0.
		for ishift in ishifts:
			sindex = index + ishift
			sweight = scipy.prod((1-dindex) + ishift*(-1+2*dindex),axis=-1)*weight
			#delta += scipy.histogramdd(sindex,bins=edges,weights=sweight)[0]
			#delta[tuple(sindex.T)] += sweight
			scipy.add.at(delta,tuple(sindex.T),sweight)
		
		return delta

	def shift_cic(self,position,fields):
		findex = ((position-self.offset)/self.CellSize)
		index = findex.astype(int)
		dindex = findex-index
		shifts = scipy.zeros_like(position)
		ishifts = scipy.array(scipy.meshgrid(*([[0,1]]*self.ndim),indexing='ij')).reshape((self.ndim,-1)).T
		for ishift in ishifts:
			sindex = tuple((index + ishift).T)
			sweight = scipy.prod((1-dindex) + ishift*(-1+2*dindex),axis=-1)
			for iaxis,field in enumerate(fields):
				shifts[:,iaxis] += field[sindex]*sweight

		return shifts

	def get_los(self,position,weight=None):
		los = self.attrs['los']
		if not scipy.isscalar(los):
			unit_vector = scipy.array(los,dtype='f8')
			unit_vector /= distance(unit_vector)
		elif los == 'local':
			unit_vector = position/distance(position)[:,None]
		elif los == 'global':
			unit_vector = scipy.average(position,weights=weight,axis=0)
			unit_vector /= distance(position)
		else:
			axis = los
			if isinstance(los,str): axis = 'xyz'.index(axis)
			unit_vector = scipy.zeros((self.ndim),dtype='f8')
			unit_vector[axis] = 1.
		return unit_vector

	def __call__(self,niter):
		self.initialize()
		for i in range(niter):
			self.iterate()

	def initialize(self,dtype='complex128',path_wisdom=None):
		self._position_rec_data = 1.*self._position_data
		self.set_empty_aligned(dtype=dtype)
		self.load_wisdom(path_wisdom=path_wisdom)
		self.fftobj = pyfftw.FFTW(self.delta,self.delta,axes=range(self.ndim),threads=self.attrs['nthreads'])
		self.ifftobj = pyfftw.FFTW(self.deltak,self.psix,axes=range(self.ndim),threads=self.attrs['nthreads'],direction='FFTW_BACKWARD')
		self.save_wisdom(path_wisdom=path_wisdom)
		self.iter = 0

	def set_density_contrast(self):
		deltag = self.sample_cic(self._position_rec_data)
		deltag = gaussian_filter(deltag,self.attrs['smooth']/self.CellSize)
		mean = scipy.mean(deltag)
		self.delta[...] = (deltag - mean)/mean

	def iterate(self):
		self.logger.info('Allocating data in cells for iteration {:d}...'.format(self.iter))
		self.set_density_contrast()
		self.fftobj(input_array=self.delta,output_array=self.delta)
		k = self.k()
		norm2 = (k[0][:, None, None]**2 + k[1][None, :, None]**2 + k[2][None, None, :]**2)
		norm2[0,0,0] = 1.
		#norm2 = (k[0][:, None, None]**2 + k[1][None, :, None]**2 + k[2][None, None, :]**2 + 1e-100)
		self.delta /= norm2
		self.delta[0,0,0] = 0.
		self.logger.info('Computing displacement field...')
		for iaxis,psi in enumerate(['psix','psiy','psiz']):
			sl = [None]*self.ndim; sl[iaxis] = slice(None)
			self.deltak[...] = self.delta*-1j*k[iaxis][tuple(sl)]/self.attrs['bias']
			self.ifftobj(input_array=self.deltak,output_array=getattr(self,psi))
		shifts = self.shift_cic(self._position_rec_data,[self.psix.real,self.psiy.real,self.psiz.real])
		self.logger.info('A few displacements values:')
		for s in shifts[:10]: self.logger.info('{}'.format(s))

		f,beta = self.attrs['f'],self.attrs['beta']
		los = self.get_los(self._position_data,weight=self._weight_data)
		# Burden 2015: 1504.02591v2, eq. 12
		if self.iter == 0:
			psi_dot_los = scipy.sum(shifts*los,axis=-1)
			shifts -= beta/(1+beta)*psi_dot_los[:,None]*los
		psi_dot_los = scipy.sum(shifts*los,axis=-1)
		self._position_rec_data = self._position_data + f*psi_dot_los[:,None]*los
		self.iter += 1

	def apply_shifts(self):
		self._position_rec_data += self.shift_cic(self._position_rec_data,[self.psix.real,self.psiy.real,self.psiz.real])

	def summary(self):
		shifts = self._position_rec_data - self._position_data
		self.logger.info('Shift statistics for each dimension: std 16th  84th  min  max')
		for s in shifts.T:
			self.logger.info('{:.4g} {:.4g} {:.4g} {:.4g} {:.4g}'.format(scipy.std(s),scipy.percentile(s,16.),scipy.percentile(s,84.),scipy.amin(s),scipy.amax(s)))
			#self.logger.info('{} {} {} {} {}'.format(scipy.std(s),scipy.percentile(s,16.),scipy.percentile(s,84.),scipy.amin(s),scipy.amax(s)))
	
	def k(self):
		return [fftfreq(n,d=d)*2*scipy.constants.pi for n,d in zip(self.Nmesh,self.CellSize)]

	@property
	def ndim(self):
		return 3

	@property
	def CellSize(self):
		return self.BoxSize/(self.Nmesh-1)

	@property
	def offset(self):
		return self.BoxCenter - self.BoxSize/2.
	
	@property
	def _position_data(self):
		return self.data[self.attrs['Position']]

	@property
	def _position_rec_data(self):
		return (self.data[self.attrs['Position_rec']] - self.offset) % self.BoxSize + self.offset

	@_position_rec_data.setter
	def _position_rec_data(self,x):
		self.data[self.attrs['Position_rec']] = (x - self.offset) % self.BoxSize + self.offset
	"""
	@property
	def _position_rec_data(self):
		return self.data[self.attrs['Position_rec']]

	@_position_rec_data.setter
	def _position_rec_data(self,x):
		self.data[self.attrs['Position_rec']] = x
	"""
	@property
	def _weight_data(self):
		return 1.

class PyFFTWReconstruction(PyFFTWBoxReconstruction):

	logger = logging.getLogger('PyFFTWReconstruction')

	def __init__(self,data,randoms,Position='Position',Weight='Weight',Position_rec='Position_rec',los='local',f=0.817,bias=2.3,beta=None,smooth=15.,nthreads=1,Nmesh=256,BoxSize=None,CellSize=None,BoxCenter=None,BoxPad=200.,path_wisdom='wisdom.nmesh-{0[0]:d}-{0[1]:d}-{0[2]:d}.nthreads{1:d}'):

		if beta is None: beta = f/bias
		if bias is None: bias = f/beta
		self.attrs = {}
		self.attrs['f'] = f
		self.attrs['beta'] = beta
		self.attrs['bias'] = bias
		self.attrs['Position'] = Position
		self.attrs['Weight'] = Weight
		self.attrs['Position_rec'] = Position_rec
		self.attrs['los'] = los
		self.attrs['smooth'] = smooth
		self.attrs['path_wisdom'] = path_wisdom
		self.attrs['nthreads'] = nthreads
		self.data = data
		self.randoms = randoms

		self.define_cartesian_box(self._position_randoms,Nmesh=Nmesh,BoxSize=BoxSize,CellSize=CellSize,BoxCenter=BoxCenter,BoxPad=BoxPad)
		for key in ['f','bias','smooth','los','nthreads']:
			self.logger.info('Using {} = {}.'.format(key,self.attrs[key]))
		for key in ['Nmesh','BoxSize']:
			self.logger.info('Using {} = {}.'.format(key,getattr(self,key)))

	def define_cartesian_box(self,position,Nmesh=256,BoxSize=None,CellSize=None,BoxCenter=None,BoxPad=0.02):
		pos_min, pos_max = position.min(axis=0),position.max(axis=0)
		delta = abs(pos_max - pos_min)
		if BoxCenter is None: BoxCenter = 0.5 * (pos_min + pos_max)
		if BoxSize is None:
			BoxSize = (delta.max() + 2.*BoxPad) # to match Julian's definition
		if (BoxSize < delta).any(): raise ValueError('BoxSize too small to contain all data.')
		if CellSize is not None:
			Nmesh = scipy.ceil(BoxSize/CellSize).astype(int) + 1
		self._set_box_(BoxSize,BoxCenter,Nmesh)	

	def initialize(self,dtype='complex128',path_wisdom=None):
		self.alpha = self._weight_data.sum()*1./self._weight_randoms.sum()
		self.min_deltar = 0.01*scipy.mean(self._weight_randoms)
		self._position_rec_data = 1.*self._position_data
		self.set_empty_aligned(dtype=dtype)
		self.logger.info('Allocating randoms in cells...')
		self.deltar = self.sample_cic(self._position_randoms,self._weight_randoms)
		self.deltar = gaussian_filter(self.deltar,self.attrs['smooth']/self.CellSize)
		self.load_wisdom(path_wisdom=path_wisdom)
		self.fftobj = pyfftw.FFTW(self.delta,self.delta,axes=range(self.ndim),threads=self.attrs['nthreads'])
		self.ifftobj = pyfftw.FFTW(self.deltak,self.psix,axes=range(self.ndim),threads=self.attrs['nthreads'],direction='FFTW_BACKWARD')
		self.save_wisdom(path_wisdom=path_wisdom)
		self.iter = 0

	def set_density_contrast(self):
		deltag = self.sample_cic(self._position_rec_data,self._weight_data)
		deltag = gaussian_filter(deltag,self.attrs['smooth']/self.CellSize)
		self.delta[...] = deltag - self.alpha*self.deltar
		mask = self.deltar > self.min_deltar
		self.delta[mask] /= (self.alpha*self.deltar[mask])
		self.delta[~mask] = 0.

	def apply_shifts(self):
		if self.attrs['Position_rec'] not in self.randoms:
			self._position_rec_randoms = 1.*self._position_randoms
		for catalogue in [self.data,self.randoms]:
			position = catalogue[self.attrs['Position_rec']]
			shifts = self.shift_cic(position,[self.psix.real,self.psiy.real,self.psiz.real])
			position += shifts

	@property
	def _position_randoms(self):
		return self.randoms[self.attrs['Position']]

	@property
	def _position_rec_data(self):
		return self.data[self.attrs['Position_rec']]

	@_position_rec_data.setter
	def _position_rec_data(self,x):
		self.data[self.attrs['Position_rec']] = x

	@property
	def _position_rec_randoms(self):
		return self.randoms[self.attrs['Position_rec']]

	@_position_rec_randoms.setter
	def _position_rec_randoms(self,x):
		self.randoms[self.attrs['Position_rec']] = x

	@property
	def _weight_data(self):
		return self.data[self.attrs['Weight']]
	
	@property
	def _weight_randoms(self):
		return self.randoms[self.attrs['Weight']]



