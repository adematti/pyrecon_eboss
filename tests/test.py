from pyrecon import PyFFTWReconstruction,setup_logging
from recon_julian import Recon
from pymakesurvey import Catalogue,distance
import os
import scipy

def test_recon(niter=3,BoxPad=200.,smooth=15.,Nmesh=512,nthreads=1,bias=1.4,f=.82):

	path_data = os.path.join(os.getenv('ELGCAT'),'ELG.release.DR16','eBOSS_ELG_clustering_NGC_v7.dat.fits')
	path_randoms = os.path.join(os.getenv('ELGCAT'),'ELG.release.DR16','eBOSS_ELG_clustering_NGC_v7.ran.fits')

	data = Catalogue.from_fits(path_data)
	randoms = Catalogue.from_fits(path_randoms)
	randoms = randoms[::20]
	for catalogue in [data,randoms]: catalogue['Weight'] = catalogue['WEIGHT_SYSTOT']*catalogue['WEIGHT_CP']*catalogue['WEIGHT_NOZ']

	refrec = Recon(data['RA'],data['DEC'],data['Z'],data['Weight'],randoms['RA'],randoms['DEC'],randoms['Z'],randoms['Weight'],nbins=Nmesh,smooth=smooth,f=f,bias=bias,padding=BoxPad,nthreads=nthreads)
	data['Position'] = scipy.array([refrec.dat.x,refrec.dat.y,refrec.dat.z]).T
	randoms['Position'] = scipy.array([refrec.ran.x,refrec.ran.y,refrec.ran.z]).T
	for i in range(niter):
		refrec.iterate(i)
	refrec.apply_shifts()
	refrec.summary()
	myrec = PyFFTWReconstruction(data,randoms,Position='Position',Weight='Weight',Position_rec='Position_rec',los='local',f=f,bias=bias,beta=None,smooth=smooth,nthreads=nthreads,Nmesh=Nmesh,BoxSize=None,CellSize=None,BoxCenter=None,BoxPad=BoxPad)
	myrec(niter)
	myrec.apply_shifts()
	myrec.summary()

	los = data['Position']/distance(data['Position'])[:,None]
	los_rec = data['Position_rec']/distance(data['Position_rec'])[:,None]
	print scipy.diff(los-los_rec).max()

setup_logging()
test_recon(niter=3,Nmesh=128,nthreads=8)
