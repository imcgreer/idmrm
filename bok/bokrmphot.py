#!/usr/bin/env python

import os
import numpy as np
import multiprocessing
from functools import partial
from astropy.io import fits
from astropy.table import Table,vstack,hstack,join
from astropy.stats import sigma_clip
from astropy.wcs import InconsistentAxisTypesError

from bokpipe import bokphot,bokpl,bokproc,bokutil
import bokrmpipe
import idmrmphot

# Nominal limits to classify night as "photometric"
zp_phot_nominal = {'g':25.90,'i':25.40}

bokrm_aperRad = np.concatenate([np.arange(2,9.51,1.5),[15.,22.5]])

def get_phot_file(photCat,inFile):
	if inFile is None:
		return '{0}_{1}.fits'.format('bokrmphot',photCat.name)
	else:
		return inFile

class Sdss2BokTransform(object):
	colorMin = 0.4
	colorMax = 3.0
	def __init__(self,clearZero=False):
		_cfgdir = os.path.join(os.environ['BOKRMDIR'],'config') # XXX
		self.cterms = {}
		for b in 'gi':
			self.cterms[b] = np.loadtxt(os.path.join(_cfgdir,
			                                   'bok2sdss_%s_gicoeff.dat'%b))
			if clearZero:
				self.cterms[b][-1] = 0
	def __call__(self,mags,colors,filt):
		corrMags = mags.copy()
		for b in 'gi':
			ii = np.where(filt==b)[0]
			corrMags[ii] += np.polyval(self.cterms[b],colors[ii])
		corrMags = np.ma.array(corrMags,
		                  mask=(colors<self.colorMin)|(colors>self.colorMax))
		return corrMags

class BokConfig(object):
	name = 'bok'
	nCcd = 4
	nAmp = 16
	ccd0 = 1
	nAper = 7
	zpAperNum = -2
	zpMinSnr = 10.
	zpMinNobs = 10
	zpMaxSeeing = 2.3/0.455
	zpMaxChiVal = 5.
	apCorrMaxRmsFrac = 0.5
	apCorrMinSnr = 20.
	apCorrMinNstar = 20
	magRange = {'g':(17.0,19.5),'i':(16.5,19.2)}
	colorxfun = Sdss2BokTransform()
	def colorXform(self,mag,clr,filt):
		return self.colorxfun(mag,clr,filt)


##############################################################################
#
# aperture photometry routines
#
##############################################################################

def aper_worker(dataMap,inputType,aperRad,refCat,catDir,catPfx,
                inp,**kwargs):
	utd,filt = inp
	redo = kwargs.pop('redo',False)
	nowrite = kwargs.pop('nowrite',False)
	fn = '.'.join([catPfx+'_aper',utd,filt,'cat','fits'])
	catFile = os.path.join(catDir,fn)
	if os.path.exists(catFile) and not redo:
		print catFile,' already exists, skipping'
		return
	files,frames = dataMap.getFiles(imType='object',utd=utd,filt=filt,
	                                with_frames=True)
	if files is None:
		return
	#
	bpMask = dataMap.getCalMap('badpix4')
	diagfile = os.path.join(dataMap.getDiagDir(), 'gainbal_%s.npz'%utd)
	gainDat = np.load(diagfile)
	#
	allPhot = []
	for f,frame in zip(files,frames):
		imageFile = dataMap(inputType)(f)
		aHeadFile = imageFile.replace('.fits','.ahead')
		fileNum = np.where(gainDat['files']==os.path.basename(f))[0][0]
		gains = gainDat['gainCor'][fileNum]
		gains = np.product(gains.squeeze(),axis=1) 
		gains *= np.array(bokproc.nominal_gain)
		skyAdu = gainDat['skys'][fileNum]
		expTime = fits.getheader(imageFile,0)['EXPTIME']
		varIm = bokpl.make_variance_image(dataMap,f,bpMask,
		                                  expTime,gains,skyAdu)
		bokutil.mplog('aperture photometering '+f)
		try:
			phot = bokphot.aper_phot_image(imageFile,
			                               refCat['ra'],refCat['dec'],
			                               aperRad,bpMask(f),varIm,
			                               aHeadFile=aHeadFile,
			                               **kwargs)
		except InconsistentAxisTypesError:
			print 'WCS FAILED!!! ',f
			continue
		except:
			print 'aper_phot_image FAILED!!! ',f
			continue
		if phot is None:
			print 'no apertures found!!!! ',f
			continue
		phot['frameIndex'] = dataMap.obsDb['frameIndex'][frame]
		allPhot.append(phot)
	allPhot = vstack(allPhot)
	if not nowrite:
		allPhot.write(catFile,overwrite=True)

def aperture_phot(dataMap,photCat,procmap,inputType='sky',**kwargs):
	kwargs.setdefault('mask_is_weight_map',False)
	kwargs.setdefault('background','global')
	aperRad = bokrm_aperRad
	catDir = os.path.join(dataMap.procDir,'catalogs')
	if not os.path.exists(catDir):
		os.mkdir(catDir)
	utdlist = [ (utd,filt) for utd in dataMap.iterUtDates() 
	                         for filt in dataMap.iterFilters() ]
	p_aper_worker = partial(aper_worker,dataMap,inputType,
	                        aperRad,photCat.refCat,catDir,photCat.name,
	                        **kwargs)
	procmap(p_aper_worker,utdlist)

def load_raw_bok_aperphot(dataMap,targetName,season=None,old=False):
	if old:
		# renaming
		pfx = {'sdssstarsold':'sdssbright'}.get(targetName,targetName)
		aperCatDir = os.environ['HOME'] + \
		                '/data/projects/SDSS-RM/rmreduce/catalogs_v2b/'
	else:
		aperCatDir = os.path.join(dataMap.procDir,'catalogs')
		pfx = targetName+'_aper'
	allTabs = []
	for utd in dataMap.iterUtDates():
		if ( season is not None and
		      not ( utd.startswith(season) or 
		              (utd.startswith('2013') and season=='2014') ) ):
			continue
		print 'loading catalogs from ',utd
		for filt in dataMap.iterFilters():
			if old and utd=='20131223':
				utd = '20131222'
			aperCatFn = '.'.join([pfx,utd,filt,'cat','fits'])
			aperCatF = os.path.join(aperCatDir,aperCatFn)
			if os.path.exists(aperCatF):
				if old:
					tab = _read_old_catf(dataMap.obsDb,aperCatF)
				else:
					tab = Table.read(aperCatF)
#				tab['filter'] = filt # handy to save this here
				allTabs.append(tab)
	#
	phot = vstack(allTabs)
	print 'stacked aperture phot catalogs into table with ',len(phot),' rows'
	#self.phot.sort(['objId','frameIndex'])
	phot.sort('objId')
	return phot


#dataMap = bokrmpipe.quick_load_datamap()
#refCat = idmrmphot.CleanSdssStarCatalog()

def bok_zeropoints(dataMap,frameList,refCat,bokCfg,debug=False):
	frameList.sort('frameIndex')
	#
	fields = ['frameIndex','utDate','filter','mjdStart','mjdMid','airmass']
	frameList = idmrmphot.join_by_id(frameList,dataMap.obsDb[fields],
	                                 'frameIndex')
	# zero point trends are fit over a season
	if 'season' not in frameList.colnames:
		frameList['season'] = idmrmphot.get_season(frameList['mjdStart'])
	# select the 7" aperture
	bokPhot = load_raw_bok_aperphot(dataMap,refCat.name)
	bok7 = idmrmphot.extract_aperture(bokPhot,bokCfg.zpAperNum,badFrames=None)
	# calculate zeropoints and aperture corrections
	zpdat = idmrmphot.iter_selfcal(bok7,frameList,refCat,bokCfg)
	frameList = idmrmphot.calc_apercorrs(bokPhot,frameList,bokCfg,mode='ccd')
	#
	if True:
		zpdat.zptrend.write('zptrend.dat',overwrite=True,format='ascii')
	if debug:
		zpdat.sePhot.write('zp_sephot.fits',overwrite=True)
		zpdat.coaddPhot.write('zp_coaddphot.fits',overwrite=True)
	return frameList


##############################################################################
#
# main()
#
##############################################################################

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser = bokpl.init_file_args(parser)
	parser.add_argument('--catalog',type=str,default='sdssrm',
	                help='reference catalog ([sdssrm]|sdss|cfht)')
	parser.add_argument('--aperphot',action='store_true',
	                help='generate aperture photometry catalogs')
	parser.add_argument('--background',type=str,default='global',
	                help='background method to use for aperture phot ([global]|local|none)')
	parser.add_argument('--zeropoint',action='store_true',
	                help='do zero point calculation')
	parser.add_argument('--lightcurves',action='store_true',
	                help='construct lightcurves')
	parser.add_argument('--aggregate',action='store_true',
	                help='construct aggregate photometry')
	parser.add_argument('--nightly',action='store_true',
	                help='construct nightly lightcurves')
	parser.add_argument('--binnedstats',action='store_true',
	                help='compute phot stats in mag bins')
	parser.add_argument('-p','--processes',type=int,default=1,
	                help='number of processes to use [default=single]')
	parser.add_argument('--old',action='store_true',
	                help='use 2014 catalogs for comparison')
	parser.add_argument('-v','--verbose',action='count',
	                    help='increase output verbosity')
	parser.add_argument('--lctable',type=str,
	                help='lightcurve table')
	parser.add_argument('--zptable',type=str,
	                help='zeropoints table')
	parser.add_argument('--catdir',type=str,default='.',
	                help='directory containing photometry catalogs')
	parser.add_argument('--nowrite',action='store_true',
	                help='skip writing output files')
	parser.add_argument('--season',type=str,
	                help='observing season')
	parser.add_argument('--aper',type=int,default=-2,
	                help='index of aperture to select [-2]')
	parser.add_argument('--outfile',type=str,default='',
	                help='output file')
	args = parser.parse_args()
	args = bokrmpipe.set_rm_defaults(args)
	dataMap = bokpl.init_data_map(args)
	dataMap = bokrmpipe.config_rm_data(dataMap,args)
	photCat = idmrmphot.load_target_catalog(args.catalog)
	bokCfg = BokConfig()
	if args.zptable:
		frameListFile = args.zptable
	else:
		_dataDir = os.path.join(os.environ['BOKRMDIR'],'data')
		frameListFile = os.path.join(_dataDir,'BokRMFrameList.fits.gz')
	bokPhot,frameList = None,None
	if args.processes == 1:
		procmap = map
	else:
		pool = multiprocessing.Pool(args.processes)
		procmap = pool.map
	timerLog = bokutil.TimerLog()
	if args.aperphot:
		aperture_phot(dataMap,photCat,procmap,redo=args.redo,
		              background=args.background,nowrite=args.nowrite)
		timerLog('aper phot')
	if args.zeropoint:
		print 'loading zeropoints table {0}'.format(frameListFile)
		frameList = Table.read(frameListFile)
		frameList = bok_zeropoints(dataMap,frameList,photCat,bokCfg)
		frameList.write(frameListFile,overwrite=True)
		timerLog('zeropoints')
	if args.lightcurves:
		if frameList is None:
			print 'loading zeropoints table {0}'.format(frameListFile)
			frameList = Table.read(frameListFile)
		bokPhot = load_raw_bok_aperphot(dataMap,photCat.name,
		                                season=args.season)
		phot = idmrmphot.calibrate_lightcurves(bokPhot,frameList,bokCfg)
		photFile = get_phot_file(photCat,args.lctable)
		print 'writing lightcurve catalog {}'.format(photFile)
		phot.write(photFile,overwrite=True)
		timerLog('lightcurves')
	if args.aggregate:
		# XXX
		which = 'nightly' if args.nightly else 'all'
		aggregate_phot(photCat,which,
		               catDir=args.catdir,outName=args.outfile)
		timerLog('aggregate phot')
	if args.binnedstats:
		if bokPhot is None:
			photFile = get_phot_file(photCat,args.lctable)
			print 'loaded lightcurve catalog {}'.format(photFile)
			bokPhot = Table.read(photFile)
		apPhot = idmrmphot.extract_aperture(bokPhot,args.aper,badFrames=None,
		                                    lightcurve=True)
		bs = idmrmphot.get_binned_stats(apPhot,photCat.refCat,bokCfg)
		outfile = args.outfile if args.outfile else 'phot_stats_bok.fits'
		bs.write(outfile,overwrite=True)
	timerLog.dump()
	if args.processes > 1:
		pool.close()

