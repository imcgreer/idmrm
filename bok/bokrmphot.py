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

seasonMjdRange = {'2014':(56600,56900),'2015':(57000,57250),
                  '2016':(57000,57600),'2017':(57770,58000)}

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


bokMagRange = {'g':(17.0,19.5),'i':(16.5,19.2)}
bokMaxSeeing = 2.3/0.455

#dataMap = bokrmpipe.quick_load_datamap()
#refCat = idmrmphot.CleanSdssStarCatalog()

def bok_selfcal(dataMap,refCat,procmap):
	frameListFile = os.path.join(os.path.join(os.environ['BOKRMDIR'],'data'),
	                             'BokRMFrameList.fits.gz')
	#
	frameList = Table.read(frameListFile)
	#badFrames = identify_bad_frames(frameList)
	#
	bokPhot = load_raw_bok_aperphot(dataMap,refCat.name)
	bok7 = idmrmphot.extract_aperture(bokPhot,-2,badFrames=None)
	if True:
		del frameList['aperZp','aperZpRms','aperNstar','psfZp','psfNstar',
		          #'aperCorr','meanAperZp','n','chi2','outlierFrac','rchi2']
		          'meanAperZp','n','chi2','outlierFrac','rchi2']
	#
	sePhot,coaddPhot,zpts = idmrmphot.iter_selfcal(bok7,frameList,refCat,
	                                          magRange=bokMagRange,
	                                          maxSeeing=bokMaxSeeing,
	                                     calColorXform=Sdss2BokTransform())
	sePhot.write('sephot.fits',overwrite=True)
	coaddPhot.write('coaddphot.fits',overwrite=True)
	zpts.write('zpts.fits',overwrite=True)


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
	parser.add_argument('--updatemeta',action='store_true',
	                help='update metadata table with outlier stats')
	parser.add_argument('--nightly',action='store_true',
	                help='construct nightly lightcurves')
	parser.add_argument('-p','--processes',type=int,default=1,
	                help='number of processes to use [default=single]')
	parser.add_argument('--old',action='store_true',
	                help='use 2014 catalogs for comparison')
	parser.add_argument('-v','--verbose',action='count',
	                    help='increase output verbosity')
	parser.add_argument('--lctable',type=str,
	                help='lightcurve table')
	parser.add_argument('--zptable',type=str,default='bokrm_zeropoints.fits',
	                help='zeropoints table')
	parser.add_argument('--catdir',type=str,default='.',
	                help='directory containing photometry catalogs')
	parser.add_argument('--nowrite',action='store_true',
	                help='skip writing output files')
	parser.add_argument('--season',type=str,
	                help='observing season')
	parser.add_argument('--outfile',type=str,default='',
	                help='output file')
	args = parser.parse_args()
	args = bokrmpipe.set_rm_defaults(args)
	dataMap = bokpl.init_data_map(args)
	dataMap = bokrmpipe.config_rm_data(dataMap,args)
	photCat = idmrmphot.load_target_catalog(args.catalog,args.catdir,
	                                        args.lctable)
#	photCat.load_ref_catalog()
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
		#zero_points(dataMap,procmap,photCat)
		bok_selfcal(dataMap,photCat,procmap)
		timerLog('zeropoints')
	if args.lightcurves:
		photCat.load_bok_phot(nogroup=True)
		calibrate_lightcurves(photCat,dataMap,zpFile=args.zptable,
		                      season=args.season,old=args.old)
		timerLog('lightcurves')
	if args.aggregate:
		photCat.load_bok_phot(nogroup=True,season=args.season)
		which = 'nightly' if args.nightly else 'all'
		aggregate_phot(photCat,which,
		               catDir=args.catdir,outName=args.outfile)
#	elif args.nightly:
#		nightly_lightcurves(refCat['filePrefix'],redo=args.redo)
		timerLog('aggregate phot')
	if args.updatemeta:
		sephotfn = phot_file_names('sephot','sdssrefstars','all',args.outfile)
		psum = load_agg_phot(os.path.join(args.catdir,sephotfn))
		frameStats,objStats = calc_frame_stats(psum)
		update_framelist_withoutliers(frameStats)
		write_object_badlist(objStats,frameStats,'sdssPhotSummary.fits')
		timerLog('update metadata')
	timerLog.dump()
	if args.processes > 1:
		pool.close()

