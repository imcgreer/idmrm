#!/usr/bin/env python

import os
import glob
import numpy as np
from astropy.table import Table,vstack,join

from bokpipe.badpixels import build_mask_from_flat
from bokpipe import bokpl,bokobsdb
from bokpipe import __version__ as pipeVersion

def set_rm_defaults(args,version=pipeVersion):
	if args.rawdir is None:
		args.rawdir = os.environ['BOK90PRIMERAWDIR']
	if args.output is None:
		args.output = os.path.join(os.environ['BOK90PRIMEOUTDIR'],
		                           version)
	if args.obsdb is None:
		args.obsdb = os.path.join('config','sdssrm-bok.fits.gz')
	return args

def build_headerfix_dict():
	hdrfix = {}
	# missing filter
	for fn in [ 'ut20140314/bokrm.20140314.%04d' % 
	                                      _i for _i in range(173,180) ]:
		hdrfix[fn] = [ (0,{'FILTER':'g'}) ]
	for fn in [ 'ut20140317/bokrm.20140317.%04d' % _i for _i in [147,153] ]:
		hdrfix[fn] = [ (0,{'FILTER':'i'}) ]
	for fn in [ 'ut20140318/bokrm.20140318.%04d' % _i for _i in [113] ]:
		hdrfix[fn] = [ (0,{'FILTER':'g'}) ]
	for fn in [ 'ut20140609/bokrm.20140609.%04d' % _i for _i in [178] ]:
		hdrfix[fn] = [ (0,{'FILTER':'g'}) ]
	# bad coordinates
	fn = 'ut20140128/bokrm.20140128.0094'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':53.08325}) for j in range(1,17) ]
	fn = 'ut20140129/bokrm.20140129.0103'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':53.08322}) for j in range(1,17) ]
	fn = 'ut20140215/bokrm.20140215.0116'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL2':213.704375}) for j in range(1,17) ]
	fn = 'ut20140219/bokrm.20140219.0136'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL2':213.7042917}) for j in range(1,17) ]
	fn = 'ut20140312/bokrm.20140312.0148'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL2':213.7042083}) for j in range(1,17) ]
	fn = 'ut20140415/bokrm.20140415.0055'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL2':213.704375}) for j in range(1,17) ]
	fn = 'ut20140416/bokrm.20140416.0148'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':53.083305}) for j in range(1,17) ]
	fn = 'ut20140513/bokrm.20140513.0080'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':52.227722}) for j in range(1,17) ]
	fn = 'ut20140518/bokrm.20140518.0076'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL2':215.09225}) for j in range(1,17) ]
	fn = 'ut20140518/bokrm.20140518.0118'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL2':213.70442}) for j in range(1,17) ]
	fn = 'ut20140518/bokrm.20140518.0130'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL2':215.09221}) for j in range(1,17) ]
	return hdrfix

def make_obs_db(args):
	# all Bok observations during RM nights (incl. IBRM)
	fullObsDbFile = os.path.join(os.environ['BOK90PRIMERAWDIR'],
	                             'sdssrm-allbok.fits')
	if not os.path.exists(fullObsDbFile) or args.redo:
		utDirs = sorted(glob.glob(os.path.join(args.rawdir,'ut201?????')))
		print utDirs
		try:
			obsDb = Table.read(fullObsDbFile)
			print 'starting with existing ',fullObsDbFile
		except IOError:
			obsDb = None
		bokobsdb.generate_log(utDirs,fullObsDbFile,inTable=obsDb)
		obsDb = Table.read(fullObsDbFile)
		# fix problems
		# 1. files that are missing FILTER values
		missing = [ 'bokrm.20140314.%04d' % _i for _i in range(173,180) ]
		obsDb['filter'][np.in1d(obsDb['fileName'],missing)] = 'g'
		missing = [ 'bokrm.20140317.%04d' % _i for _i in [147,153] ]
		obsDb['filter'][np.in1d(obsDb['fileName'],missing)] = 'i'
		missing = [ 'bokrm.20140318.%04d' % _i for _i in [113] ]
		obsDb['filter'][np.in1d(obsDb['fileName'],missing)] = 'g'
		missing = [ 'bokrm.20140609.%04d' % _i for _i in [178] ]
		obsDb['filter'][np.in1d(obsDb['fileName'],missing)] = 'g'
		# 2. flag bad images
		good = np.ones(len(obsDb),dtype=bool)
		#       #184 telescope was moving
		bad = [ 'd6649.%04d' % _i for _i in [184] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       #1 flat has weird stripes
		bad = [ 'a%04d' % _i for _i in [1] ]
		good[(obsDb['utDate']=='20140114') &
		     np.in1d(obsDb['fileName'],bad)] = False
		#       first survey image was 300s and way overexposed
		bad = [ 'bokrm.20140114.%04d' % _i for _i in [1] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       trailed image
		bad = [ 'bok.20140115.%04d' % _i for _i in [3] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       trailed image
		bad = [ 'bokrm.20140118.%04d' % _i for _i in [130] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       trailed image
		bad = [ 'bokrm.20140123.%04d' % _i for _i in [201] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       telescope was moving
		bad = [ 'bokrm.20140129.%04d' % _i for _i in [123] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       telescope was moving
		bad = [ 'bokrm.20140215.%04d' % _i for _i in [140] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       lots of passing clouds with saturated ims this night
		#       #124 - telescope was moving
		bad = [ 'bokrm.20140219.%04d' % _i for _i in [93,124,144,145,146,147,
		                                              150,152,153,158] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       #173 telescope was moving
		bad = [ 'bokrm.20140312.%04d' % _i for _i in [173] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       lots of passing clouds with saturated ims this night
		bad = [ 'bokrm.20140313.%04d' % _i for _i in [119,121,122] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       #26 flat is truncated, passing clouds
		bad = [ 'bokrm.20140319.%04d' % _i for _i in [26,51,52,53,54,55,
		                                              56,57,58,59,60] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       trailed image
		bad = [ 'bokrm.20140514.%04d' % _i for _i in [104,116] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       double image, telescope jumped
		bad = [ 'bokrm.20140518.%04d' % _i for _i in [114] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       #181 telescope was moving
		bad = [ 'bokrm.20140609.%04d' % _i for _i in [181] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       #116,171 telescope was moving
		bad = [ 'bokrm.20140610.%04d' % _i for _i in [116,171] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       lots of passing clouds with saturated ims this night
		bad = [ 'bokrm.20140612.%04d' % _i for _i in [88,89,90,102,103,104,
		                                              105,107,110,111,112,
		                                              113,114,115,119,120,
		                                              121,122,123] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       bad read, looks like aborted exposure
		bad = [ 'd7467.%04d' % _i for _i in [101] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		# write the edited table
		obsDb['good'] = good
		obsDb.write(fullObsDbFile,overwrite=True)
	obsDb = Table.read(fullObsDbFile)
	# all RM observations
	iszero = obsDb['imType']=='zero'
	isflat = ( (obsDb['imType']=='flat') & 
	           ((obsDb['filter']=='g')|(obsDb['filter']=='i')) )
	isrmfield = np.array([n.startswith('rm') for n in obsDb['objName']])
	isrmfield &= (obsDb['imType']=='object')
	isrm = iszero | isflat | isrmfield
	rmObsDbFile = os.path.join('config','sdssrm-bok.fits')
	obsDb[isrm].write(rmObsDbFile,overwrite=True)
	if False:
		# all RM observations in 2014
		isrm2014 = isrm & (obsDb['mjd']<57000)
		rmObsDbFile = os.path.join('config','sdssrm-bok2014.fits')
		obsDb[isrm2014].write(rmObsDbFile,overwrite=True)

def update_db_with_badsky(maxSkyCounts=40000):
	from astropy.io import fits
	rmObsDbFile = os.path.join('config','sdssrm-bok.fits.gz')
	f = fits.open(rmObsDbFile,mode='update')
	skyDat = Table.read(os.path.join('data','bokrm_skyadu.fits'))
	ii = np.where(f[1].data['imType']=='object')[0]
	m = join(Table(f[1].data[ii])['frameIndex',],skyDat,'frameIndex')
	assert np.all(f[1].data['frameIndex'][ii]==m['frameIndex'])
	f[1].data['good'][ii] &= m['skyMean'] < maxSkyCounts
	f.flush()
	f.close()


def get_observing_season(dataMap):
	year = np.unique([ utd[:4] for utd in dataMap.getUtDates() ])
	if np.all(sorted(year) == ['2013','2014']):
		year = ['2014']
	elif len(year) > 1:
		raise ValueError("Currently processing only one observing season "
		                 "at a time is supported")
	return year[0]

class IllumSelector(object):
	minNImg = 15
	maxNImg = 30
	skySigThresh = 3.0
	maxCounts = 20000
	def __init__(self):
		self.sky = vstack([Table.read('data/bokrm2014skyg.fits.gz'),
		                   Table.read('data/bokrm2014skyi.fits.gz')])
	def __call__(self,obsDb,ii):
		keep = np.ones(len(ii),dtype=bool)
		# this whole night is bad due to passing clouds
		keep[:] ^= obsDb['utDate'][ii] == '20140612'
		# repeated images at same pointing center
		jj = np.arange(1,len(ii),2)
		keep[jj] ^= obsDb['objName'][ii[jj]] == obsDb['objName'][ii[jj-1]]
		jj = np.arange(2,len(ii),2)
		keep[jj] ^= obsDb['objName'][ii[jj]] == obsDb['objName'][ii[jj-1]]
		# pointings swamped with bright stars
		for field in ['rm10','rm11','rm12','rm13']:
			keep[obsDb['objName'][ii]==field] = False
		return keep
		if keep.sum()==0:
			return keep
		# filter out bright sky values
		t = join(obsDb[ii],self.sky,'fileName')
		if len(t) < len(ii):
			raise ValueError('missing files!')
		skyMean = t['skyMean'].mean()
		skyRms = t['skyMean'].std()
		jj = t['skyMean'].argsort()
		nimg = keep[jj].cumsum()
		jlast = np.where((nimg>self.minNImg) & 
		            ((t['skyMean'][jj] > skyMean+self.skySigThresh*skyRms) |
		             (t['skyMean'][jj] > self.maxCounts)))[0]
		if len(jlast) > 0:
			keep[jj[jlast:]] = False
		if keep.sum() > self.maxNImg:
			keep[jj[nimg>self.maxNImg]] = False
		return keep

class SkyFlatSelector2014(object):
	def __init__(self):
		skyfile = lambda b: os.path.join('config',
		                                 'bokrm2014_darksky_%s.txt'%b)
		skytab = vstack([Table.read(skyfile(b),format='ascii') 
		                     for b in 'gi'])
		self.darkSkyFrames = skytab['fileName']
	def __call__(self,obsDb,ii):
		return np.in1d(obsDb['fileName'][ii],self.darkSkyFrames)

class GenericFlatSelector(object):
	minNImg = 15
	maxNImg = 100
	maxCounts = 20000
	def __init__(self):
		self.skyDat = Table.read(os.path.join('data','bokrm_skyadu.fits'))
	@staticmethod
	def __check_same(obsDb,ii1,ii2):
		return ( (obsDb['objName'][ii1] == obsDb['objName'][ii2]) &
		         (obsDb['filter'][ii1] == obsDb['filter'][ii2]) &
		         (obsDb['utDate'][ii1] == obsDb['utDate'][ii2]) )
	def __call__(self,obsDb,ii):
		keep = np.ones(len(ii),dtype=bool)
		# repeated images at same pointing center
		jj = np.arange(1,len(ii),2)
		keep[jj] ^= self.__check_same(obsDb,ii[jj],ii[jj-1])
		jj = np.arange(2,len(ii),2)
		keep[jj] ^= self.__check_same(obsDb,ii[jj],ii[jj-1])
		# pointings swamped with bright stars
		for field in ['rm10','rm11','rm12','rm13']:
			keep[obsDb['objName'][ii]==field] = False
		print 'selected ',','.join(list(obsDb['fileName'][ii[keep]]))
		return keep

class SkyFlatSelector(GenericFlatSelector):
	def __call__(self,obsDb,ii):
		keep = super(SkyFlatSelector,self).__call__(obsDb,ii)
		m = join(obsDb['frameIndex',][ii],self.skyDat,'frameIndex')
		assert np.all(m['frameIndex']==obsDb['frameIndex'][ii])
		keep &= m['skyMean'] < self.maxCounts
		if keep.sum() > self.maxNImg:
			jj = np.where(keep)[0]
			jj = jj[m['skyMean'][jj].argsort()]
			keep[jj[self.maxNImg:]] = False
		return keep

def config_rm_data(dataMap,args,season=None):
	if args.band is None:
		# set default for RM
		dataMap.setFilters(['g','i'])
	dataMap.setFringeFilters(['i'])
	if season:
		sfx = '_%s' % season
	else:
		sfx = ''
	dataMap.setCalMap('badpix','master',fileName='BadPixMask%s'%sfx)
	dataMap.setCalMap('badpix4','master',fileName='BadPixMask%s_x4'%sfx)
	dataMap.setCalMap('ramp','master',fileName='BiasRamp%s'%sfx)
	return dataMap

if __name__=='__main__':
	import sys
	import argparse
	parser = argparse.ArgumentParser()
	parser = bokpl.init_file_args(parser)
	parser = bokpl.init_pipeline_args(parser)
	parser.add_argument('--skyflatframes',action='store_true',
	                help='load only the sky flat frames')
	parser.add_argument('--makeobsdb',action='store_true',
	                help='make the observations database')
	parser.add_argument('--gaia',action='store_true',
	                help='use GAIA astrometry')
	parser.add_argument('--makebpmask',type=str,
	                help='make quick badpix mask from flat <FILENAME>')
	parser.add_argument('--plver',type=str,default=pipeVersion,
	                help='pipeline version (default is current)')
	args = parser.parse_args()
	args = set_rm_defaults(args,args.plver)
	if args.makeobsdb:
		# this needs to go here
		make_obs_db(args)
		sys.exit(0)
	dataMap = bokpl.init_data_map(args)
	season = get_observing_season(dataMap)
	dataMap = config_rm_data(dataMap,args)
	if args.gaia:
		sdir = 'scamp_refs_gaia'
	else:
		sdir = 'scamp_refs'
	dataMap.setScampRefCatDir(os.path.join(args.output,'..',sdir))
	if args.skyflatframes:
		# XXX should use classmethod
		if season=='2014':
			dataMap.setFileFilter(SkyFlatSelector2014())
		else:
			dataMap.setFileFilter(SkyFlatSelector())
	if args.makebpmask:
		build_mask_from_flat(dataMap('cal')(args.makebpmask),
		                     dataMap.getCalMap('badpix').getFileName(),
		                     dataMap.getCalDir())
	kwargs = {}
	kwargs['illum_filter_fun'] = IllumSelector()
#	kwargs['skyflat_selector'] = SkyFlatSelector(season)
	kwargs['header_fixes'] = build_headerfix_dict()
	bokpl.run_pipe(dataMap,args,**kwargs)

