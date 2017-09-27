#!/usr/bin/env python

import os
import glob
import numpy as np
from astropy.table import Table,vstack,join
from astropy.stats import sigma_clip

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

def load_field_centers():
	from astropy.coordinates import SkyCoord
	fieldcenters = {}
	with open('observe/bokrm_g_300s.txt') as pfile:
		for l in pfile:
			d = l.strip().split()
			f = d[3].strip("'")
			ra = d[6]
			ra = ra[:2]+'h'+ra[2:4]+'m'+ra[4:]+'s'
			dec = d[7]
			dec = dec[:3]+'d'+dec[3:5]+'m'+dec[6:]+'s'
			c = SkyCoord(ra+' '+dec)
			fieldcenters[f] = (c.ra.value,c.dec.value)
	return fieldcenters

def build_headerfix_dict():
	hdrfix = {}
	with open('config/badheaders.txt') as badhf:
		for l in badhf:
			fn,dat = l.strip().split()
			keyvals = {}
			for card in dat.split(','):
				k,v = card.split('=')
				if k.startswith('CRVAL'):
					keyvals[k] = float(v)
				else:
					keyvals[k] = v
			flist = [ (0, { k:v for k,v in keyvals.items() 
			                      if not k.startswith('CRVAL')} )]
			for j in range(1,17):
				flist.append( ('IM%d'%j, { k:v for k,v in keyvals.items()
				                                 if k.startswith('CRVAL')} ) )
			hdrfix[fn] = flist
	return hdrfix

def make_obs_db(args):
	# first generate observations log for all Bok observations during RM 
	# nights (incl. IBRM or other ancillary data)
	fullObsDbFile = os.path.join(os.environ['BOK90PRIMERAWDIR'],
	                             'sdssrm-allbok.fits')
	if not os.path.exists(fullObsDbFile) or args.redo:
		utDirs = sorted(glob.glob(os.path.join(args.rawdir,'ut201[3-7]????')))
		print utDirs
		try:
			obsDb = Table.read(fullObsDbFile)
			print 'starting with existing ',fullObsDbFile
		except IOError:
			obsDb = None
		bokobsdb.generate_log(utDirs,fullObsDbFile,inTable=obsDb)
	# then read in the auto-generated table and then fix problems
	obsDb = Table.read(fullObsDbFile)
	obsDb.sort('frameIndex')
	# flag bad exposures
	files = np.array([obs['utDir']+'/'+obs['fileName'] for obs in obsDb])
	badexps = [ l[:l.find(' ')] 
	               for l in open('config/badexposures.txt').readlines() ]
	obsDb['good'] = np.logical_not(np.in1d(files,badexps))
	assert (~obsDb['good']).sum() == len(badexps)
	hdr2tab = {'FILTER':'filter','OBJECT':'objName',
	           'CRVAL1':'targetDec','CRVAL2':'targetRa'}
	# and apply fixes to header values
	with open('config/badheaders.txt') as badhf:
		for l in badhf:
			fn,dat = l.strip().split()
			i = np.where(files==fn)[0][0]
			for card in dat.split(','):
				k,v = card.split('=')
				obsDb[hdr2tab[k]][i] = v
	# and now restrict to only RM observations
	iszero = obsDb['imType']=='zero'
	isflat = ( (obsDb['imType']=='flat') & 
	           ((obsDb['filter']=='g')|(obsDb['filter']=='i')) )
	isrmfield = np.array([n.startswith('rm') for n in obsDb['objName']])
	isrmfield &= (obsDb['imType']=='object')
	isrm = iszero | isflat | isrmfield
	rmObsDbFile = os.path.join('config','sdssrm-bok.fits')
	obsDb[isrm].write(rmObsDbFile,overwrite=True)
	if False:
		update_db_with_badsky()

def update_db_with_badsky(maxSkyCounts=40000):
	from astropy.io import fits
	rmObsDbFile = os.path.join('config','sdssrm-bok.fits')
	f = fits.open(rmObsDbFile,mode='update')
	skyDat = Table.read(os.path.join('data','bokrm_skyadu.fits'))
	ii = np.where(f[1].data['imType']=='object')[0]
	m = join(Table(f[1].data[ii])['frameIndex',],skyDat,'frameIndex')
	assert np.all(f[1].data['frameIndex'][ii]==m['frameIndex'])
	good = m['skyMean'] < maxSkyCounts
	f[1].data['good'][ii] &= good
	f.flush()
	f.close()
	m['frameIndex','utDate','fileName','filter','skyMean'][~good].write(
	     'data/badskylist.txt',format='ascii',overwrite=True)


def get_observing_season(dataMap):
	year = np.unique([ utd[:4] for utd in dataMap.getUtDates() ])
	if np.all(sorted(year) == ['2013','2014']):
		year = ['2014']
	elif len(year) > 1:
		raise ValueError("Currently processing only one observing season "
		                 "at a time is supported")
	return year[0]

class IllumSelector(object):
	minNImg = 10
	maxNImg = 30
	maxCounts = 20000
	def __init__(self):
		self.sky = Table.read(os.path.join('data','bokrm_skyadu.fits'))
	def __call__(self,obsDb,ii):
		keep = np.ones(len(ii),dtype=bool)
		if ( np.all(obsDb['utDate'][ii] == '20150405') or
		     obsDb['utDate'][ii[0]].startswith('2017') ):
			return self.filtered_selection(obsDb,ii)
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
	def filtered_selection(self,obsDb,ii):
		# filter out bright sky values
		t = join(obsDb[ii],self.sky,'frameIndex')
		if len(t) < len(ii):
			raise ValueError('missing files!')
		#
		badfield = np.in1d(obsDb['objName'][ii],
		                   ['rm10','rm11','rm12','rm13'])
		badfield |= obsDb['fileName'][ii] == 'bokrm.20150405.0059'
		fvar = t['skyRms']**2/t['skyMean']
		badrms = sigma_clip(fvar,sigma=3.0,iters=2).mask
		keep = (t['skyMean'] < self.maxCounts) & ~badrms & ~badfield
		if keep.sum() < self.minNImg:
			return ~badrms & ~badfield
		#
		grpNum = 0
		pgrp = [grpNum]
		for _i in range(1,len(ii)):
			if obsDb['objName'][ii[_i]] != obsDb['objName'][ii[_i-1]]:
				grpNum += 1
			pgrp.append(grpNum)
		pgrp = np.array(pgrp)
		#
		jj = np.where(keep)[0]
		keep[jj] = False
		jj2 = t['skyMean'][jj].argsort()[:self.maxNImg*2]
		_,jj3 = np.unique(pgrp[jj[jj2]],return_index=True)
		jj4 = t['skyMean'][jj[jj2[jj3]]].argsort()[:self.maxNImg]
		keep[jj[jj2[jj3[jj4]]]] = True
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
	parser.add_argument('--updatecaldb',action='store_true',
	                help='update calibration database')
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
	dataMap = bokpl.init_data_map(args,updatecaldb=args.updatecaldb)
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
		dataMap.groupByUtdFilt = True
	if args.makebpmask:
		build_mask_from_flat(dataMap('cal')(args.makebpmask),
		                     dataMap.getCalMap('badpix').getFileName(),
		                     dataMap.getCalDir())
	kwargs = {}
	kwargs['illum_filter_fun'] = IllumSelector()
#	kwargs['skyflat_selector'] = SkyFlatSelector(season)
	kwargs['header_fixes'] = build_headerfix_dict()
	if True:
		badccds= { 'ut20150205/bokrm.20150205.00%02d'%fr:
		           np.array([False,False,False,True]) for fr in range(76,87) }
		for f in ['ksb_170601_073801_ori','ksb_170601_074339_ori',
		          'ksb_170601_074916_ori','ksb_170601_075454_ori',
		          'ksb_170601_080031_ori']:
			badccds['ut20170601/'+f] = np.array([False,False,False,True])
		kwargs['gainMaskDb'] = { 'amp':{}, 'ccd':badccds }
	if True:
		kwargs['gainBalCfg'] = {'20160320':dict(splineOrder=1,
		                                        nSplineRejIter=2),
		                        '20160416':dict(splineOrder=1,
		                                        nSplineRejIter=2),
		                        '20160709':dict(gainTrendMethod='median')}
	bokpl.run_pipe(dataMap,args,**kwargs)

