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
		args.obsdb = os.path.join('config','sdssrm-bok2014.fits.gz')
	return args

def make_obs_db(args):
	# all Bok observations during RM nights (incl. IBRM)
	fullObsDbFile = os.path.join('config','sdssrm-allbok.fits')
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
		# 2. bad images
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
		#       lots of passing clouds with saturated ims this night
		bad = [ 'bokrm.20140219.%04d' % _i for _i in [93,144,145,146,147,
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
		#       #181 telescope was moving
		bad = [ 'bokrm.20140609.%04d' % _i for _i in [181] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       #171 telescope was moving
		bad = [ 'bokrm.20140610.%04d' % _i for _i in [171] ]
		good[np.in1d(obsDb['fileName'],bad)] = False
		#       lots of passing clouds with saturated ims this night
		bad = [ 'bokrm.20140612.%04d' % _i for _i in [88,89,90,102,103,104,
		                                              105,107,110,111,112,
		                                              113,114,115,119,120,
		                                              121,122,123] ]
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
	# all RM observations in 2014
	isrm2014 = isrm & (obsDb['mjd']<57000)
	rmObsDbFile = os.path.join('config','sdssrm-bok2014.fits')
	obsDb[isrm2014].write(rmObsDbFile,overwrite=True)

def load_darksky_frames(season,filt):
	tab = []
	for b in filt:
		t = Table.read(os.path.join('config',
		                            'bokrm%s_darksky_%s.txt'%(season,b)),
		               format='ascii')
		tab.append(t)
	t = vstack(tab)
	ut = t['utDate']
	del t['utDate']
	t['utDate'] = ut.astype('S8')
	return t

class IllumFilter(object):
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

if __name__=='__main__':
	import sys
	import argparse
	parser = argparse.ArgumentParser()
	parser = bokpl.init_file_args(parser)
	parser = bokpl.init_pipeline_args(parser)
	parser.add_argument('--darkskyframes',action='store_true',
	                help='load only the dark sky frames')
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
	if args.band is None:
		# set default for RM
		dataMap.setFilters(['g','i'])
	dataMap.setCalMap('badpix','master',fileName='BadPixMask')
	dataMap.setCalMap('badpix4','master',fileName='BadPixMask4')
	dataMap.setCalMap('ramp','master',fileName='BiasRamp')
	if args.gaia:
		sdir = 'scamp_refs_gaia'
	else:
		sdir = 'scamp_refs'
	dataMap.setScampRefCatDir(os.path.join(args.output,'..',sdir))
	if args.darkskyframes:
		filt = args.band if args.band else dataMap.getFilters()
		frames = load_darksky_frames('2014',filt)
		dataMap.setFileList(frames['utDate'],frames['fileName'])
	elif args.makebpmask:
		build_mask_from_flat(args.makebpmask,
		                     dataMap.getCalMap('badpix').getFileName(),
		                     dataMap.getCalDir())
	kwargs = {}
	kwargs['illum_filter_fun'] = IllumFilter()
	bokpl.run_pipe(dataMap,args,**kwargs)

