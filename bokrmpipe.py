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
	fc = load_field_centers()
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
	# wrong field name
	for fn in [ 'ut20150306/bokrm.20150306.%04d' % _i for _i in [43] ]:
		hdrfix[fn] = [ (0,{'OBJECT':'rm12'}) ]
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
	fn = 'ut20150110/d7032.0190'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':52.956,'CRVAL2':212.238}) 
	                   for j in range(1,17) ]
	fn = 'ut20150110/d7032.0191'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':52.956,'CRVAL2':212.238}) 
	                   for j in range(1,17) ]
	fn = 'ut20150110/d7032.0199'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':53.078}) for j in range(1,17) ]
	fn = 'ut20150209/d7062.0115'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':52.23816,'CRVAL2':211.98905}) 
	                   for j in range(1,17) ]
	fn = 'ut20150505/d7148.0107'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':53.07775,'CRVAL2':212.122625}) 
	                   for j in range(1,17) ]
	# the following are here for reference -- the pointing was totally off
	# and so these fields are useless. but got the coords from astrometry.net
	# so saving them here.
	# XX should insert a primary hdu OBJECT=unknown so astrom doesnt use refcat
	fn = 'ut20150505/d7148.0108'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':53.89656,'CRVAL2':209.18793}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150505/d7148.0109'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':53.89656,'CRVAL2':209.18893}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150505/d7148.0110'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':54.76956,'CRVAL2':209.24527}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150505/d7148.0111'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':54.76956,'CRVAL2':209.24527}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150505/d7148.0112'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':54.75356,'CRVAL2':209.32710}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150505/d7148.0113'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':54.75356,'CRVAL2':209.32710}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150506/d7149.0035' # has nice ngc galaxies
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':54.74856,'CRVAL2':212.10304}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150506/d7149.0036'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':54.74856,'CRVAL2':212.10304}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150506/d7149.0037'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':54.73356,'CRVAL2':212.18788}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150506/d7149.0038'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':54.73356,'CRVAL2':212.18788}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150506/d7149.0039' # this seems to have some overlap, keep
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':53.99256,'CRVAL2':213.7449}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150506/d7149.0040' # this seems to have some overlap, keep
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':53.99256,'CRVAL2':213.7449}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150506/d7149.0041' # way out again
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':55.82556,'CRVAL2':215.13028}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150506/d7149.0042' 
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':55.82556,'CRVAL2':215.13028}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150506/d7149.0049' 
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':fc['rm07'][1],'CRVAL2':fc['rm07'][0]}) 
	                   for j in range(1,17) ]
	for fn in [ 'ut20150506/d7149.%04d'  % _i for _i in [50,51] ]:
		hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':fc['rm08'][1],
		                          'CRVAL2':fc['rm08'][0]+4./60/.59}) 
		                   for j in range(1,17) ]
	for fn in [ 'ut20150506/d7149.%04d'  % _i for _i in [52,53] ]:
		hdrfix[fn] = [ (0,{'OBJECT':'rm10'}) ]
		hdrfix[fn] += [ ('IM%d'%j,{'CRVAL1':fc['rm10'][1],
		                           'CRVAL2':fc['rm10'][0]+4./60/.59}) 
		                   for j in range(1,17) ]
	for fn in [ 'ut20150506/d7149.%04d'  % _i for _i in [54,55] ]:
		hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':51.16356,'CRVAL2':213.74266})
		                   for j in range(1,17) ]
		hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	for fn in [ 'ut20150506/d7149.%04d'  % _i for _i in [56,57] ]:
		hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':50.20356,'CRVAL2':213.74238})
		                   for j in range(1,17) ]
		hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20150506/d7149.0069' 
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':fc['rm13'][1]+1/60.,
	                          'CRVAL2':fc['rm13'][0]-3.5/60/.59}) 
	                  for j in range(1,17) ]
	for fn in [ 'ut20150506/d7149.%04d'  % _i for _i in [70,71] ]:
		hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':fc['rm14'][1]+0.5/60,
		                          'CRVAL2':fc['rm14'][0]-2./60/.59}) 
		                  for j in range(1,17) ]
	fn = 'ut20150506/d7149.0072'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':53.91756,'CRVAL2':215.59415}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20160501/d7509.0141'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':53.30456,'CRVAL2':214.79687}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20160501/d7509.0142'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':54.11256,'CRVAL2':214.59420}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	fn = 'ut20160501/d7509.0143'
	hdrfix[fn] = [ ('IM%d'%j,{'CRVAL1':54.12256,'CRVAL2':214.61930-4./60/.59}) 
	                   for j in range(1,17) ]
	hdrfix[fn].insert(0,(0,{'OBJECT':'rm_unknown'}))
	return hdrfix

def make_obs_db(args):
	# first generate observations log for all Bok observations during RM 
	# nights (incl. IBRM or other ancillary data)
	fullObsDbFile = os.path.join(os.environ['BOK90PRIMERAWDIR'],
	                             'sdssrm-allbok.fits')
	if not os.path.exists(fullObsDbFile) or args.redo:
		utDirs = sorted(glob.glob(os.path.join(args.rawdir,'ut201[3-6]????')))
		print utDirs
		try:
			obsDb = Table.read(fullObsDbFile)
			print 'starting with existing ',fullObsDbFile
		except IOError:
			obsDb = None
		bokobsdb.generate_log(utDirs,fullObsDbFile,inTable=obsDb)
	# then read in the auto-generated table and then fix problems
	obsDb = Table.read(fullObsDbFile)
	# 1. files that are missing FILTER values
	missing = [ 'bokrm.20140314.%04d' % _i for _i in range(173,180) ]
	obsDb['filter'][np.in1d(obsDb['fileName'],missing)] = 'g'
	missing = [ 'bokrm.20140317.%04d' % _i for _i in [147,153] ]
	obsDb['filter'][np.in1d(obsDb['fileName'],missing)] = 'i'
	missing = [ 'bokrm.20140318.%04d' % _i for _i in [113] ]
	obsDb['filter'][np.in1d(obsDb['fileName'],missing)] = 'g'
	missing = [ 'bokrm.20140609.%04d' % _i for _i in [178] ]
	obsDb['filter'][np.in1d(obsDb['fileName'],missing)] = 'g'
	# 2. files with bad pointing that are way off from target field
	badpoint = [ 'd7148.%04d' % _i for _i in range(108,123) ]
	obsDb['objName'][np.in1d(obsDb['fileName'],badpoint)] = 'rm_unknown'
	badpoint = [ 'd7149.%04d' % _i for _i in range(35,42+1) ]
	obsDb['objName'][np.in1d(obsDb['fileName'],badpoint)] = 'rm_unknown'
	#        ....or skipped to next pointing for some reason
	badpoint = [ 'd7149.%04d' % _i for _i in range(52,53+1) ]
	obsDb['objName'][np.in1d(obsDb['fileName'],badpoint)] = 'rm10'
	badpoint = [ 'd7149.%04d' % _i for _i in range(54,57+1) ]
	obsDb['objName'][np.in1d(obsDb['fileName'],badpoint)] = 'rm_unknown'
	badpoint = [ 'd7149.%04d' % _i for _i in [72] ]
	obsDb['objName'][np.in1d(obsDb['fileName'],badpoint)] = 'rm_unknown'
	badpoint = [ 'd7509.%04d' % _i for _i in range(141,143+1) ]
	obsDb['objName'][np.in1d(obsDb['fileName'],badpoint)] = 'rm_unknown'
	# 3. flag bad images
	good = np.ones(len(obsDb),dtype=bool)
	#       test images with short exposure times
	good[(obsDb['imType']=='object')&(obsDb['expTime']<30)] = False
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
	#       double image, telescope jumped
	bad = [ 'd7062.%04d' % _i for _i in [137] ]
	good[np.in1d(obsDb['fileName'],bad)] = False
	#       trailed image
	bad = [ 'bokrm.20150307.%04d' % _i for _i in [9] ]
	good[np.in1d(obsDb['fileName'],bad)] = False
	#       passing clouds so images near saturation, bad gradients
	bad = [ 'bokrm.20150405.%04d' % _i for _i in [15,16,17,18,19,54,55] ]
	good[np.in1d(obsDb['fileName'],bad)] = False
	#       bad pointings, then trailed image (115); then bad sky (>=116)
	bad = [ 'd7148.%04d' % _i for _i in [#108,109,110,111,112,113,
	                                     115,118,119,120,121,122] ]
	good[np.in1d(obsDb['fileName'],bad)] = False
	#       bad read, looks like aborted exposure
	bad = [ 'd7467.%04d' % _i for _i in [101] ]
	good[np.in1d(obsDb['fileName'],bad)] = False
	# write the edited table
	obsDb['good'] = good
	# and now restrict to only RM observations
	iszero = obsDb['imType']=='zero'
	isflat = ( (obsDb['imType']=='flat') & 
	           ((obsDb['filter']=='g')|(obsDb['filter']=='i')) )
	isrmfield = np.array([n.startswith('rm') for n in obsDb['objName']])
	isrmfield &= (obsDb['imType']=='object')
	isrm = iszero | isflat | isrmfield
	rmObsDbFile = os.path.join('config','sdssrm-bok.fits')
	obsDb[isrm].write(rmObsDbFile,overwrite=True)
	if True:
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
		dataMap.groupByUtdFilt = True
	if args.makebpmask:
		build_mask_from_flat(dataMap('cal')(args.makebpmask),
		                     dataMap.getCalMap('badpix').getFileName(),
		                     dataMap.getCalDir())
	kwargs = {}
	kwargs['illum_filter_fun'] = IllumSelector()
#	kwargs['skyflat_selector'] = SkyFlatSelector(season)
	kwargs['header_fixes'] = build_headerfix_dict()
	bokpl.run_pipe(dataMap,args,**kwargs)

