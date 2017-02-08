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

# Nominal limits to classify night as "photometric"
zp_phot_nominal = {'g':25.90,'i':25.40}

def map_ids(ids1,ids2):
	ii = -np.ones(ids2.max()+1,dtype=int)
	ii[ids2] = np.arange(len(ids2))
	return ii[ids1]

def join_by_id(tab1,tab2,idkey):
	ii = map_ids(tab1[idkey],tab2[idkey])
	# avoid duplication
	tab2 = tab2.copy()
	for c in tab1.colnames:
		if c in tab2.colnames:
			del tab2[c]
	return hstack([tab1,tab2[ii]])

def join_by_frameid(tab1,tab2):
	return join_by_id(tab1,tab2,'frameIndex')

def join_by_objid(tab1,tab2):
	return join_by_id(tab1,tab2,'objId')

def aper_worker(dataMap,inputType,aperRad,refCat,catDir,catPfx,
                inp,**kwargs):
	utd,filt = inp
	redo = kwargs.pop('redo',False)
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
		if phot is None:
			print 'no apertures found!!!! ',f
			continue
		phot['frameIndex'] = dataMap.obsDb['frameIndex'][frame]
		allPhot.append(phot)
	allPhot = vstack(allPhot)
	allPhot.write(catFile,overwrite=True)

def aperture_phot(dataMap,refCat,procmap,inputType='sky',**kwargs):
	kwargs.setdefault('mask_is_weight_map',False)
	kwargs.setdefault('background','global')
	aperRad = np.concatenate([np.arange(2,9.51,1.5),[15.,22.5]])
	catDir = os.path.join(dataMap.procDir,'catalogs')
	if not os.path.exists(catDir):
		os.mkdir(catDir)
	catPfx = refCat['filePrefix']
	refCat = refCat['catalog']
	utdlist = [ (utd,filt) for utd in dataMap.iterUtDates() 
	                         for filt in dataMap.iterFilters() ]
	p_aper_worker = partial(aper_worker,dataMap,inputType,
	                        aperRad,refCat,catDir,catPfx,**kwargs)
	procmap(p_aper_worker,utdlist)

def load_full_ref_cat(phot,magRange=(17.,20.)):
	cdir = os.environ['BOKRMDIR']
	sdssRef = Table.read(cdir+'/data/sdssRefStars_DR13.fits.gz')
	metaDat = Table.read(cdir+'/data/BokRMFrameList.fits.gz')
	isgrange = (sdssRef['g']>magRange[0]) & (sdssRef['g']<=magRange[1])
	isirange = (sdssRef['i']>magRange[0]) & (sdssRef['i']<=magRange[1])
	sdssRef = sdssRef[isgrange|isirange]
	t = join_by_objid(phot,sdssRef['objId','g','i'])
	t = join_by_frameid(t,metaDat['frameIndex','aperZp','psfNstar','fwhmPix'])
#	t = join(phot,sdssRef['objId','g','i'],keys='objId')
#	t = join(t,metaDat['frameIndex','aperZp','psfNstar','fwhmPix'],
#	         keys='frameIndex')
	return t

def calc_color_terms(t,band,aperNum=-2,ref='sdss',airmassLim=1.2,
                     magRange=(17.,20.),colorRange=(0.4,3.0),
                     doplots=False,savefit=False):
	cdir = os.environ['BOKRMDIR']
	zpLim = zp_phot_nominal
	extCorr = {'g':0.17,'i':0.06} # SDSS values, should be close enough
	ii = np.arange(len(t))
	ccdj = t['ccdNum'] = 1
	refmag = t[band]
	refclr = t['g'] - t['i']
	inmag = (refmag > magRange[0]) & (refmag < magRange[1])
	incolor = (refclr > colorRange[0]) & (refclr < colorRange[1])
	mag = np.ma.array(t['counts'][:,aperNum],
	                  mask=((t['flags'][:,aperNum] > 0) |
	                        (t['filter'] != band) |
	                        (t['countsErr'][:,aperNum] <= 0) |
	                        (t['aperZp'][ii,ccdj] <= zpLim[band]) |
	                        (t['psfNstar'][ii,ccdj] < 100) |
	                        (t['fwhmPix'][ii,ccdj] > 4.0) |
	                        (t['airmass'] > airmassLim) |
	                        ~inmag | ~incolor))
	mag = -2.5*np.ma.log10(mag) - extCorr[band]*t['airmass']
	dmag = sigma_clip(mag-refmag)
	zp0 = dmag.mean()
	bokmag = mag - zp0
	dmag = bokmag - refmag
	# iteratively fit a polynomial of increasing order to the
	# magnitude differences
	mask = np.abs(dmag) > 0.25
	for iternum in range(3):
		order = iternum+1
		tmp_dmag = np.ma.array(dmag,mask=mask)
		cterms = np.ma.polyfit(refclr,tmp_dmag,order)
		magoff = sigma_clip(dmag-np.polyval(cterms,refclr))
		mask = magoff.mask
	if savefit:
		np.savetxt(cdir+'/config/bok2%s_%s_gicoeff.dat'%(ref,band),cterms)
	if doplots:
		import matplotlib.pyplot as plt
		from matplotlib import ticker
		gridsize = (50,20)
		ii = np.where(np.abs(dmag)<0.25)[0]
		fig = plt.figure(figsize=(10,6))
		fig.subplots_adjust(0.09,0.1,0.98,0.92)
		ax1 = fig.add_subplot(211)
		ax1.hexbin(refclr[ii],dmag[ii],#gridsize=gridsize,
		           cmap=plt.get_cmap('gray_r'),bins='log')
		ax1.axhline(0,c='c')
		xx = np.linspace(-1,5,100)
		if True and band=='g':
			desicterms = np.loadtxt(os.path.join(os.environ['BOKPIPE'],
			                        '..','survey','config',
			                        'bok2sdss_g_gicoeff.dat'))
			desicterms[-1] = 0.05055 # argh, removed zero level
			ax1.plot(xx[::3],np.polyval(desicterms,xx[::3]),'--',
			         c='orange',alpha=0.7)
		ax1.plot(xx,np.polyval(cterms,xx),c='r')
		ax1.set_ylabel('%s(Bok) - %s(%s)'%(band,band,ref.upper()))
		ax1.set_ylim(-0.25,0.25)
		order = len(cterms)-1
		polystr = ' '.join(['%+.5f*%s^%d'%(c,'gi',order-d) 
		                      for d,c in enumerate(cterms)])
		ax1.set_title(('$%s(Bok) - %s(%s) = '%(band,band,ref.upper())) +
		              polystr+'$',size=14)
		ax2 = fig.add_subplot(212)
		ax2.hexbin(refclr[ii],dmag[ii]-np.polyval(cterms,refclr[ii]),
		           #gridsize=gridsize,
		           cmap=plt.get_cmap('gray_r'),bins='log')
		ax2.axhline(0,c='r')
		ax2.set_ylim(-0.15,0.15)
		for ax in [ax1,ax2]:
			if ref=='sdss':
				ax.axvline(0.4,c='b')
				ax.axvline(3.0,c='b')
				ax.set_xlabel('SDSS g-i')
				ax.set_xlim(-0.05,3.3)
			else:
				ax.axvline(0.4,c='b')
				ax.axvline(2.7,c='b')
				ax.set_xlabel('PS1 g-i')
				ax.set_xlim(-0.05,2.8)
			ax.axvline(0,c='MidnightBlue')
			ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
			ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.02))
		if False:
			fig.savefig('%s_%s_to_bok_gicolors.png'%(band,ref))

def srcor(ra1,dec1,ra2,dec2,sep):
	from astropy.coordinates import SkyCoord,match_coordinates_sky
	from astropy import units as u
	c1 = SkyCoord(ra1,dec1,unit=(u.degree,u.degree))
	c2 = SkyCoord(ra2,dec2,unit=(u.degree,u.degree))
	idx,d2d,d3c = match_coordinates_sky(c1,c2)
	ii = np.where(d2d.arcsec < sep)[0]
	return ii,idx[ii],d2d.arcsec[ii]

class Sdss2BokTransform(object):
	colorMin = 0.4
	colorMax = 3.0
	def __init__(self,band):
		self.colorTerms = np.loadtxt('config/bok2sdss_%s_gicoeff.dat'%band)
	def __call__(self,mags,colors):
		corrMags = mags + np.polyval(self.colorTerms,colors)
		corrMags = np.ma.array(corrMags,
		                  mask=(colors<self.colorMin)|(colors>self.colorMax))
		return corrMags

def zp_worker(dataMap,aperCatDir,sdss,pfx,magRange,aperNum,inp):
	utd,filt = inp
	applyColorCorrection = True
	is_mag = ( (sdss[filt]>=magRange[0]) & (sdss[filt]<=magRange[1]) )
	ref_ii = np.where(is_mag)[0]
	aperCatFn = '.'.join([pfx+'_aper',utd,filt,'cat','fits'])
	files,frames = dataMap.getFiles(imType='object',utd=utd,filt=filt,
	                                with_frames=True)
	if files is None:
		return None
	bokutil.mplog('calculating zero points for %s [%d images]' % 
	               (utd,len(files)))
	sdss2bok = Sdss2BokTransform(filt)
	aperCat = fits.getdata(os.path.join(aperCatDir,aperCatFn))
	nAper = aperCat['counts'].shape[-1]
	aperCorrs = np.zeros((len(frames),nAper,4),dtype=np.float32)
	aperZps = np.zeros((len(frames),4),dtype=np.float32)
	aperZpRms = np.zeros((len(frames),4),dtype=np.float32)
	aperNstar = np.zeros((len(frames),4),dtype=np.int32)
	psfZps = np.zeros_like(aperZps)
	psfNstar = np.zeros_like(aperNstar)
	for n,(f,i) in enumerate(zip(files,frames)):
		#expTime =  dataMap.obsDb['expTime'][i]
		frameId =  dataMap.obsDb['frameIndex'][i]
		ii = np.where(aperCat['frameIndex']==frameId)[0]
		if len(ii)==0:
			print 'no data for frame ',f
			continue
		xCat = fits.open(dataMap('cat')(f))
		for ccd in range(1,5):
			# first for the aperture photometry
			c = np.where(aperCat['ccdNum'][ii]==ccd)[0]
			mask = ( (aperCat['counts'][ii[c],aperNum]<=0) |
			         (aperCat['flags'][ii[c],aperNum]>0) |
			         ~is_mag[aperCat['objId'][ii[c]]] )
			nstar = (~mask).sum()
			if nstar > 20:
				counts = np.ma.masked_array(
				            aperCat['counts'][ii[c],aperNum],mask=mask)
				aperMags = -2.5*np.ma.log10(counts)
				snr = counts / aperCat['countsErr'][ii[c],aperNum]
				aperMags[snr<15] = np.ma.masked
				jj = aperCat['objId'][ii[c]]
				refMags = sdss[filt][jj]
				if applyColorCorrection:
					refMags = sdss2bok(refMags,sdss['g'][jj]-sdss['i'][jj])
				dMag = sigma_clip(refMags-aperMags,sigma=2.2,iters=3)
				zp,ivar = np.ma.average(dMag,weights=snr**2,returned=True)
				M = np.float((~dMag.mask).sum())
				wsd = np.sqrt(np.sum((dMag-zp)**2*snr**2) /
				                   ((M-1)/M*np.sum(snr**-2)))
				aperNstar[n,ccd-1] = len(dMag.compressed())
				aperZps[n,ccd-1] = zp
				aperZpRms[n,ccd-1] = 1.0856 / wsd #* ivar**-0.5
				# now aperture corrections
				mask = ( (aperCat['counts'][ii[c]]<=0) |
				         (aperCat['flags'][ii[c]]>0) |
				         ~is_mag[aperCat['objId'][ii[c]]][:,np.newaxis] )
				counts = np.ma.masked_array(
				            aperCat['counts'][ii[c]],mask=mask)
				fratio = np.ma.divide(counts,counts[:,-1][:,np.newaxis])
				fratio = np.ma.masked_outside(fratio,0,1.5)
				fratio = sigma_clip(fratio,axis=0)
				invfratio = np.ma.power(fratio,-1)
				aperCorrs[n,:,ccd-1] = invfratio.mean(axis=0).filled(0)
			else:
				print 'WARNING: only %d stars for %s[%d]' % (nstar,f,ccd)
			# then for the sextractor PSF mags
			m1,m2,s = srcor(xCat[ccd].data['ALPHA_J2000'],
			                xCat[ccd].data['DELTA_J2000'],
			                sdss['ra'][ref_ii],sdss['dec'][ref_ii],2.5)
			if len(m1) > 20:
				refMags = sdss[filt][ref_ii[m2]]
				psfMags = xCat[ccd].data['MAG_PSF'][m1]
				dMag = sigma_clip(refMags-psfMags,sigma=2.2,iters=3)
				zp = np.ma.average(dMag)#,weights=snr**2)
				psfNstar[n,ccd-1] = len(dMag.compressed())
				# have to convert from the sextractor zeropoint
				psfZps[n,ccd-1] = 25 + zp
			else:
				print 'WARNING: only %d psf stars for %s[%d]' % (len(m1),f,ccd)
	aperCorrs = np.clip(aperCorrs,1,np.inf)
	tab = Table([np.repeat(utd,len(frames)),
	             dataMap.obsDb['frameIndex'][frames],
	             aperZps,aperZpRms,aperNstar,psfZps,psfNstar,aperCorrs],
	            names=('utDate','frameIndex',
	                   'aperZp','aperZpRms','aperNstar','psfZp','psfNstar',
	                   'aperCorr'),
	            dtype=('S8','i4','f4','f4','i4','f4','i4','f4'))
	return tab

def zero_points(dataMap,procmap,refCat,magRange=(17.,19.5),aperNum=-2):
	aperCatDir = os.path.join(dataMap.procDir,'catalogs')
	utdlist = [ (utd,filt) for utd in dataMap.iterUtDates() 
	                         for filt in dataMap.iterFilters() ]
	p_zp_worker = partial(zp_worker,dataMap,aperCatDir,
	                      refCat['catalog'],refCat['filePrefix'],
	                      magRange,aperNum)
	tabs = procmap(p_zp_worker,utdlist)
	tab = vstack(filter(None,tabs))
	tab.write('bokrm_zeropoints.fits',overwrite=True)

def match_to(ids1,ids2):
	idx = { j:i for i,j in enumerate(ids2) }
	return np.array([idx[i] for i in ids1])

def _read_old_catf(obsDb,catf):
	print 'loading ',catf
	dat1 = fits.getdata(catf,1)
	dat2 = fits.getdata(catf,2)
	idx = np.zeros(len(dat1),dtype=np.int32)
	for i,i1,i2 in zip(dat2['TINDEX'],dat2['i1'],dat2['i2']):
		idx[i1:i2] = i
	fns = [ f[:f.find('_ccd')] for f in dat1['fileName'] ]
	ii = match_to(fns,obsDb['fileName'])
	frameId = obsDb['frameIndex'][ii]
	t = Table([dat1['x'],dat1['y'],idx,
	           dat1['aperCounts'],dat1['aperCountsErr'],dat1['flags'],
	           dat1['ccdNum'],frameId],
	          names=('x','y','objId','counts','countsErr','flags',
	                 'ccdNum','frameIndex'))
	return t

def stack_catalogs(dataMap,refCat,old=False):
	pfx = refCat['filePrefix']
	if old:
		# renaming
		pfx = {'sdssstars':'sdssbright'}.get(pfx,pfx)
		aperCatDir = os.environ['HOME']+'/data/projects/SDSS-RM/rmreduce/catalogs_v2b/'
	else:
		aperCatDir = os.path.join(dataMap.procDir,'catalogs')
		pfx = pfx+'_aper'
	allTabs = []
	for utd in dataMap.iterUtDates():
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
				tab['filter'] = filt # handy to save this here
				allTabs.append(tab)
	#
	tab = vstack(allTabs)
	print 'stacked aperture phot catalogs into table with ',len(tab),' rows'
	tab.sort(['objId','frameIndex'])
	return tab

def calibrate_lightcurves(lcTab,dataMap,refCat,minNstar=70,
                          zpFile='bokrm_zeropoints.fits',outfn=None):
	if outfn is None:
		outfn = 'bokrmphot_%s.fits' % refCat['filePrefix']
	if lcTab is None:
		tab = stack_catalogs(dataMap,refCat)
	else:
		tab = lcTab
	apDat = Table.read(zpFile)
	ii = map_ids(tab['frameIndex'],apDat['frameIndex'])
	print '--> ',len(tab),len(ii)
	nAper = tab['counts'].shape[-1]
	apCorr = np.zeros((len(ii),nAper),dtype=np.float32)
	# cannot for the life of me figure out how to do this with indexing
	for apNum in range(nAper):
		apCorr[np.arange(len(ii)),apNum] = \
		            apDat['aperCorr'][ii,apNum,tab['ccdNum']-1]
	zp = np.ma.array(apDat['aperZp'][ii],mask=apDat['aperNstar'][ii]<minNstar)
	zp = zp[np.arange(len(ii)),tab['ccdNum']-1][:,np.newaxis]
	corrCps = tab['counts'] * apCorr 
	poscounts = np.ma.array(corrCps,mask=tab['counts']<=0)
	magAB = zp - 2.5*np.ma.log10(poscounts)
	magErr = 1.0856*np.ma.divide(tab['countsErr'],poscounts)
	tab['aperMag'] = magAB.filled(99.99)
	tab['aperMagErr'] = magErr.filled(0)
	# convert AB mag to nanomaggie
	fluxConv = 10**(-0.4*(zp-22.5))
	flux = corrCps * fluxConv
	fluxErr = tab['countsErr'] * apCorr * fluxConv
	tab['aperFlux'] = flux.filled(0)
	tab['aperFluxErr'] = fluxErr.filled(0)
	tab = join_by_frameid(tab,dataMap.obsDb['frameIndex','airmass','mjd'])
	print 'writing to ',outfn
	tab.write(outfn,overwrite=True)

def nightly_lightcurves(catName,lcs=None,redo=False):
	from collections import defaultdict
	filt = 'g'
	if lcs is None:
		lcs = Table.read('lightcurves_%s_%s.fits'%(catName,filt))
	else:
		lcs = lcs.copy()
	lcFn = 'nightly_lcs_%s_%s.fits' % (catName,filt)
	if os.path.exists(lcFn) and not redo:
		print lcFn,' already exists, exiting'
		return
	# mjd-0.5 gives a nightly UT date, but pad with 0.1 because 
	# ~6am MT is >noon UT
	lcs['mjdInt'] = np.int32(np.floor(lcs['mjd']-0.6))
	# group by object and then by night
	lcs = lcs.group_by(['objId','mjdInt'])
	cols = defaultdict(list)
	for obj_night in lcs.groups:
		objId = obj_night['objId'][0]
		fluxes = np.ma.masked_array(obj_night['aperFlux'],
		                       mask=((obj_night['flags']>4) |
		                             ~np.isfinite(obj_night['aperFlux'])))
		# sigma_clip barfs even when nan's are masked...
		fluxes.data[np.isnan(fluxes.data)] = 0
		fluxes = sigma_clip(fluxes,iters=2,sigma=4.0,axis=0)
		errs = np.ma.masked_array(obj_night['aperFluxErr'],
		                          mask=~np.isfinite(obj_night['aperFluxErr']))
		# see above
		errs.data[np.isnan(errs.data)] = 0
		ivars = errs**-2
		flux,ivar = np.ma.average(fluxes,weights=ivars,axis=0,returned=True)
		mjd = obj_night['mjd'].mean()
		err = np.ma.sqrt(ivar)**-1
		df = fluxes - flux
		wsd = np.sqrt(np.sum(ivars*df**2,axis=0)/np.sum(ivars,axis=0))
		cols['objId'].append(objId)
		cols['aperFlux'].append(flux.filled(0))
		cols['aperFluxErr'].append(err.filled(0))
		cols['aperFluxWsd'].append(wsd.filled(0))
		cols['mean_mjd'].append(mjd)
		cols['nObs'].append(fluxes.shape[0])
		cols['nGood'].append((~fluxes[:,3].mask).sum())
	tab = Table(cols,names=('objId','mean_mjd',
	                        'aperFlux','aperFluxErr','aperFluxWsd',
	                        'nObs','nGood'))
	flux = np.ma.masked_array(tab['aperFlux'],mask=tab['aperFlux']<=0)
	mag = 22.5 - 2.5*np.ma.log10(flux)
	tab['aperMag'] = mag.filled(99.99)
	err = 1.0856*np.ma.divide(tab['aperFluxErr'],flux)
	tab['aperMagErr'] = err.filled(99.99)
	tab.write(lcFn,overwrite=redo)

def map_group_to_items(tab):
	'''Given an astropy.table.Table that has been grouped, return the
	   set of indices that maps the groups back to the original table.
	   Useful for comparing aggregate values to individual measurements.
	   Example:
	     g = tab.group_by('object')
	     mean_mag = g['mag'].groups.aggregate(np.mean)
	     ii = map_group_to_items(g)
	     tab['delta_mag'] = g['mag'] - mean_mag[ii]
	'''
	return np.concatenate( 
	          [ np.repeat(i,np.diff(ii))
	              for i,ii in enumerate(zip(tab.groups.indices[:-1],
	                                        tab.groups.indices[1:])) ] )

def weighted_mean_mag(phot,magkey='mag',ivarkey='ivar',rms=True):
	'''Given a grouped table and keys for the columns containing 
	   magnitudes and inverse variances of the magnitudes, calculate
	   the inverse-variance weighted mean magnitude.
	   Also return the scatter in the magnitudes if rms=True.
	'''
	phot['wtmag'] = phot[magkey]*phot[ivarkey]
	aggphot = phot['wtmag',ivarkey].groups.aggregate(np.ma.sum)
	mean_mag = aggphot['wtmag']/aggphot[ivarkey]
	if not rms:
		return mean_mag
	else:
		photrms = phot[magkey].groups.aggregate(np.ma.std)
		return mean_mag,photrms

def load_masked_phot(lcTab,apNum=2,maskBits=None):
	phot = Table(lcTab,masked=True)
	phot.meta['APERNUM'] = apNum
	phot.meta['MASKBITS'] = str(maskBits)
	# select aperture (aggregate functions collapse along 2nd dim)
	phot['mag'] = phot['aperMag'][:,apNum]
	phot['err'] = phot['aperMagErr'][:,apNum]
	phot['flag'] = phot['flags'][:,apNum]
	# reduce to only the necessary columns
	phot = phot['frameIndex','objId','mjd','filter','mag','err','flag']
	# add masks
	if maskBits is None:
		maskBits = 2**32-1
	mask = (phot['mag']>90) | (phot['err']==0) | ((phot['flag']&maskBits)>0)
	phot['mag'].mask |= mask
	phot['err'].mask |= mask
	phot['ivar'] = np.ma.power(phot['err'],-2)
	return phot

def calc_aggregate_phot(phot,clip=True,sigma=3.0,iters=3):
	ii = map_group_to_items(phot)
	# have to do sigma-clipping by hand
	mean_mag,rms_mag = weighted_mean_mag(phot)
	if iters >= 1:
		phot['clipped_mag'] = phot['mag'].copy()
		phot['clipped_ivar'] = phot['ivar'].copy()
	for iterNum in range(iters):
		dmag = phot['clipped_mag'] - mean_mag[ii]
		dev = np.ma.abs(dmag/rms_mag[ii])
		reject = np.ma.greater(dev,sigma)
		phot['clipped_mag'].mask |= reject
		phot['clipped_ivar'].mask |= reject
		mean_mag,rms_mag = weighted_mean_mag(phot,
		                                     'clipped_mag','clipped_ivar')
	# get chi^2 using weighted mean
	phot['dmag'] = phot['mag'] - mean_mag[ii]
	phot['chival'] = phot['dmag']*np.ma.sqrt(phot['ivar'])
	phot['chi2'] = phot['dmag']**2*phot['ivar']
	phot['n'] = ~phot['mag'].mask
	if iters >= 1:
		dmag = phot['clipped_mag'] - mean_mag[ii]
		phot['clipped_chi2'] = dmag**2*phot['clipped_ivar']
		phot['clipped_n'] = ~phot['clipped_mag'].mask
	aggphot = phot['chi2','n',
	               'clipped_chi2','clipped_n'].groups.aggregate(np.ma.sum)
	mean_mag.name = 'mean_mag'
	rms_mag.name = 'rms_mag'
	aggphot.add_columns([phot.groups.keys['objId'],
	                     phot.groups.keys['filter'],
	                     mean_mag,rms_mag],[0,0,0,0])
	#
	aggphot['rchi2'] = aggphot['chi2']/(aggphot['n']-1)
	aggphot['clipped_rchi2'] = aggphot['clipped_chi2']/(aggphot['clipped_n']-1)
	# cleanup intermediate columns used during aggregation
	del phot['wtmag','clipped_mag','clipped_ivar','clipped_chi2','clipped_n']
	return phot,aggphot

def aggregate_phot_byobj(phot,**kwargs):
	phot = phot.group_by(['objId','filter'])
	return calc_aggregate_phot(phot,**kwargs)

def aggregate_phot_nightly(phot,**kwargs):
	# mjd-0.5 gives a nightly UT date, but pad with 0.1 because 
	# ~6am MT is >noon UT
	phot['mjdInt'] = np.int32(np.floor(phot['mjd']-0.6))
	phot = phot.group_by(['objId','filter','mjdInt'])
	return calc_aggregate_phot(phot,**kwargs)

def aggregate_phot(lcTab,refCat,which,**kwargs):
	phot = load_masked_phot(lcTab)
	if which=='all':
		phot,aggPhot = aggregate_phot_byobj(phot,**kwargs)
	elif which=='nightly':
		phot,aggPhot = aggregate_phot_nightly(phot,**kwargs)
	# XXX why aren't masks carrying through here?
	phot.write('photsum_%s_%s.fits' % (refCat['filePrefix'],which),
	           overwrite=True)
	aggPhot.write('agg_phot_%s_%s.fits' % (refCat['filePrefix'],which),
	              overwrite=True)

def find_star_outliers(starPhot,fthresh=10.0,othresh=5.0):
	chival = starPhot['chival'].filled(0)
	starPhot['outlier'] = np.abs(chival) > fthresh
	fgroup = starPhot.group_by('frameIndex')
	nFrameBad = fgroup['n','outlier'].groups.aggregate(np.sum)
	nFrameBad = hstack([fgroup.groups.keys,nFrameBad])
	nFrameBad['outlierFrac'] = ( nFrameBad['outlier'].astype(float) /
	                             nFrameBad['n'] )
	starPhot['outlier'] = np.abs(chival) > othresh
	ogroup = starPhot.group_by(['objId','filter'])
	nObjBad = ogroup['n','outlier'].groups.aggregate(np.sum)
	nObjBad = hstack([ogroup.groups.keys,nObjBad])
	nObjBad['outlierFrac'] = ( nObjBad['outlier'].astype(float) /
	                           nObjBad['n'] )
	return nFrameBad,nObjBad

# psum2 = psum.group_by('filter')['objId','mean_mag','rms_mag']
# merged = join(psum2.groups[0].filled(),psum2.groups[1].filled(),'objId',table_names=list(psum2.groups.keys['filter']))
# m = join(merged,sdss,'objId')
# scatter(m['g']-m['i'],m['mean_mag_g']-m['mean_mag_i'],s=1)

def load_catalog(catName):
	dataDir = os.path.join(os.environ['SDSSRMDIR'],'data')
	if catName == 'sdssrm':
		cat = Table.read(os.path.join(dataDir,'target_fibermap.fits'),1)
		cat.rename_column('RA','ra')
		cat.rename_column('DEC','dec')
		catPfx = 'rmqso'
	elif catName == 'allqsos':
		cat = Table.read(os.path.join(dataDir,'allqsos_rmfield.fits'),1)
		catPfx = 'allqso'
	elif catName == 'sdss':
		rmdir = os.environ['BOKRMDIR']
		catfn = os.path.join(rmdir,'data','sdssRefStars_DR13.fits.gz')
		print 'loading references from sdssRefStars_DR13.fits.gz'
		cat = Table.read(catfn,1)
		catPfx = 'sdssstars'
	elif catName == 'sdssold':
		cat = Table.read(os.path.join(dataDir,'sdss.fits'),1)
		catPfx = 'sdssstarsold'
	elif catName == 'cfht':
		cat = Table.read(os.path.join(dataDir,'CFHTLSW3_starcat.fits'),1)
		catPfx = 'cfhtstars'
	else:
		raise ValueError
	return dict(catalog=cat,filePrefix=catPfx)

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
	parser.add_argument('--lightcurves',action='store_true',
	                help='construct lightcurves')
	parser.add_argument('--aggregate',action='store_true',
	                help='construct aggregate photometry')
	parser.add_argument('--nightly',action='store_true',
	                help='construct nightly lightcurves')
	parser.add_argument('--zeropoint',action='store_true',
	                help='do zero point calculation')
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
	parser.add_argument('--outfile',type=str,
	                help='output file')
	args = parser.parse_args()
	args = bokrmpipe.set_rm_defaults(args)
	dataMap = bokpl.init_data_map(args)
	dataMap = bokrmpipe.config_rm_data(dataMap,args)
	refCat = load_catalog(args.catalog)
	if args.processes == 1:
		procmap = map
	else:
		pool = multiprocessing.Pool(args.processes)
		procmap = pool.map
	timerLog = bokutil.TimerLog()
	if args.aperphot:
		aperture_phot(dataMap,refCat,procmap,redo=args.redo,
		              background=args.background)
		timerLog('aper phot')
	elif args.lightcurves:
		if args.lctable:
			lcTab = Table.read(args.lctable)
		else:
			lcTab = None
		calibrate_lightcurves(lcTab,dataMap,refCat,zpFile=args.zptable,
		                      outfn=args.outfile)#,old=args.old)
		timerLog('lightcurves')
	elif args.aggregate:
		lcTab = Table.read(args.lctable)
		which = 'nightly' if args.nightly else 'all'
		aggregate_phot(lcTab,refCat,which)#,**kwargs)
#	elif args.nightly:
#		nightly_lightcurves(refCat['filePrefix'],redo=args.redo)
		timerLog('aggregate phot')
	elif args.zeropoint:
		zero_points(dataMap,procmap,refCat)
		timerLog('zeropoints')
	timerLog.dump()
	if args.processes > 1:
		pool.close()

