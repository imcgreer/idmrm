#!/usr/bin/env python

import os,sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.interpolate import LSQUnivariateSpline
from astropy.io import fits
from astropy.wcs import WCS
from astropy.table import Table,join,vstack
from astropy.stats import sigma_clip
from astropy.nddata import block_reduce

from bokpipe import bokphot,bokpl,bokgnostic
from bokpipe.bokproc import ampOrder,BokCalcGainBalanceFactors, \
                            BokImStatWithPreProcessing
import bokrmpipe
import bokrmphot

def plot_gain_vals(g,raw=False,debug=False):
	plt.figure(figsize=(12,8))
	plt.subplots_adjust(0.04,0.02,0.97,0.99,0.28,0.05)
	if True:
		bad = np.any(np.isnan(g['gainCor'].reshape(-1,32)),axis=1)
		if bad.sum() > 0:
			print "NaN's encountered in ",list(g['fileName'][bad])
		g = g[~bad]
	if raw:
		gains = np.ma.dstack([g['rawAmpGain'],
		                      np.repeat(g['rawCcdGain'],4,axis=1)])
		ampgaincor = g['ampTrend']
		ccdgaincor = g['ccdTrend']
	else:
		gains = np.ma.array(g['gains'],mask=g['gains']==0)
		ampgaincor = g['gainCor'][:,:,0]
		ccdgaincor = g['gainCor'][:,::4,1]
	if debug:
		gainBal = BokCalcGainBalanceFactors()
#		gainBal.nSplineRejIter = 2
#		gainBal.splineRejThresh = 1.5
		gainBal.ampRelGains = g['rawAmpGain']
		gainBal.ccdRelGains = g['rawCcdGain']
		gainBal.files = g['fileName']
		gainBal.filters = g['filter']
		badccds= { 'bokrm.20150205.00%02d'%fr:
		           np.array([False,False,False,True]) for fr in range(76,87) }
		gainBal.maskDb = { 'amp':{}, 'ccd':badccds }
		dbgGainCor = gainBal.calc_mean_corrections()
		dbg_ampgaincor = dbgGainCor[:,:,0]
		dbg_ccdgaincor = dbgGainCor[:,::4,1]
	axs = []
	nimg = gains.shape[0]
	seqno = np.arange(nimg,dtype=np.float32)
	for ccd in range(4):
		amp = 4*ccd
		ax = plt.subplot(5,4,ccd+1)
		for b,sym in zip('gi','so'):
			ii = np.where(g['filter']==b)[0]
			if len(ii)>0:
				ampgains = gains[ii,amp,1]
				jj = np.where(~ampgains.mask)[0]
				ax.plot(seqno[ii[jj]],ampgains[jj],'r'+sym,
				        ms=2.5,mfc='none',mec='r',mew=1.1)
				jj = np.where(ampgains.mask & ~(ampgains.data==0))[0]
				if len(jj)>0:
					ax.plot(seqno[ii[jj]].data,ampgains[jj],'rx',
					        ms=2.5,mfc='none',mec='r',mew=1.1)
		ax.plot(seqno,ccdgaincor[:,ccd],c='orange',ls='-',lw=1.4)
		if debug:
			ax.plot(seqno,dbg_ccdgaincor[:,ccd],color='g',ls='--',lw=1.4)
		ax.text(0.05,0.99,'CCD%d'%(ccd+1),
		        size=8,va='top',transform=ax.transAxes)
		medgain = np.median(ccdgaincor[:,ccd])
		ax.text(0.50,0.99,'%.3f'%medgain,color='red',
		        size=8,va='top',transform=ax.transAxes)
		ax.set_ylim(medgain-0.025,medgain+0.025)
		axs.append(ax)
	for amp in range(16):
		rowNum = amp//4
		colNum = amp%4
		ax = plt.subplot(5,4,4+(4*colNum+rowNum)+1)
		for b,sym in zip('gi','so'):
			ii = np.where(g['filter']==b)[0]
			if len(ii)>0:
				ampgains = gains[ii,amp,0]
				jj = np.where(~ampgains.mask)[0]
				ax.plot(seqno[ii[jj]],ampgains[jj],'b'+sym,
				        ms=2.5,mfc='none',mec='b',mew=1.1)
				jj = np.where(ampgains.mask & ~(ampgains.data==0))[0]
				if len(jj)>0:
					ax.plot(seqno[ii[jj]].data,ampgains[jj],'bx',
					        ms=2.5,mfc='none',mec='b',mew=1.1)
		ax.text(0.05,0.99,'IM%d'%ampOrder[amp],
		        size=8,va='top',transform=ax.transAxes)
		medgain = np.median(ampgaincor[:,amp])
		ax.text(0.25,0.99,'%.3f'%medgain,color='blue',
		        size=8,va='top',transform=ax.transAxes)
		ax.plot(seqno,ampgaincor[:,amp],c='purple',ls='-',lw=1.4)
		if debug:
			ax.plot(seqno,dbg_ampgaincor[:,amp],color='g',ls='--',lw=1.4)
		ax.set_ylim(medgain-0.025,medgain+0.025)
		axs.append(ax)
		logsky = np.log10(g['skys'][:,amp])
		rax = ax.twinx()
		rax.plot(logsky,c='0.2',alpha=0.8,ls='-.',lw=1.5)
		rax.tick_params(labelsize=8)
		rax.set_ylim(0.99*logsky.min(),1.01*logsky.max())
	for ax in axs:
		ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.01))
		ax.xaxis.set_visible(False)
		ax.tick_params(labelsize=8)
		ax.set_xlim(-1,g['gains'].shape[0]+1)
		#ax.set_ylim(0.945,1.055)

def load_gain_data(gfile,obsDb):
	gdat = Table(dict(np.load(gfile)),masked=True) 
	gdat.rename_column('files','fileName')
	gdat = join(gdat,obsDb['fileName','filter','utDate'],'fileName')
	for c in ['gains','rawAmpGain','rawCcdGain']:
		gdat[c][gdat[c]==0] = np.ma.masked
	return gdat

def all_gain_vals(diagdir,obsDb,utDates=None):
	from glob import glob
	if utDates is not None:
		gfiles = [ os.path.join(diagdir,'gainbal_%s.npz'%utd) 
		              for utd in utDates ]
	else:
		gfiles = sorted(glob(os.path.join(diagdir,'gainbal*.npz')))
	tabs = []
	for gfile in gfiles:
		try:
			tabs.append(load_gain_data(gfile,obsDb))
		except:
			print 'failed to load %s' % gfile
			if False:
				sys.exit(1)
	return vstack(tabs)

def all_gain_plots(gainDat=None,diagdir=None,obsDb=None,utDates=None,
                   raw=False,debug=False,pdfFile=None):
	from matplotlib.backends.backend_pdf import PdfPages
	if pdfFile is None:
		pdfFile = 'bok_gain_vals.pdf'
	plt.ioff()
	if gainDat is None:
		gainDat = all_gain_vals(diagdir,obsDb,utDates=utDates)
	gainDat = gainDat.group_by('utDate')
	with PdfPages(pdfFile) as pdf:
		for kdat,gdat in zip(gainDat.groups.keys,gainDat.groups):
			print 'plotting gains for ',kdat['utDate']
			plot_gain_vals(gdat,raw=raw,debug=debug)
			plt.title('%s'%kdat['utDate'])
			pdf.savefig()
			plt.close()
	plt.ion()

def calc_sky_backgrounds(dataMap,outputFile,redo=False):
	extns = ['IM4']
	imstat = BokImStatWithPreProcessing(extensions=extns,
	                   stats_region='amp_central_quadrant',stats_stride=8)
	skyvals = []
	files,ii = dataMap.getFiles(imType='object',includebad=True,
	                            with_frames=True)
	if os.path.exists(outputFile) and not redo:
		oldTab = Table.read(outputFile)
		inOld = np.in1d(dataMap.obsDb['frameIndex'][ii],oldTab['frameIndex'])
		files = files[~inOld]
		ii = ii[~inOld]
	else:
		oldTab = None
	rawfiles = map(dataMap('raw'),files)
	imstat.process_files(rawfiles)
	sky = imstat.data['mean'][:,0].astype(np.float32)
	tab = Table(dict(frameIndex=dataMap.obsDb['frameIndex'][ii],
	                 utDate=dataMap.obsDb['utDate'][ii],
	                 fileName=dataMap.obsDb['fileName'][ii],
	                 filter=dataMap.obsDb['filter'][ii],
	                 skyMean=sky))
	if oldTab is not None:
		tab = vstack([oldTab,tab])
	tab.write(outputFile,overwrite=True)
	logf = open('data/bokrm_skysum.txt','w')
	tab = Table.read(outputFile)
	tab = tab.group_by(['utDate','filter'])
	for k,g in zip(tab.groups.keys,tab.groups):
		sky = g['skyMean']
		logf.write('%8s %3s %4d %10.1f %10.1f %10.1f\n' % \
		           (k['utDate'],k['filter'],len(g),
		            sky.mean(),sky.min(),sky.max()))
	logf.close()

def id_sky_frames(obsDb,skytab,utds,thresh=10000.):
	frametab = obsDb['frameIndex','utDate','fileName','objName'].copy()
	ii = np.where(np.in1d(frametab['utDate'],utds))[0]
	skytab = join(skytab,frametab[ii],'fileName')
	assert np.all(skytab['utDate_1']==skytab['utDate_2'])
	del skytab['utDate_2']
	skytab.rename_column('utDate_1','utDate')
	# first cut the repeated pointings
	ii = np.where((skytab['objName'][:-1]==skytab['objName'][1:]) & 
	              (skytab['utDate'][:-1]==skytab['utDate'][1:]))[0]
	skytab.remove_rows(1+ii)
	# then cut on the sky threshold
	ii = np.where(skytab['skyMean'] < thresh)[0]
	skytab = skytab[ii]
	return skytab

def id_sky_frames_2014():
	obsDb = bokpl._load_obsdb('config/sdssrm-bok2014.fits.gz')
	darkNights = {
	  'g':['20140126','20140128','20140312','20140424','20140426',
	       '20140427','20140518','20140630','20140702','20140718'],
	  'i':['20140123','20140126','20140129','20140425','20140428',
	       '20140701','20140717'],
	}
	for filt in 'gi':
		skytab = Table.read('data/bokrm2014sky%s.fits.gz'%filt)
		skyframes = id_sky_frames(obsDb,skytab,darkNights[filt],
		                          thresh={'g':1500.,'i':5000.}[filt])
		outf = 'config/bokrm2014_darksky_%s.txt' % filt
		skyframes['skyMean'].format = '{:8.1f}'
		skyframes['utDate','fileName','skyMean'].write(outf,format='ascii')

def check_img_astrom(imgFile,refCat,catFile=None,mlim=19.5,band='g'):
	imFits = fits.open(imgFile)
	if catFile is None:
		catFile = imgFile.replace('.fits','.cat.fits')
	catFits = fits.open(catFile)
	try:
		ahead = bokastrom.read_headers(imgFile.replace('.fits','.ahead'))
	except:
		ahead = None
	rv = []
	for ccd in range(1,5):
		ccdCat = catFits[ccd].data
		hdr = imFits[ccd].header
		if ahead is not None:
			hdr.update(ahead[ccd-1].items())
		try:
			w = WCS(hdr)
		except:
			rv.append(None)
			continue
		foot = w.calc_footprint()
		ras = sorted(foot[:,0])
		decs = sorted(foot[:,1])
		ii = np.where((refCat['ra']>ras[1])&(refCat['ra']<ras[2]) &
		              (refCat['dec']>decs[1])&(refCat['dec']<decs[2]) &
		              True)[0]
		              #(refCat[band]<mlim))[0]
		if len(ii) < 10:
			rv.append(None)
			continue
		m1,m2,sep = bokrmphot.srcor(ccdCat['ALPHA_J2000'],ccdCat['DELTA_J2000'],
		                            refCat['ra'][ii],refCat['dec'][ii],5.0)
		rv.append(dict(N=len(ii),nMatch=len(ii),
		               ra=ccdCat['ALPHA_J2000'][m1],
		               dec=ccdCat['DELTA_J2000'][m1],
		               raRef=refCat['ra'][ii[m2]],
		               decRef=refCat['dec'][ii[m2]],
		               sep=sep))
	return rv

def rmobs_meta_data(dataMap):
	tabf = 'data/BokRMFrameList.fits'
	bokgnostic.obs_meta_data(dataMap,outFile=tabf)
	tab = Table.read(tabf)
	for b in 'g':
		zpdat = Table.read('bokrm_zeropoints.fits')
		del zpdat['utDate']
		tab = join(tab,zpdat,'frameIndex','left')
	tab.write(tabf,overwrite=True)

def dump_data_summary(dataMap,splitrm=False):
	for utd in dataMap.iterUtDates():
		files,ii = dataMap.getFiles(with_frames=True)
		ntotal = len(ii)
		nbiases = np.sum(dataMap.obsDb['imType'][ii] == 'zero')
		missing = []
		if nbiases<=3: missing.append('nobias')
		print '%8s %7d %7d ' % (utd,ntotal,nbiases),
		nflt = {}
		for filt in 'gi':
			nflats = np.sum( (dataMap.obsDb['imType'][ii] == 'flat') &
			                 (dataMap.obsDb['filter'][ii] == filt) )
			print '%7d ' % nflats,
			nflt[filt] = nflats
		isrm = np.array([n.startswith('rm') 
		                    for n in dataMap.obsDb['objName'][ii]])
		for filt in 'gi':
			nsci = np.sum( (dataMap.obsDb['imType'][ii] == 'object') &
			               (dataMap.obsDb['filter'][ii] == filt) )
			nrm = np.sum( (dataMap.obsDb['imType'][ii] == 'object') &
			              (dataMap.obsDb['filter'][ii] == filt) & isrm )
			if splitrm:
				print '%7d %7d' % (nsci,nrm),
			else:
				print '%7d ' % (nrm),
			if nrm>0 and nflt[filt]<=3: missing.append('no%sflat'%filt)
		print '  ',','.join(missing)

def check_processed_data(dataMap):
	import fitsio
	sdss = fits.getdata(os.environ['BOK90PRIMEDIR']+'/../data/sdss.fits',1)
	gaia = fits.getdata(os.environ['BOK90PRIMEOUTDIR'] +
	                    '/scamp_refs_gaia/gaia_sdssrm.fits',1)
	try:
		zeropoints = fits.getdata('bokrm_zeropoints.fits')
	except IOError:
		zeropoints = None
	tabf = open(os.path.join('proc_diag.html'),'w')
	tabf.write(bokgnostic.html_diag_head)
	rowstr = ''
	files_and_frames = dataMap.getFiles(imType='object',with_frames=True)
	for f,i in zip(*files_and_frames):
		frameId = dataMap.obsDb['frameIndex'][i]
		filt = dataMap.obsDb['filter'][i]
		rowstr = ''
		procf = dataMap('proc2')(f)
		rowstr += bokgnostic.html_table_entry('%d'%frameId,'nominal')
		rowstr += bokgnostic.html_table_entry(f,'nominal')
		print procf
		try:
			hdr0 = fitsio.read_header(procf,ext=0)
			for k in ['OSCNSUB','CCDPROC','CCDJOIN','CCDPRO2','SKYSUB']:
				if k in hdr0:
					rowstr += bokgnostic.html_table_entry(r'&#10004;',
					                                      'nominal')
				else:
					rowstr += bokgnostic.html_table_entry(r'&#9747;',
					                                      'bad')
				status = 'nominal' if k in hdr0 else 'missing'
		except:
			print procf,' does not exist'
			for k in ['OSCNSUB','CCDPROC','CCDJOIN','CCDPRO2','SKYSUB']:
				rowstr += bokgnostic.html_table_entry('','missing')
		try:
			zpi = np.where(dataMap.obsDb['frameIndex'][i] ==
			                                zeropoints['frameIndex'])[0][0]
		except:
			zpi = None
		for ccdi in range(4):
			if zpi is not None:
				zp = zeropoints['aperZp'][zpi,ccdi]
				if zp > bokrmphot.zp_phot_nominal[filt]:
					status = 'nominal'
				elif zp > bokrmphot.zp_phot_nominal[filt]-0.2:
					status = 'warning'
				elif zp > bokrmphot.zp_phot_nominal[filt]-0.5:
					status = 'bad'
				else:
					status = 'weird'
				# psfNstar tells you how many catalog detections matched,
				# which is a better indication of failures
				nstar = zeropoints['psfNstar'][zpi,ccdi]
				if nstar < 20:
					status2 = 'bad'
				elif nstar < 100:
					status2 = 'warning'
				else:
					status2 = 'nominal'
			else:
				zp = 0.0
				status = 'missing'
				nstar = 0
				status2 = 'missing'
			rowstr += bokgnostic.html_table_entry('%.2f'%zp,status)
			rowstr += bokgnostic.html_table_entry('%d'%nstar,status2)
		catf = dataMap('cat')(f)
		try:
			m = check_img_astrom(procf,gaia,catFile=catf)
			for c in m:
				if c is None:
					rowstr += bokgnostic.html_table_entry('BAD','bad')
					continue
				sep = np.median(c['sep'])
				if sep > 0.4:
					status = 'bad'
				elif sep > 0.2:
					status = 'warning'
				elif sep > 0.0:
					status = 'nominal'
				else:
					status = 'weird'
				rowstr += bokgnostic.html_table_entry('%.3f'%sep,status)
		except IOError:
			for i in range(4):
				rowstr += bokgnostic.html_table_entry('','missing')
		tabf.write(r'<tr>'+rowstr+r'</tr>'+'\n')
		tabf.flush()
	tabf.write(bokgnostic.html_diag_foot)
	tabf.close()

# list of flats that get missed by the search below but are needed to have
# a full set for each run
_extra_flats = [ ('20150205','g'), ('20150210','i'), ('20150211','g'),
                 ('20150306','i'), ('20160320','g'), ('20160320','i'),  
                 ('20160501','g')
]

def find_bass_cals(bassLog):
	localpath = os.environ['BOK90PRIMERAWDIR']
	nerscpath = os.environ['NERSCDTN']+':'+'bok/BOK_Raw'
	obsDb = bokpl._load_obsdb('config/sdssrm-allbok.fits')
	dataMap = bokpl.BokDataManager(obsDb,'.','.')
	isbassbias = bassLog['imType'] == 'zero'
	isbassgflat = ( (bassLog['imType'] == 'flat') & 
	                (bassLog['filter'] == 'g') )
	isbassiflat = ( (bassLog['imType'] == 'flat') & 
	                (bassLog['filter'] == 'i') )
	xfer = False
	for utd in dataMap.iterUtDates():
		if utd.startswith('2014'): continue
		isbassutd = bassLog['utDate'] == utd
		files,ii = dataMap.getFiles(with_frames=True)
		nbiases = np.sum(dataMap.obsDb['imType'][ii] == 'zero')
		if nbiases < 10:
			xfer |= isbassutd & isbassbias
		ngflats = np.sum( (dataMap.obsDb['imType'][ii] == 'flat') &
		                  (dataMap.obsDb['filter'][ii] == 'g') )
		niflats = np.sum( (dataMap.obsDb['imType'][ii] == 'flat') &
		                  (dataMap.obsDb['filter'][ii] == 'i') )
		isrm = np.array([n.startswith('rm') 
		                     for n in dataMap.obsDb['objName'][ii]])
		ngrm = np.sum( (dataMap.obsDb['imType'][ii] == 'object') &
		               (dataMap.obsDb['filter'][ii] == 'g') & isrm )
		nirm = np.sum( (dataMap.obsDb['imType'][ii] == 'object') &
		               (dataMap.obsDb['filter'][ii] == 'i') & isrm )
		if ngflats < 10 and ngrm > 0:
			xfer |= isbassutd & isbassgflat
		if niflats < 10 and nirm > 0:
			xfer |= isbassutd & isbassiflat
	for utd,b in _extra_flats:
		xfer |= ( (bassLog['utDate'] == utd) & 
		          (bassLog['imType'] == 'flat') & 
		          (bassLog['filter'] == b) )
	l = bassLog
	for i in np.where(xfer)[0]:
		remotef = nerscpath+'/'+l['utDir'][i]+'/'+l['fileName'][i]+'.fits.fz'
		localdir = localpath+'/ut'+l['utDate'][i]
		if not os.path.exists(localdir):
			os.mkdir(localdir)
		localf = localdir+'/'+l['DTACQNAM'][i]+'.fz'
		if not os.path.exists(localf):
			cmd = ['scp',remotef,localf]
			print ' '.join(cmd)
			subprocess.call(cmd)

def check_bias_ramp(dataMap):
	dataMap.setUtDates(['20140123','20140415','20140427'])
	dataMap.setFilters('g')
	rdxdir = dataMap.getProcDir()
	tmpdir = dataMap.getTmpDir()
	ii = np.arange(2016)
	rampFits = fits.open(os.path.join(dataMap.getCalDir(),'BiasRamp.fits'))
	biasMap = dataMap.getCalMap('bias')
	for utd in dataMap.iterUtDates():
		if utd != '20140415': continue
		files = dataMap.getFiles(imType='object')
		biasMap.setTarget(files[0])
		n = len(files) // 4
		grps = [files[i:i+n] for i in range(0,len(files),n)]
		for grp in grps:
			plt.figure(figsize=(12,5))
			ax1 = plt.subplot(211)
			ax2 = plt.subplot(212)
			for extn in ['IM9']:
				v1,v2 = [],[]
				for f in grp:
					im_pre = fits.getdata(os.path.join(rdxdir,f+'.fits'),extn)
					im_cor = fits.getdata(os.path.join(tmpdir,f+'.fits'),extn)
					sky_pre = np.median(im_pre[1024:1042,1024:1042])
					sky_cor = np.median(im_cor[1024:1042,1024:1042])
					print f,sky_pre,sky_cor
					ax1.plot(im_pre[ii,ii]-sky_pre,c='0.6',alpha=0.5,lw=0.5)
					ax2.plot(im_cor[ii,ii]-sky_cor,c='0.6',alpha=0.5,lw=0.5)
					v1.append(im_pre[ii,ii]-sky_pre)
					v2.append(im_cor[ii,ii]-sky_cor)
				v1 = np.median(v1,axis=0)
				v2 = np.median(v2,axis=0)
				ramp = rampFits[extn].data
				bias = biasMap.getImage(extn)
				ax1.plot(v1,c='k',lw=0.5)
				ax2.plot(v2,c='k',lw=0.5)
				ax1.plot(ramp[ii,ii],c='r',lw=1.0)
				ax1.plot(bias[ii,ii],c='orange',lw=1.0)
				for ax in [ax1,ax2]:
					ax.axhline(0,c='DarkCyan',ls='--',lw=1.5)
					ax.set_xlim(0,2016)
					ax.set_ylim(-150,150)
			#break
		break
	plt.show()

def image_cutouts(dataMap,photCat,band='g',objid=None,old=False):
	from astropy.nddata import Cutout2D
	objs = photCat.bokPhot
	try:
		objs['frameIndex'] = objs['frameId']
	except:
		pass
	if objid is not None:
		obj_ii = np.where(objs['objId']==objid)[0]
		objs = objs[obj_ii]
	objs = bokrmphot.join_by_frameid(objs,
	                      dataMap.obsDb['frameIndex','utDir','fileName'])
	objs = objs.group_by(['objId','filter'])
	if old:
		datadir = '/media/ian/stereo/data/projects/SDSS-RM/rmreduce/'
	else:
		#datadir = '/d2/data/projects/SDSS-RM/RMpipe/bokpipe_v0.1'
		datadir = '/d2/data/projects/SDSS-RM/RMpipe/bokpipe_v0.3'
	size = 65
	if old:
		outdir = 'bokcutouts_old/'
	else:
		outdir = 'bokcutouts/'
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	for k,obj in zip(objs.groups.keys,objs.groups):
		objId = k['objId']
		band = k['filter']
		cutfile = outdir+'bok%s%03d_%s.fits'%(photCat.name,objId,band)
		if os.path.exists(cutfile) or len(obj)==0:
			continue
		hdus = [ fits.PrimaryHDU() ]
		for i,obs in enumerate(obj):
			sys.stdout.write('\rRM%03d %4d/%4d' % 
			                 (objId,(i+1),len(obj)))
			sys.stdout.flush()
			ccdNum = obs['ccdNum']
			if old:
				fn = os.path.join(datadir,obs['utDir'],'ccdproc3',
				                  obs['fileName']+'_ccd%d.fits'%ccdNum)
				im,hdr = fits.getdata(fn,header=True)
				hdr0 = hdr
			else:
				fn = os.path.join(datadir,obs['utDir'],
				                  obs['fileName']+'.fits')
				im,hdr = fits.getdata(fn,'CCD%d'%ccdNum,header=True)
				hdr0 = fits.getheader(fn,0)
			wcs = WCS(hdr)
			cutobj = Cutout2D(im,(obs['x'],obs['y']),size,mode='partial',
			                  wcs=wcs,fill_value=0)
			newhdr = cutobj.wcs.to_header()
			newhdr['OBJECT'] = '_'.join([hdr0['DATE-OBS'],hdr0['UTC-OBS'][:5],
			                             hdr0['OBJECT']])
			hdus.append(fits.ImageHDU(cutobj.data,newhdr))
		fits.HDUList(hdus).writeto(cutfile)
	print

def image_thumbnails(dataMap,photCat,band='g',objid=None,
                     nbin=None,old=False,trim=None):
	from astropy.visualization import ZScaleInterval
	from matplotlib.backends.backend_pdf import PdfPages
	# load object database
	objs = photCat.bokPhot
	try:
		objs['frameIndex'] = objs['frameId']
	except:
		pass
	if objid is not None:
		obj_ii = np.where(objs['objId']==objid)[0]
		objs = objs[obj_ii]
	tmpObsDb = dataMap.obsDb.copy()
	tmpObsDb['mjd_mid'] = tmpObsDb['mjd'] + (tmpObsDb['expTime']/2)/(3600*24.)
	objs = bokrmphot.join_by_frameid(objs,tmpObsDb)
	objs = objs.group_by(['objId','filter'])
	# configure figures
	nrows,ncols = 8,6
	figsize = (7.0,10.25)
	subplots = (0.11,0.07,0.89,0.93,0.00,0.03)
	size = 65
	zscl = ZScaleInterval()
	nplot = nrows*ncols
	if old:
		outdir = 'bokcutouts_old/'
	else:
		outdir = 'bokcutouts/'
	ccdcolors = ['darkblue','darkgreen','darkred','darkmagenta']
	if True and photCat.name=='rmqso':
		diffphot = Table.read('bok%s_photflags.fits'%band)
		errlog = open('bokflags_%s_err.log'%band,'a')
		bitstr = [ 'TinyFlux','BigFlux','TinyErr','BigErr','BigOff']
		frameid = np.zeros(diffphot['MJD'].shape,dtype=np.int32)
		for i in range(len(diffphot)):
			jj = np.where(diffphot['MJD'][i]>0)[0]
			for j in jj:
				dt = diffphot['MJD'][i,j]-tmpObsDb['mjd_mid']
				_j = np.abs(dt).argmin()
				if np.abs(dt[_j]) > 5e-4:
					raise ValueError("no match for ",i,diffphot['MJD'][i,j])
				else:
					frameid[i,j] = tmpObsDb['frameIndex'][_j]
		diffphot['frameIndex'] = frameid
		matched = diffphot['MJD'] == 0 # ignoring these
	plt.ioff()
	for k,obj in zip(objs.groups.keys,objs.groups):
		objId = k['objId']
		_band = k['filter']
		if _band != band: continue
		if photCat.name=='rmqso' and objId >= 850:
			break
		cutfile = outdir+'bok%s%03d_%s.fits' % (photCat.name,
		                                        obj['objId'][0],band)
		pdffile = cutfile.replace('.fits','.pdf')
		if os.path.exists(pdffile) or len(obj)==0:
			continue
		pdf = PdfPages(pdffile)
		cutfits = fits.open(cutfile)
		# number cutouts matches number observations
		if len(cutfits)-1 != len(obj):
			errlog.write('[RM%03d]: %d cutouts, %d obs; skipping\n' %
			              (obj['objId'][0],len(cutfits)-1,len(obj)))
		pnum = -1
		for i,(obs,hdu) in enumerate(zip(obj,cutfits[1:])):
			sys.stdout.write('\rRM%03d %4d/%4d' % 
			                 (obs['objId'],(i+1),len(obj)))
			sys.stdout.flush()
			ccdNum = obs['ccdNum']
			cut = hdu.data
			try:
				z1,z2 = zscl.get_limits(cut[cut>0])
			except:
				try:
					z1,z2 = np.percentile(cut[cut>0],[10,90])
				except:
					z1,z2 = cut.min(),cut.max()
			if not old:
				# rotate to N through E
				if ccdNum==1:
					cut = cut[:,::-1]
				elif ccdNum==2:
					cut = cut[::-1,::-1]
				elif ccdNum==3:
					pass
				elif ccdNum==4:
					cut = cut[::-1,:]
				# except now flip x-axis so east is right direction
				cut = cut[:,::-1]
			if trim is not None:
				cut = cut[trim:-trim,trim:-trim]
			if nbin is not None:
				cut = block_reduce(cut,nbin,np.mean)
			#
			if pnum==nplot+1 or pnum==-1:
				if pnum != -1:
					pdf.savefig()
					plt.close()
				plt.figure(figsize=figsize)
				plt.subplots_adjust(*subplots)
				pnum = 1
			ax = plt.subplot(nrows,ncols,pnum)
			plt.imshow(cut,origin='lower',interpolation='nearest',
			           vmin=z1,vmax=z2,cmap=plt.cm.gray_r,aspect='equal')
			framestr1 = '(%d,%d,%d)' % (obs['ccdNum'],obs['x'],obs['y'])
			framestr2 = '%.3f' % (obs['mjd'])
			utstr = obs['utDate'][2:]+' '+obs['utObs'][:5]
			frameclr = ccdcolors[obs['ccdNum']-1]
			ax.set_title(utstr,size=7,color='k',weight='bold')
			t = ax.text(0.01,0.98,framestr1,
			            size=7,va='top',color=frameclr,
			            transform=ax.transAxes)
			t.set_bbox(dict(color='white',alpha=0.45,boxstyle="square,pad=0"))
			t = ax.text(0.01,0.02,framestr2,
			            size=7,color='blue',
			            transform=ax.transAxes)
			t.set_bbox(dict(color='white',alpha=0.45,boxstyle="square,pad=0"))
			if obs['flags'][2] > 0:
				t = ax.text(0.03,0.7,'%d' % obs['flags'][2],
				            size=10,ha='left',va='top',color='red',
				            transform=ax.transAxes)
			if True and photCat.name=='rmqso':
				_j = np.where(diffphot['frameIndex'][objId] ==
				              obs['frameIndex'])[0]
				if len(_j)>0:
					matched[objId,_j] = True
					flg = diffphot['FLAG'][objId,_j]
					if flg > 0:
						flgstr = [ s for bit,s in enumerate(bitstr)
						               if (flg & (1<<bit)) > 0 ]
						t = ax.text(0.97,0.8,'\n'.join(flgstr),
						            size=10,ha='right',va='top',color='red',
						            transform=ax.transAxes)
				else:
					errlog.write('no diff phot for %d %.4f %.4f\n' % 
					             (objId,obs['mjd'],obs['mjd_mid']))
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)
			pnum += 1
		if True and photCat.name=='rmqso':
			jj = np.where(~matched[objId])[0]
			if len(jj)>0:
				errlog.write('unmatched for %d:\n'%objId)
				for j in jj:
					errlog.write('    %.5f  %d\n'%
					     (diffphot['MJD'][objId,j],diffphot['FLAG'][objId,j]))
		try:
			pdf.savefig()
			plt.close()
		except:
			pass
		pdf.close()
	plt.ion()
	if True and photCat.name=='rmqso':
		errlog.close()

if __name__=='__main__':
	import argparse
	parser = argparse.ArgumentParser()
	parser = bokpl.init_file_args(parser)
	parser.add_argument('--refcatalog',type=str,default='sdssrm',
	                help='reference catalog ([sdssrm]|sdss|cfht)')
	parser.add_argument('--catalog',type=str,
	                help='object catalog (filename)')
	parser.add_argument('--metadata',action='store_true',
	                help='construct observations meta data table')
	parser.add_argument('--datasum',action='store_true',
	                help='output summary of available data')
	parser.add_argument('--checkproc',action='store_true',
	                help='check processing status of individual files')
	parser.add_argument('--calcsky',type=str,
	                help='calculate sky backgrounds')
	parser.add_argument('--checkramp',action='store_true',
	                help='check bias ramp')
	parser.add_argument('--checkgains',action='store_true',
	                help='generate gain correction diagnostic plots')
	parser.add_argument('--cutouts',action='store_true',
	                help='make image cutouts from object catalog')
	parser.add_argument('--thumbnails',action='store_true',
	                help='make image thumbnails from object catalog')
	parser.add_argument('--nbin',type=int,
	                help='factor by which to bin the thumbnail images')
	parser.add_argument('--old',action='store_true',
	                help='use old (09/2014) processed images')
	parser.add_argument('--debug',action='store_true',
	                help='extra debugging info')
	parser.add_argument('--rawgains',action='store_true',
	                help='raw gain correction values')
	parser.add_argument('--objid',type=int,
	                help='object number')
	parser.add_argument('--outfile',type=str,
	                help='output file')
	args = parser.parse_args()
	args = bokrmpipe.set_rm_defaults(args)
	dataMap = bokpl.init_data_map(args)
	photCat = bokrmphot.load_target_catalog(args.refcatalog,None,
	                                        args.catalog)
	photCat.load_ref_catalog()
	if args.datasum:
		dump_data_summary(dataMap)
	if args.checkproc:
		check_processed_data(dataMap)
	if args.calcsky:
		calc_sky_backgrounds(dataMap,args.calcsky)
	if args.metadata:
		rmobs_meta_data(dataMap)
	if args.checkramp:
		check_bias_ramp(dataMap)
	if args.checkgains:
		all_gain_plots(diagdir=dataMap.getDiagDir(),obsDb=dataMap.obsDb,
		               utDates=dataMap.getUtDates(),debug=args.debug,
		               raw=args.rawgains,pdfFile=args.outfile)
	if args.cutouts:
		photCat.load_bok_phot()
		image_cutouts(dataMap,photCat,band=args.band,
		              objid=args.objid,old=args.old)
	if args.thumbnails:
		if args.band is None:
			raise ValueError("must specify filter (g or i)")
		photCat.load_bok_phot()
		image_thumbnails(dataMap,photCat,band=args.band,
		                 objid=args.objid,nbin=args.nbin,old=args.old)

