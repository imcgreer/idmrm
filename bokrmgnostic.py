#!/usr/bin/env python

import os,sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from scipy.interpolate import LSQUnivariateSpline
from astropy.io import fits
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord,match_coordinates_sky
from astropy import units as u
from astropy.table import Table,join
from astropy.stats import sigma_clip

from bokpipe import bokphot,bokpl,bokgnostic
from bokpipe.bokproc import ampOrder,BokImStat
import bokrmpipe
import bokrmphot

def plot_gain_vals(g,raw=False,splinepars=None):
	plt.figure(figsize=(12,8))
	plt.subplots_adjust(0.04,0.02,0.97,0.99,0.28,0.05)
	if g['gains'].shape[2] == 16:
		g['gains'] = g['gains'].swapaxes(1,2)
		g['gainCor'] = g['gainCor'].swapaxes(1,2)
	if raw:
		gains = np.dstack([g['rawAmpGain'],np.repeat(g['rawCcdGain'],4,axis=1)])
		ampgaincor = g['ampTrend']
		ccdgaincor = g['ccdTrend']
	else:
		gains = np.ma.array(g['gains'],mask=g['gains']==0)
		ampgaincor = g['gainCor'][:,:,0]
		ccdgaincor = g['gainCor'][:,::4,1]
	axs = []
	for ccd in range(4):
		amp = 4*ccd
		ax = plt.subplot(5,4,ccd+1)
		ax.plot(gains[:,amp,1],'rs',ms=2.5,mfc='none',mec='r',mew=1.1)
		ax.plot(ccdgaincor[:,ccd],c='orange',ls='-',lw=1.4)
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
		ax.plot(gains[:,amp,0],'bs',ms=2.5,mfc='none',mec='b',mew=1.1)
		if splinepars is not None:
			nimg = gains.shape[0]
			xx = np.arange(nimg)
			knots = np.linspace(0,gains.shape[0],splinepars['nknots'])[1:-1]
			ii = np.where(~gains[:,amp,0].mask)[0]
			spfit = LSQUnivariateSpline(xx[ii],gains[ii,amp,0].filled(),
			                            knots,bbox=[0,nimg],
			                            k=splinepars['order'])
			ax.plot(xx,spfit(xx),c='r')
		ax.text(0.05,0.99,'IM%d'%ampOrder[amp],
		        size=8,va='top',transform=ax.transAxes)
		medgain = np.median(ampgaincor[:,amp])
		ax.text(0.25,0.99,'%.3f'%medgain,color='blue',
		        size=8,va='top',transform=ax.transAxes)
		ax.plot(ampgaincor[:,amp],c='purple',ls='-',lw=1.4)
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

def all_gain_vals(diagdir,obsDb=None):
	from glob import glob
	if obsDb is None:
		obsDb = Table.read('config/sdssrm-bok.fits')
	obsDb = obsDb.group_by(['utDate','filter'])
	skys,gains,gainCor,utds,filts,files = [],[],[],[],[],[]
	gfiles = sorted(glob(os.path.join(diagdir,'gainbal*.npz')))
	for gfile in gfiles:
		fn = os.path.basename(gfile).rstrip('.npz')
		_,utd,filt = fn.split('_')
		k = np.where((obsDb.groups.keys['utDate']==utd) &
		             (obsDb.groups.keys['filter']==filt))[0]
		g = np.load(gfile)
		n = g['skys'].shape[0]
		utds.extend([utd]*n)
		filts.extend([filt]*n)
		gn = np.ma.array(g['gains'],mask=g['gains']==0)
		gn = sigma_clip(gn,iters=2,sigma=2.0,axis=0)
		gains.append(gn)
		skys.append(np.ma.array(g['skys'],mask=gn.mask[:,0]))
		gainCor.append(np.tile(g['gainCor'],(n,1,1)))
#		jj = np.where(obsDb.groups[k]['imType']=='object')[0]
#		if len(jj)!=g['skys'].shape[0]:
#			import pdb; pdb.set_trace()
#		assert len(jj)==g['skys'].shape[0]
#		files.append(obsDb.groups[k]['fileName'][jj])
	return Table(dict(skys=np.ma.vstack(skys),gains=np.ma.vstack(gains),
	                  gainCor=np.vstack(gainCor),utDate=np.array(utds),
	                  filt=np.array(filts)),masked=True)

def all_gain_plots(gainDat=None,diagdir=None,pdfFile='bok_gain_vals.pdf'):
#	from glob import glob
	from matplotlib.backends.backend_pdf import PdfPages
	plt.ioff()
	if gainDat is None:
		gainDat = all_gain_vals(diagdir)
	gainDat = gainDat.group_by(['utDate','filt'])
	with PdfPages(pdfFile) as pdf:
		for kdat,gdat in zip(gainDat.groups.keys,gainDat.groups):
			plot_gain_vals(gdat)
			plt.title('%s-%s'%(kdat['utDate'],kdat['filt']))
			pdf.savefig()
			plt.close()
	plt.ion()

def srcor(ra1,dec1,ra2,dec2,sep):
	c1 = SkyCoord(ra1,dec1,unit=(u.degree,u.degree))
	c2 = SkyCoord(ra2,dec2,unit=(u.degree,u.degree))
	idx,d2d,d3c = match_coordinates_sky(c1,c2)
	ii = np.where(d2d.arcsec < sep)[0]
	return ii,idx[ii],d2d.arcsec[ii]

def calc_sky_backgrounds(dataMap,outputFile):
	extns = ['IM4']
	imstat = BokImStat(extensions=extns,quickprocess=True,
	                   stats_region='amp_central_quadrant')
	skyvals = []
	for utd in dataMap.iterUtDates():
		files,ii = dataMap.getFiles(imType='object',with_frames=True)
		if files is not None:
			rawfiles = [ dataMap('raw')(f) for f in files ]
			imstat.process_files(rawfiles)
			sky = imstat.meanVals.squeeze()
			print '%8s %4d %10.1f %10.1f %10.1f' % \
			      (utd,len(files),sky.mean(),sky.min(),sky.max())
			skyvals.extend([ (utd,f.split('/')[1],s) 
			                    for f,s in zip(files,sky) ])
		imstat.reset()
	tab = Table(rows=skyvals,names=('utDate','fileName','skyMean'))
	tab.write(outputFile)

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
		w = WCS(hdr)
		foot = w.calc_footprint()
		ras = sorted(foot[:,0])
		decs = sorted(foot[:,1])
		ii = np.where((refCat['ra']>ras[1])&(refCat['ra']<ras[2]) &
		              (refCat['dec']>decs[1])&(refCat['dec']<decs[2]) &
		              (refCat[band]<mlim))[0]
		m1,m2,sep = srcor(ccdCat['ALPHA_J2000'],ccdCat['DELTA_J2000'],
		                  refCat['ra'][ii],refCat['dec'][ii],5.0)
		rv.append(dict(N=len(ii),nMatch=len(ii),
		               ra=ccdCat['ALPHA_J2000'][m1],
		               dec=ccdCat['DELTA_J2000'][m1],
		               raRef=refCat['ra'][ii[m2]],
		               decRef=refCat['dec'][ii[m2]],
		               sep=sep))
	return rv

def rmobs_meta_data(dataMap):
	bokgnostic.obs_meta_data(dataMap,outFile='bokrmMetaData.fits')

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
	zeropoints = fits.getdata('zeropoints_g.fits')
	tabf = open(os.path.join('proc_diag.html'),'w')
	tabf.write(bokgnostic.html_diag_head)
	rowstr = ''
	files_and_frames = dataMap.getFiles(with_frames=True)
	for f,i in zip(*files_and_frames):
		frameId = dataMap.obsDb['frameIndex'][i]
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
			                                  zeropoints['frameId'])[0][0]
		except IndexError:
			zpi = None
		for ccdi in range(4):
			if zpi is not None:
				zp = zeropoints['aperZp'][zpi,ccdi]
				if zp > 25.90:
					status = 'nominal'
				elif zp > 25.70:
					status = 'warning'
				elif zp > 25.40:
					status = 'bad'
				else:
					status = 'weird'
			else:
				zp = 0.0
				status = 'missing'
			rowstr += bokgnostic.html_table_entry('%.2f'%zp,status)
		catf = dataMap('cat')(f)
		try:
			m = check_img_astrom(procf,sdss,catFile=catf)
			for c in m:
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

def image_thumbnails(dataMap,catFile,band='g',old=False):
	from astropy.nddata import Cutout2D
	from astropy.visualization import ZScaleInterval
	from matplotlib.backends.backend_pdf import PdfPages
	objs = Table.read(catFile)
	objs['frameIndex'] = objs['frameId']
	objs = join(objs,dataMap.obsDb,'frameIndex')
	objs = objs.group_by('objId')
	if old:
		datadir = '/media/ian/stereo/data/projects/SDSS-RM/rmreduce/'
	else:
		datadir = '/d2/data/projects/SDSS-RM/RMpipe/bokpipe_v0.1'
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
	if not os.path.exists(outdir):
		os.makedirs(outdir)
	ccdcolors = ['darkblue','darkgreen','darkred','darkmagenta']
	plt.ioff()
	for obj in objs.groups:
		pdffile = outdir+'bokrm%03d_%s.pdf'%(obj['objId'][0],band)
		if os.path.exists(pdffile) or len(obj)==0:
			continue
		pdf = PdfPages(pdffile)
		pnum = -1
		for i,obs in enumerate(obj):
			sys.stdout.write('\rRM%03d %4d/%4d' % 
			                 (obs['objId'],(i+1),len(obj)))
			sys.stdout.flush()
			ccdNum = obs['ccdNum']
			if old:
				fn = os.path.join(datadir,obs['utDir'],'ccdproc3',
				                  obs['fileName']+'_ccd%d.fits'%ccdNum)
				im = fits.getdata(fn)
			else:
				fn = os.path.join(datadir,obs['utDir'],
				                  obs['fileName']+'.fits')
				im = fits.getdata(fn,'CCD%d'%ccdNum)
			cutobj = Cutout2D(im,(obs['x'],obs['y']),size,mode='partial',
			                  fill_value=0)
			z1,z2 = zscl.get_limits(im)
			cut = cutobj.data
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
			framestr = '(%d,%d) [%d,%d]' % (obs['x'],obs['y'],
			                                obs['ccdNum'],obs['frameId'])
			utstr = obs['utDate'][2:]+' '+obs['utObs'][:5]
			frameclr = ccdcolors[obs['ccdNum']-1]
			ax.set_title(utstr,size=7,color='k',weight='bold')
			t = ax.text(0.01,0.98,framestr,
			            size=7,va='top',color=frameclr,
			            transform=ax.transAxes)
			t.set_bbox(dict(color='white',alpha=0.45,boxstyle="square,pad=0"))
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)
			pnum += 1
		if pnum != nplot+1:
			pdf.savefig()
			plt.close()
		pdf.close()
	plt.ion()
	print

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
	parser.add_argument('--thumbnails',action='store_true',
	                help='make image thumbnails from object catalog')
	parser.add_argument('--old',action='store_true',
	                help='use old (09/2014) processed images')
	args = parser.parse_args()
	args = bokrmpipe.set_rm_defaults(args)
	dataMap = bokpl.init_data_map(args)
	refCat = bokrmphot.load_catalog(args.refcatalog)
	if args.datasum:
		dump_data_summary(dataMap)
	elif args.checkproc:
		check_processed_data(dataMap)
	elif args.calcsky:
		calc_sky_backgrounds(dataMap,args.calcsky)
	elif args.metadata:
		rmobs_meta_data(dataMap)
	elif args.checkramp:
		check_bias_ramp(dataMap)
	elif args.thumbnails:
		image_thumbnails(dataMap,args.catalog,old=args.old)

