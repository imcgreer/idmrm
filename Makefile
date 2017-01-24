#!/usr/bin/make

.PHONY: initproc badpix flats proc redo

TOOLS := $(HOME)/dev/rapala/bokpipe/tools

ifdef LOGFILE
	LOGARGS := --obsdb $(LOGFILE)
endif

ifdef RAWDATA
	DATAARGS := -r $(RAWDATA)
endif

ifdef UTDATE
	UTARGS := -u $(UTDATE)
endif

ifdef NPROC
	MPARGS := -p 7
endif

ifndef VERBOSE
	VERBOSE := -vvvv
endif

ifdef BANDS
	BANDARGS := -b $(BANDS)
endif

INITARGS := $(LOGARGS) $(DATAARGS) $(UTARGS) $(BANDARGS) \
            $(MPARGS) $(VERBOSE)

all_detrend: initproc badpix proc1 makeillum flats proc2

# Overscan-subtract all images, generate 2D biases and dome flats
initproc:
	python bokrmpipe.py $(INITARGS) $(PROCARGS) \
	                    -s oscan,bias2d,flat2d \
	                    $(XARGS)

# Generate the bias ramp image (<=2015 data)
biasramp:
	python bokrmpipe.py $(INITARGS) \
	                    -s ramp \
	                    $(XARGS)

# XXX copy in a master bp mask from config dir
badpix:
	cp $(BOK90PRIMEOUTDIR)/bokpipe_v0.2/cals/BadPix* $(BOK90PRIMEOUTDIR)/bokpipe_v0.3/cals/

# First-round processing: bias/domeflat correction, combine into CCD extensions
proc1:
	python bokrmpipe.py $(INITARGS) $(PROCARGS) \
	                    -s proc1 \
	                    $(XARGS)

# Make the illumination correction image
makeillum:
	python bokrmpipe.py $(INITARGS) \
	                    -s illum 
	                    $(XARGS)

#
# Sky flat generation (processing output to temp directory)
#

#  ... apply the illumination correction to the sky flat images
flats_illumcorr:
	python bokrmpipe.py $(INITARGS) \
	                    -s proc2 --prockey TMPPRO2 \
	                    --nofringecorr --noskyflatcorr \
	                    --noskysub --noweightmap \
	                    --darkskyframes --tmpdirout 
	                    $(XARGS)

#  ... make fringe masters from sky flat images
flats_makefringe:
	python bokrmpipe.py $(INITARGS) \
	                    -s fringe \
	                    --darkskyframes --tmpdirin --tmpdirout 
	                    $(XARGS)

# ... apply fringe correction to sky flat images
flats_fringeskycorr:
	python bokrmpipe.py $(INITARGS) \
	                    -s proc2 --prockey TMPPRO3 \
	                    --noillumcorr --noskyflatcorr --noweightmap \
	                    --skymethod polynomial --skyorder 1 \
	                    --darkskyframes --tmpdirin --tmpdirout 
	                    $(XARGS)

# ... combine temp processed images to make sky flat
flats_makeskyflat:
	python bokrmpipe.py $(INITARGS) \
	                    -s skyflat \
	                    --darkskyframes --tmpdirin --tmpdirout 
	                    $(XARGS)

# all the steps to generate sky flats in one target
flats: flats_illumcorr \
       flats_makefringe flats_fringeskycorr \
       flats_makeskyflat

# Second-round processing: apply illumination, skyflat, and fringe corrections
#  and do sky subtraction
proc2:
	python bokrmpipe.py $(INITARGS) $(PROCARGS) \
	                    -s proc2 \
	                    $(XARGS)

# Perform individual processing steps as listed in STEPS
steps:
	python bokrmpipe.py $(INITARGS) $(PROCARGS) -s $(STEPS) $(XARGS)

# Assuming cals already exist, perfrom all the processing steps on science ims
procall:
	python bokrmpipe.py $(INITARGS) $(PROCARGS) \
	                    -s oscan,proc1,proc2 -t object \
	                    $(XARGS)


# Obtain astrometric solutions
wcs:
	python bokrmpipe.py $(INITARGS) -s wcs --gaia $(XARGS)


# Generate object catalogs and PSF models with sextractor+psfex
catalogs:
	python bokrmpipe.py $(INITARGS) -s cat $(XARGS)


# Generate aperture photometry catalogs for RM targets
aperphot_rm:
	python bokrmphot.py $(INITARGS) --aperphot $(XARGS)

# Generate aperture photometry catalogs for SDSS reference stars
aperphot_sdss:
	python bokrmphot.py $(INITARGS) --aperphot $(XARGS) --catalog sdss

# Generate aperture photometry catalogs for CFHT reference stars
aperphot_cfht:
	python bokrmphot.py $(INITARGS) --aperphot $(XARGS) --catalog cfht

aperphot: aperphot_sdss aperphot_cfht aperphot_rm

zeropoints_sdss:
	python bokrmphot.py $(INITARGS) --zeropoint $(XARGS) --catalog sdss

# Generate lightcurve tables for RM targets
lightcurves_rm:
	python bokrmphot.py $(INITARGS) --lightcurves $(XARGS)

# Generate lightcurve tables for SDSS reference stars
lightcurves_sdss:
	python bokrmphot.py $(INITARGS) --lightcurves $(XARGS) --catalog sdss

# Generate lightcurve tables for CFHT reference stars
lightcurves_cfht:
	python bokrmphot.py $(INITARGS) --lightcurves $(XARGS) --catalog cfht

all_sdssref: aperphot_sdss zeropoints_sdss lightcurves_sdss

metadata:
	python bokrmgnostic.py $(LOGARGS) $(UTARGS) $(BANDARGS) --metadata $(XARGS)


# Redo all processing, WCS, and catalogs from scratch
redo:
	make XARGS="-R $(XARGS)" procall wcs catalogs aperphot

#
# Diagnostic stuff
#

# Make PNG images for inspection purposes
images:
	python bokrmpipe.py $(INITARGS) --images $(XARGS)

# Tabulate the rms of the astrometric solutions
checkwcs:
	python bokrmpipe.py $(INITARGS) --wcscheck $(XARGS)

# Check status of image processing
checkproc:
	python bokrmgnostic.py $(LOGARGS) $(UTARGS) $(BANDARGS) --checkproc $(XARGS)

