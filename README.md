# bokrm
Bok 90Prime data for the SDSS-RM project

### Examples of using Makefiles to run processing steps

Process all images from 2014 through basic detrending (bias removal, flat correction, sky subtraction):

`make -f Makefile.2014`

Only the g-band data:

`make -f Makefile.2014 BANDS=g`

A specific night:

`make -f Makefile.2014 UTDATE=20140425`

Assuming all necessary calibration files exist, reprocess the science images for one night:

`make -f Makefile.2014 procall UTDATE=20140425 XARGS="-R"`

Redo individual processing steps:

`make -f Makefile.2014 procall XARGS="-R -f bokrm.20140425.0121" NPROC=1`

Pipeline failures? Single-process debug mode:

`make -f Makefile.2014 procall UTDATE=20140425 NPROC=1 XARGS="-R --debug"`

Redo just one file:

`make -f Makefile.2014 steps STEPS=oscan,proc1 UTDATE=20140425 XARGS="-R -t object"`

Post-processing steps: astrometric solutions, sextractor and aperture photometry catalogs:

`make -f Makefile.2014 wcs catalogs aperphot_sdss aperphot_rm`

Determine SDSS zeropoints and generate lightcurves for SDSS reference stars:

`make -f Makefile.2014 zeropoints_sdss lightcurves_sdss lightcurves_rm`

Generate the meta-data table (zeropoints and other metrics for each image)

`make -f Makefile.2014 metadata`

Generate PNG images for inspection:

`make -f Makefile.2014 images`

Check the processing status of each science image (produces HTML table):

`make -f Makefile.2014 checkproc`
