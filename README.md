# bokrm
Bok 90Prime data for the SDSS-RM project

See additional documentation on the [wiki](https://github.com/imcgreer/idmrm/wiki).

### Examples of using Makefiles to run processing steps

First build a log of observations:

`make -f Makefile.2014 obsdb`

Process all images from 2014 through basic detrending (bias removal, flat correction, sky subtraction):

`make -f Makefile.2014`

Only the g-band data:

`make -f Makefile.2014 BANDS=g`

A specific night:

`make -f Makefile.2014 UTDATE=20140425`

Assuming all necessary calibration files exist, reprocess the science images for one night:

`make -f Makefile.2014 procall UTDATE=20140425 XARGS="-R"`

Redo individual processing steps:

`make -f Makefile.2014 steps STEPS=oscan,proc1 UTDATE=20140425 XARGS="-R -t object"`

Pipeline failures? Single-process debug mode:

`make -f Makefile.2014 procall UTDATE=20140425 NPROC=1 XARGS="-R --debug"`

Redo just one file:

`make -f Makefile.2014 procall XARGS="-R -f bokrm.20140425.0121" NPROC=1`

Post-processing steps: astrometric solutions, sextractor catalogs, and PSF models:

`make -f Makefile.2014 wcs catalogs`

Generate aperture photometry catalogs and lightcurves for SDSS reference stars,
calculate zeropoints from the aperture photometry:

`make aperphot_sdss zeropoints_sdss lightcurves_sdss`

Catalogs and lightcurves for the RM targets:

`make aperphot_rm lightcurves_rm`

Generate the meta-data table (collates the zeropoints and derives other metrics for each image):

`make metadata`

Generate PNG images for inspection:

`make images`

Check the processing status of each science image (produces HTML table):

`make checkproc`
