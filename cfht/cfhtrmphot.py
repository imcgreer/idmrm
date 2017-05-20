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

nom_pixscl = 0.18555

cfhtrm_aperRad = np.array([0.8,1.6,2.4,3.2,4.0,7.0,10.]) / nom_pixscl

