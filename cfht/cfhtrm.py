#!/usr/bin/env python

import os
import glob
import numpy as np
from astropy.table import Table

cfhtdir = os.environ.get('CFHTRMDATA',
                         os.path.join(os.environ['HOME'],'data','public',
                                      'CFHT','archive','SDSSRM'))

def load_obs_table():
	return Table.read(os.path.join(cfhtdir,'result_g5s2i0ftxwta9jbn.csv'))

