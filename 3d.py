# -*- coding: utf-8 -*-
# ipython -wthread

import bombo as bombo

bombo.PlotOnMap3D(track_lat=None,
                  track_lon=None,
                  tile_selection='auto',
                  margin=200,
                  elevation_scale=1.0,
                  mapping='coords',
                  #mapping='meters',
                  use_osm_texture=True,
                  showplot=True,
                  verbose=False)