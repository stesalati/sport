# -*- coding: utf-8 -*-
# ipython -wthread

import bombo as bombo

map_elements, terrain, track, warnings = bombo.PlotOnMap3D(track_lat=[],
                                                           track_lon=[],
                                                           tile_selection='auto',
                                                           margin=300,
                                                           elevation_scale=1.0,
                                                           mapping='coords',
                                                           #mapping='meters',
                                                           use_osm_texture=True,
                                                           texture_type='osm',
                                                           texture_zoom=13,
                                                           showplot=True,
                                                           animated=False,
                                                           verbose=False)
