# -*- coding: utf-8 -*-
# ipython -wthread

from mayavi import mlab
from PIL import Image
#import vtk
from tvtk.api import tvtk
#from tvtk.common import configure_input
from traits.api import HasTraits, Instance#, on_trait_change
from traitsui.api import View, Item
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor

import bombo as bombo

terrain, track, warnings = bombo.Generate3DMap(track_lat=[],
                                               track_lon=[],
                                               tile_selection='auto',
                                               margin=200,
                                               elevation_scale=1.0,
                                               mapping='coords',
                                               #mapping='meters',
                                               use_osm_texture=True,
                                               texture_type='osm',
                                               texture_zoom=13,
                                               texture_invert=True,
                                               verbose=False)

map_elements = bombo.Plot3DMap(terrain, track,
                               use_osm_texture=True,
                               animated=False)

"""
fig = mlab.figure(figure='3D Map', size=(500, 500))

# Plot the elevation mesh
elevation_mesh = mlab.mesh(terrain['x'],
                           terrain['y'],
                           terrain['z'],
                           figure=fig)

# Read and apply texture if needed
bmp = tvtk.PNGReader(file_name=bombo.TEXTURE_FILE)
texture = tvtk.Texture(input_connection=bmp.output_port, interpolate=1)
elevation_mesh.actor.actor.mapper.scalar_visibility = False
elevation_mesh.actor.enable_texture = True
elevation_mesh.actor.tcoord_generator_mode = 'plane'
elevation_mesh.actor.actor.texture = texture

# Set camera position
mlab.view(azimuth=-90.0,
          elevation=60.0,
          # distance=1.0,
          distance='auto',
          # focalpoint=(1000.0, 1000.0, 1000.0),
          focalpoint='auto',
          roll=0.0,
          figure=fig)

# Show the 3D map
mlab.show()
"""
