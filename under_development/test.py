# -*- coding: utf-8 -*-
"""
Created on Fri May 12 15:49:42 2017

@author: salatis
"""
import gpxpy
import bombo as bombo


a, longest_traseg, Ntracks, Nsegments, infos = bombo.LoadGPX('tracks/AVP/avp01.gpx')
b, longest_traseg, Ntracks, Nsegments, infos = bombo.LoadGPX('tracks/2017-04-09 1552__20170409_1552 MTB tra Epen e il confine belga.gpx')

segment = b.tracks[0].segments[0]
print segment.has_elevations()