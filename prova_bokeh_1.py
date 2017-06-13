#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 19:27:42 2017

@author: ste
"""

"""
https://stackoverflow.com/questions/35983029/bokeh-synchronizing-hover-tooltips-in-linked-plots
http://bokeh.pydata.org/en/0.10.0/docs/user_guide/interaction.html
http://bokeh.pydata.org/en/latest/docs/user_guide/tools.html
"""

from bokeh.io import gridplot
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, Circle, HoverTool, CustomJS, Text
import numpy as np
(x, y, z) = np.arange(0, 100, 10), 100-np.arange(0, 100, 10), np.arange(0, 100, 10)/5

output_file("hover_callback.html")

p = figure(width=300, height=300, title='Hover over points', x_axis_label='x', y_axis_label='y')
p.scatter(x, y)
p2 = figure(width=300, height=300, title='Hover over points', x_axis_label='x', y_axis_label='z', x_range=p.x_range)
p2.scatter(x, z)

source = ColumnDataSource({'x': x, 'y': y, 'z': z, 'txt': ['x='+str(x[i])+', y='+str(y[i]) for i in range(len(x))], 'txt2': ['x='+str(x[i])+', z='+str(z[i]) for i in range(len(x))]})

invisible_circle = Circle(x='x', y='y', fill_color='gray', fill_alpha=0.0, line_color=None, size=20) # size determines how big the hover area will be
invisible_circle2 = Circle(x='x', y='z', fill_color='gray', fill_alpha=0.0, line_color=None, size=20)

invisible_text = Text(x='x', y='y', text='txt', text_color='black', text_alpha=0.0)
visible_text = Text(x='x', y='y', text='txt', text_color='black', text_alpha=0.5)

invisible_text2 = Text(x='x', y='z', text='txt2', text_color='black', text_alpha=0.0)
visible_text2 = Text(x='x', y='z', text='txt2', text_color='black', text_alpha=0.5)

cr = p.add_glyph(source, invisible_circle, selection_glyph=invisible_circle, nonselection_glyph=invisible_circle)
crt = p.add_glyph(source, invisible_text, selection_glyph=visible_text, nonselection_glyph=invisible_text)
cr2 = p2.add_glyph(source, invisible_circle2, selection_glyph=invisible_circle2, nonselection_glyph=invisible_circle2)
cr2t = p2.add_glyph(source, invisible_text2, selection_glyph=visible_text2, nonselection_glyph=invisible_text2)

code = "source.set('selected', cb_data['index']);"
callback = CustomJS(args={'source': source}, code=code)
p.add_tools(HoverTool(tooltips=None, callback=callback, renderers=[cr, crt]))
p2.add_tools(HoverTool(tooltips=None, callback=callback, renderers=[cr2, cr2t]))
layout = gridplot([[p, p2]])
show(layout)