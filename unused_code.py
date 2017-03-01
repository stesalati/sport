## Functions to plot on OpenStreetMaps maps downloaded tile by tile. Now useless as I use folium and mplleaflet.

import math
import StringIO
from PIL import Image
from mpld3 import plugins, utils
import urllib2, urllib
from mpl_toolkits.axes_grid.anchored_artists import AnchoredText
import matplotlib.pyplot as mpld3
from pylab import imshow, imread, show

def MapTilesDeg2Num(lat_deg, lon_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
  return (xtile, ytile)

def MapTilesNum2Deg(xtile, ytile, zoom):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lat_deg, lon_deg)

def GetMapImageCluster(use_proxy, proxy_data, lat_deg, lon_deg, delta_lat, delta_long, zoom, verbose):
    # Proxy setup with urllib
    # proxies = {'http': 'salatis:Alzalarosa01@userproxy.tmg.local:8080'}
    
    # Proxy setup with urllib2
    if use_proxy:
        proxy = urllib2.ProxyHandler({'http': proxy_data})
        opener = urllib2.build_opener(proxy)
        urllib2.install_opener(opener)
    
    # Local storage folder
    image_storage_path = 'map_tiles/'
    savename = r"{0}_{1}_{2}.png"
    
    # Request url
    smurl = r"http://a.tile.openstreetmap.org/{0}/{1}/{2}.png"
    
    # These are the tiles to download
    xmin, ymax = MapTilesDeg2Num(lat_deg, lon_deg, zoom)
    xmax, ymin = MapTilesDeg2Num(lat_deg + delta_lat, lon_deg + delta_long, zoom)
    if verbose:
        print "Tiles %d - %d (horizontally) and %d - %d (vertically) are needed" % (xmin, xmax, ymin, ymax)
    
    # Margin coordinates of the tiles that were downloaded (adding 1 to the max
    # tiles as apparently the coordinates returned by MapTilesNum2Deg refer to
    # the origin of the tile)
    lat_min, lon_min = MapTilesNum2Deg(xmin, ymin, zoom)
    lat_max, lon_max = MapTilesNum2Deg(xmax + 1, ymax + 1, zoom)
    lat_min_actual = np.min((lat_min, lat_max))
    lat_max_actual = np.max((lat_min, lat_max))
    lon_min_actual = np.min((lon_min, lon_max))
    lon_max_actual = np.max((lon_min, lon_max))
    tiles_edges_coords = (lat_min_actual, lat_max_actual, lon_min_actual, lon_max_actual)
    if verbose:
        print "User requested area >>>>>>>>>>>>> lat: %f - %f, lon: %f - %f" % (lat_deg, lat_deg + delta_lat, lon_deg, lon_deg + delta_long)
        print "Returned area (must be wider) >>> lat: %f - %f, lon: %f - %f" % (lat_min_actual, lat_max_actual, lon_min_actual, lon_max_actual)
    
    # Populate the desired map with tiles
    Cluster = Image.new('RGB',((xmax-xmin+1)*256-1, (ymax-ymin+1)*256-1))
    for xtile in range(xmin, xmax+1):
        for ytile in range(ymin,  ymax+1):
            try:
                # Check if the tile is already present locally
                if os.path.isfile(image_storage_path + savename.format(zoom, xtile, ytile)):
                    tile = Image.open(image_storage_path + savename.format(zoom, xtile, ytile))
                else:
                    # Download from the Internet and save it locally for future
                    # use
                    imgurl = smurl.format(zoom, xtile, ytile)
                    print("Tile not found locally, have to download it from: " + imgurl)
                    imgstr = urllib2.urlopen(imgurl).read()
                    tile = Image.open(StringIO.StringIO(imgstr))
                    with open(image_storage_path + savename.format(zoom, xtile, ytile), 'wb') as f:
                        f.write(imgstr)
                        f.close()
                # Append it to the rest of the cluster
                Cluster.paste(tile, box=((xtile-xmin)*256 ,  (ytile-ymin)*255))
            except: 
                print("Tile loading (either from local repository or the Internet failed.")
                tile = None
                
    return Cluster, tiles_edges_coords

def MapInteractivePlot(fig, s, h, dh, lat, lon, zoom_level, margin_percentage, use_proxy, proxy_data, verbose):
    fig.subplots_adjust(hspace=0.1, wspace=0.1)
    gs = gridspec.GridSpec(1, 3, width_ratios=[5, 0.001, 5])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    ax2 = plt.subplot(gs[2])
    X = np.vstack((s, h[0:-1], lon[0:-1], lat[0:-1]))
    
    # Elevation over distance
    points = ax0.plot(X[0], X[1], color = '0.5')
    points = ax0.scatter(X[0], X[1], s=4)
    ax0.set_ylabel("Elevation (m)")
    ax0.set_xlabel("Distance (m)")
    ax0.grid(True)
    ax0.set_xlim(np.min(s), np.max(s))
    ax0.set_ylim(0, np.max(h) + 100)
    
    # Total ascent/descent
    #at = AnchoredText("Total ascent. %dm\nTotal descent: %dm" % (TotalAscentDescent(dh_filtered, 1), TotalAscentDescent(dh_filtered, -1)),
    #                  prop=dict(size=12), frameon=True,
    #                  loc=2,
    #                  )
    #at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
    #ax1.add_artist(at)
    
    # Fake subplot
    points = ax1.scatter(X[0], X[2])
    ax1.xaxis.set_major_formatter(plt.NullFormatter())
    ax1.yaxis.set_major_formatter(plt.NullFormatter())
    
    # Map with route
    margin = np.max((np.max(lat) - np.min(lat), np.max(lon) - np.min(lon))) * margin_percentage
    lat_origin = np.min(lat) - margin
    lon_origin = np.min(lon) - margin
    lat_span = np.max(lat) - np.min(lat) + 2 * margin
    lon_span = np.max(lon) - np.min(lon) + 2 * margin
    a, tiles_edges_coords = GetMapImageCluster(use_proxy, proxy_data, lat_origin, lon_origin, lat_span, lon_span, zoom_level, verbose)
    fig.patch.set_facecolor('white')
    points = ax2.plot(X[2], X[3], color = '0.5')
    img = np.asarray(a)
    # Extent simply associates values, in this case longitude and latitude, to
    #	the map's corners.
    ax2.imshow(img, extent=[tiles_edges_coords[2], tiles_edges_coords[3], tiles_edges_coords[0], tiles_edges_coords[1]], zorder=0, origin="lower")
    points = ax2.scatter(X[2], X[3], s=4)
    ax2.set_xlim(lon_origin, lon_origin + lon_span)
    ax2.set_ylim(lat_origin, lat_origin + lat_span)    
    ax2.set_xlabel("Lat")
    ax2.set_ylabel("Lon")
    ax2.grid(True)
    plugins.connect(fig, plugins.LinkedBrush(points))
    mpld3.show()
    
    return fig
