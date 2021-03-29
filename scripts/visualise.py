import pandas as pd
import geopandas as gpd
from six.moves import reduce
import numpy as np

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import shapely

from matplotlib import cm
from matplotlib.colors import to_hex
from mpl_toolkits.axes_grid1 import make_axes_locatable

from regions import NetworkReducer

from sklearn import preprocessing

def plot_cluster_histogram(cluster_buses, network, feature, save_to_path=None):
    data           = feature.get_data(network)
    buses_data     = data.reindex(cluster_buses)
    upper_quartile = round(buses_data.quantile(q=0.75).values[0], 2)

    fig, ax = plt.subplots()
    ax.set_ylabel('Frequency')
    ax.set_title('$\\tilde{x}_{0.75} = $' + '${}$'.format(upper_quartile))
    ax.hist(buses_data)
    fig.tight_layout()

    if save_to_path is None:
        plt.show()
    else:
        plt.savefig(save_to_path, bbox_inches='tight')


def plot_clustered_buses(clustered_busmap,
                       network,
                       feature,
                       jitter=0.,
                       even_bus_sizes=False,
                       bus_size_range=(0.001, 0.01),
                       cmap='Spectral',
                       color_geomap={'ocean': 'lightblue', 'land': 'lightgrey'},
                       label_fontsize=12,
                       save_to_path=None):    
    # Use colormap to assign a color to each bus depending on its cluster.
    clusters       = clustered_busmap.unique()
    color_map      = cm.get_cmap(cmap, len(clusters))
    colors         = color_map(np.linspace(0, 1, len(clusters)))
    cluster_colors = pd.Series(index=clusters, data=list(colors))
    cluster_colors = cluster_colors.apply(lambda x: to_hex(x))

    data          = feature.get_data(network, add_missing_rows=False, apply_std_scaler=False)
    missing_buses = set(network.buses.index) - set(data.index)
    
    bus_sizes = pd.Series([])
    if even_bus_sizes:
        bus_sizes = pd.Series(index=data.index, data=bus_size_range[1])
        bus_sizes = bus_sizes.append(pd.Series(index=missing_buses, data=bus_size_range[0]))
    else:        
        data = data.apply(np.log)
        scaled_values = preprocessing.MinMaxScaler(feature_range=bus_size_range).fit_transform(data.values)
        bus_sizes = pd.Series(index=data.index, data=scaled_values.flatten())
        bus_sizes = bus_sizes.append(pd.Series(index=missing_buses, data=bus_size_range[0] / 10))

    bus_to_color = clustered_busmap.apply(lambda x: cluster_colors[x])

    _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(15,15))
    network.plot(ax=ax, 
                 bus_sizes=bus_sizes,
                 bus_colors=bus_to_color,
                 line_widths=1.2,
                 link_widths=1.2,
                 color_geomap=color_geomap,
                 jitter=jitter)
    
    if save_to_path is None:
        plt.show()
    else:
        plt.savefig(save_to_path, bbox_inches='tight')

def plot_cluster_context(busmap,
                       clustered_busmap,
                       network,
                       feature,
                       onshore_path,
                       number_clusters=False,
                       cluster_border_width=1.3,
                       cmap='Blues',
                       plot_values=False,
                       legend=True,
                       title="",
                       label_fontsize=12,
                       save_to_path=None):
    """
    Show the original Voronoi-regions and their feature values in a choropleth map.
    Show the borders of the determined Clusters and plot their indices.

    Best suited for a more detailed look at e.g. a single country.
    
    TODO: add offshore!
    """    
    plt.rcParams.update({'axes.labelsize': label_fontsize})
    
    busmap_s  = [busmap, clustered_busmap]
    original  = gpd.read_file(onshore_path).set_index('name')
    clustered = cluster_regions(busmap_s, path=onshore_path)

    data = feature.get_data(network, apply_std_scaler=False)
    data = data.rename(columns={0: 'feature'})

    original = original.join(data)
    original['geometry'] = original['geometry'].buffer(0.05)

    fig, ax = plt.subplots(figsize=(15,15))
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.1)
        
    ax.set_title(title)
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')

    if number_clusters:
        for i in range(len(clustered)):
            _plot_cluster_numbers(clustered, ax)

    if plot_values:
        for i in range(len(original)):
            x, y = original.centroid[i].coords.xy
            s    = round(original.feature[i], 2)
            ax.text(x[0], y[0], s, fontsize=6)

    original.plot(column='feature',
                  linewidth=0.8,
                  edgecolor='grey',
                  ax=ax,
                  cmap=cmap,
                  legend=legend,
                  legend_kwds={'label': feature.value}, cax=cax)
    clustered.plot(linewidth=cluster_border_width, edgecolor='black', ax=ax, facecolor='none', cax=cax)

    if save_to_path is None:
        plt.show()
    else:
        plt.savefig(save_to_path, bbox_inches='tight')

def plot_original_regions(network,
         feature,
         onshore_path,
         offshore_path,
         only_onshore=False,
         cmap='Blues',
         figsize=(15,15),
         label_fontsize=12,
         legend=True,
         save_to_path=None):
    
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel('longitude', fontsize=label_fontsize)
    ax.set_ylabel('latitude', fontsize=label_fontsize)
    
    data = feature.get_data(network, apply_std_scaler=False)
    data = data.rename(columns={0: 'feature'})

    vmin = np.max([0, data['feature'].min() - .1])
    vmax = np.min([1, data['feature'].max() + .1])
    
    scalar_map = cm.ScalarMappable(norm=plt.Normalize(vmin=vmin, vmax=vmax), cmap=cmap)
    cbar = fig.colorbar(scalar_map, ax=ax, pad=0.01)
    cbar.set_label(feature.value, fontsize=label_fontsize, labelpad=10)
    
    onshore  = gpd.read_file(onshore_path).set_index('name')
    onshore  = onshore.join(data)
    if only_onshore:
        onshore.plot(column='feature',ax=ax, cmap=cmap)    
    else:
        offshore = gpd.read_file(offshore_path).set_index('name')    
        offshore = offshore.join(data)    
        onshore.plot(column='feature',ax=ax, cmap=cmap, edgecolor='black', linewidth=0.5)    
        offshore.plot(column='feature',ax=ax, cmap=cmap, alpha=0.7)

    if save_to_path is None:
        plt.show()
    else:
        plt.savefig(save_to_path, bbox_inches='tight')
        
        
def plot_cluster_regions(busmap,
                  clustered_busmap,
                  clustered_nw,
                  feature,
                  onshore_path,
                  # offshore_path,
                  cmap='Blues',
                  number_clusters=False,
                  legend=True,
                  color=None,
                  figsize=(15,15),
                  plot_values=False,
                  # label_fontsize=12,
                  buffer=0.005,
                  save_to_path=None):
    # plt.rcParams.update({'axes.labelsize': label_fontsize})
    
    _, ax   = plt.subplots(figsize=figsize)
    divider = make_axes_locatable(ax)
    cax     = divider.append_axes("right", size="5%", pad=0.1)
    
    ax.set_xlabel('longitude')
    ax.set_ylabel('latitude')
    
    busmap_s  = [busmap, clustered_busmap]
    regions_c = cluster_regions(busmap_s, path=onshore_path)
    # offshore  = cluster_regions(busmap_s, path=offshore_path)

    data_c = feature.get_data(clustered_nw, apply_std_scaler=False)
    data_c = data_c.rename(columns={0: 'feature'})

    regions_c = regions_c.join(data_c)
    regions_c['geometry'] = regions_c['geometry'].buffer(buffer)
    
    # offshore = offshore.join(data_c)
    # offshore['geometry'] = offshore['geometry'].buffer(0.05)

    vmin = data_c['feature'].min()
    vmax = data_c['feature'].max()

    regions_c.plot(column='feature',
                   linewidth=1.2,
                   edgecolor= 'black',
                   norm=plt.Normalize(vmin=vmin, vmax=vmax),
                   ax=ax,
                   cax=cax,
                   cmap=cmap,
                   color=color,
                   legend=legend,
                   legend_kwds={'label': feature.value})
    
    # offshore.plot(column='feature',
    #               linewidth=1.2,
    #               edgecolor= 'black',
    #               norm=plt.Normalize(vmin=vmin, vmax=vmax),
    #               ax=ax,
    #               cax=cax,
    #               cmap=cmap,
    #               color=color,
    #               legend=legend,
    #               legend_kwds={'label': feature.value})
    if plot_values:
        for i in range(len(regions_c)):
            x, y = regions_c.centroid[i].coords.xy
            s    = round(regions_c.feature[i], 2)
            ax.text(x[0], y[0], s, fontsize=10, bbox=dict(facecolor='lightyellow', alpha=0.8))

    if number_clusters:
        _plot_cluster_numbers(regions_c, ax)

    if save_to_path is None:
        plt.show()
    else:
        plt.savefig(save_to_path, bbox_inches='tight')

def plot_components(busmap,
                    clustered_busmap,
                    onshore_path,
                    network=None,
                    cmap='Spectral',
                    figsize=(15,15),
                    number_components=True,
                    save_to_path=None):
    ax = None

    if network is not None:
        _, ax = plt.subplots(subplot_kw={"projection": ccrs.PlateCarree()}, figsize=(15,15))
        network.plot(ax=ax, bus_sizes=0.001, bus_colors='gray', line_widths=1.2, link_widths=1.2)
    else:
        _, ax = plt.subplots(figsize=figsize)
    
    busmap_s  = [busmap, clustered_busmap]
    regions_c = cluster_regions(busmap_s, path=onshore_path)

    regions_c.plot(linewidth=1.2,
                   ax=ax,
                   cmap=cmap)

    if number_components:
        _plot_cluster_numbers(regions_c, ax)

    if save_to_path is None:
        plt.show()
    else:
        plt.savefig(save_to_path, bbox_inches='tight')


def highlight_regions(buses,
                     busmap,
                     clustered_busmap,
                     network,
                     onshore_path):
    """

    """
    other_buses = set(network.buses.index) - set(buses)
    bm_1 = pd.Series(index=buses, data=0)
    bm_2 = pd.Series(index=other_buses, data=1)
    bm = bm_1.append(bm_2)

    onshore  = gpd.read_file(onshore_path).set_index('name')

    busmap_s  = [busmap, clustered_busmap]
    busmap    = reduce(lambda x, y: x.map(y), busmap_s[1:], busmap_s[0])

    # join this geoseries with cluster values, then plot it
    onshore = gpd.GeoDataFrame(geometry=onshore.geometry).join(busmap).rename(columns={'0' : 'cluster'})

    fig, ax = plt.subplots(figsize=(15,15))
    onshore.plot(column='cluster', linewidth=0.8, edgecolor='grey', ax=ax, cmap='tab20b')
    plt.show()  

def _plot_cluster_numbers(regions_c, ax, alpha=0.7):
    for i in range(len(regions_c)):
        x, y = regions_c.centroid[i].coords.xy
        s    = regions_c.index[i]
        ax.text(x[0], y[0], s, fontsize=10, bbox=dict(facecolor='lightyellow', alpha=alpha))
        
def cluster_regions(busmap_s, path='../resources/regions_onshore.geojson'):
    busmap = reduce(lambda x, y: x.map(y), busmap_s[1:], busmap_s[0])

    regions = gpd.read_file(path).set_index('name')
    geom_c  = regions.geometry.groupby(busmap).apply(shapely.ops.cascaded_union)
    regions_c = gpd.GeoDataFrame(dict(geometry=geom_c))
    regions_c.index.name = 'name'
    return regions_c