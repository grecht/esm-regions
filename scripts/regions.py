from typing import List
import sys

from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics.pairwise import haversine_distances

from scipy import sparse
from scipy.sparse.csgraph import dijkstra

# Imports for max-p-regions
from region import max_p_regions as maxpr
from region import p_regions as pr

import pypsa
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import geopandas as gpd
import shapely

import pyomo.environ as po

from pypsa.networkclustering import _make_consense, get_clustering_from_busmap

sys.path.insert(0, '../scripts/')

from features import NetworkFeature

class NetworkReducer:
    """
    Uses a reduction methodology to aggregate buses in a network.
    """
    def __init__(self, aggr_nom_power=np.sum):
        self._aggr_nom_power = aggr_nom_power

    def apply_reduction(self,
                        network: pypsa.Network,
                        only_synchronous=True,
                        cross_border=False,
                        include_hvdc=True) -> (pypsa.Network, pd.Series):
        """
        Apply the reduction methodology, return the reduced network and the reduction busmap.
        
        TODO: comment on only_synchronous
        """
        if cross_border:
            network = self._prepare_cross_border(network)
        busmap = self.get_busmap(network, only_synchronous, cross_border, include_hvdc)
        return self._reduce_network_from_busmap(network, busmap)

    def get_busmap(self,
                   network: pypsa.Network,
                   only_synchronous=True,
                   cross_border=False,
                   include_hvdc=True) -> pd.Series:
        """
        Obtain a mapping (bus -> aggregated bus) without applying it on the network.
        """
        if cross_border:
            network = self._prepare_cross_border(network)
        return self._determine_reduction(network, only_synchronous, cross_border, include_hvdc)
        
    def _determine_reduction(self,
                             network: pypsa.Network,
                             only_synchronous=True,
                             cross_border=False,
                             include_hvdc=True) -> pd.Series:        
        raise NotImplementedError('This method should be implemented by a subclass.')
    
    def _prepare_cross_border(self, network):
        network_copy = network.copy()
        network_copy.buses['country'] = np.NAN
        return network_copy
    
    def _reduce_network_from_busmap(self,
                                   original_network: pypsa.Network,
                                   busmap: pd.Series) -> (pypsa.Network, pd.Series):
        """
        Create the reduced network from a mapping (bus -> aggregated bus).
        """
        print("Creating the reduced network from the busmap.")        
        reduced = get_clustering_from_busmap(
            original_network, busmap,
            bus_strategies=dict(country=_make_consense("Bus", "country")),
            aggregate_generators_weighted=True,
            aggregate_generators_carriers=None, # TODO: hier ursprünglich variable aggregate_carriers
            aggregate_one_ports=["Load", "StorageUnit"],
            line_length_factor=1.25,
            generator_strategies={'p_nom_max': self._aggr_nom_power}, # np.sum / np.min
            scale_link_capital_costs=False)
        return reduced.network, reduced.busmap

class MetaReducer(NetworkReducer):
    """
    A NetworkReducer which uses a chain of NetworkReducers to aggregate buses in a network.
    """
    def __init__(self, reducers: List[NetworkReducer], aggr_nom_power=np.sum):
        super().__init__(aggr_nom_power)
        if len(reducers) < 2:
            raise ValueError('MetaReducer expects a list of at least 2 reducers.')
        self._reducers = reducers

    def _determine_reduction(self,
                             network: pypsa.Network,
                             only_synchronous=True,
                             cross_border=False,
                             include_hvdc=True) -> pd.Series:
        print(f'Applying chain of {len(self._reducers)} reducers.')
        
        n_to_m_buses       = self._reducers[0].get_busmap(network, only_synchronous, cross_border, include_hvdc)
        reduced_network, _ = self._reduce_network_from_busmap(network, n_to_m_buses)

        for reducer in self._reducers[1:-1]:
            m_to_l_buses = reducer.get_busmap(reduced_network, only_synchronous, cross_border, include_hvdc)
            n_to_l_buses = self._merge_consecutive_reductions(n_to_m_buses, m_to_l_buses)

            reduced_network, _ = self._reduce_network_from_busmap(network, n_to_l_buses)
            n_to_m_buses       = n_to_l_buses

        # Determine last reduction separately to avoid creating reduced_network one too many times.
        m_to_l_buses = self._reducers[-1].get_busmap(reduced_network, only_synchronous, cross_border, include_hvdc)
        n_to_l_buses = self._merge_consecutive_reductions(n_to_m_buses, m_to_l_buses)
        return n_to_l_buses

    def _merge_consecutive_reductions(self,
                                      n_to_m_buses: pd.Series,
                                      m_to_l_buses: pd.Series) -> pd.Series:
        """
        Join reduction from original network (n -> m) with a subsequent reduction of the reduced
        network (m -> l) to obtain the reduction from the original buses (n -> l).
        """
        return n_to_m_buses.map(m_to_l_buses)

class ConcreteReducer(NetworkReducer):
    """
    Applies a clustering or regionalisation algorithm on a NetworkFeature (e.g. coordinates,
    solar availability timeseries) to reduce the size of a network.
    """
    def __init__(self, feature: NetworkFeature, aggr_nom_power=np.sum):
        super().__init__(aggr_nom_power)
        self._feature      = feature
        self._feature_data = None
        self._adj_list     = None
        self._components   = None

    def _determine_reduction(self,
                             network: pypsa.Network,
                             only_synchronous=True,
                             cross_border=False,
                             include_hvdc=True) -> pd.Series:
        """
        Obtain a mapping (bus -> aggregated bus) without applying it on the network.
        """
        print(f"Determining reduction for {network.buses.index.size} buses " \
              f"with 'cross_border'={cross_border}, 'include_hvdc'={include_hvdc}.")
        
        # Reset data to allow "reusing" reducer
        self._feature_data = None

        groups         = []
        components     = []
        self._adj_list = self._build_adjacency_list(network, include_hvdc)

        if only_synchronous:
            network.determine_network_topology()
            if cross_border:
                # Synchronous areas
                groups = network.buses.groupby(['sub_network']).groups.values()
            else:
                # Countries and synchronous areas
                groups = network.buses.groupby(['country', 'sub_network']).groups.values()        
        elif cross_border:
            # Whole network
            groups = [list(network.buses.index.values)]
        else:
            # Countries, ignore synchronous areas.
            groups = network.buses.groupby(['country']).groups.values()
        
        # Check connectivity of groups and extract components
        for group in groups:
            group_adj_list = self._get_reduced_adj_list(group)
            for component in self._connected_components(group_adj_list):
                components.append(component)

        print(f'Number of network components: {len(components)}')

        self._components = {k:v for (k,v) in enumerate(components)}
        busmap           = pd.Series([])
        next_label       = 0

        for component_id in self._components.keys():
            component_bm             = self._apply_algorithm(network, component_id)
            component_bm, next_label = self._relabel_busmap(component_bm, next_label)
            busmap                   = busmap.append(component_bm)
        return self._correct_bus_names(busmap)

    def _apply_algorithm(self, network: pypsa.Network, component_id: int) -> pd.Series:
        """
        Use a clustering or regionalisation algorithm to determine a mapping
        (bus -> aggregated bus) for the buses belonging to the respective component.
        """
        raise NotImplementedError('This method should be implemented by a subclass.')

    def _get_reduced_adj_list(self, buses):
        """
        Return an adjacency list which only contains lines between
        'buses'.
        """
        reduced_list = {}
        for bus in buses:
            reduced_list[bus] = [i for i in self._adj_list[bus] if i in buses]
        return reduced_list

    def _build_adjacency_list(self, network: pypsa.Network, include_hvdc: bool) -> dict:
        """
        Returns an adjacency list based on the network topology of the whole network.
        If HVDC lines are excluded, the returned adjacency list contains disconnected components.
        """
        def add_line(neighbors, fr, to):
            if fr not in neighbors:
                neighbors[fr] = [to]
            else:
                neighbors[fr].append(to)

        neighbors = {}
        for _, row in network.lines.iterrows():
            fr = row.bus0
            to = row.bus1
            add_line(neighbors, fr, to)
            add_line(neighbors, to, fr)

        if include_hvdc:
            for _, row in network.links.iterrows():
                fr = row.bus0
                to = row.bus1
                add_line(neighbors, fr, to)
                add_line(neighbors, to, fr)

        # Add empty list for buses without neighbors.
        for bus in set(network.buses.index) - set(neighbors.keys()):
            neighbors[bus] = []
        return neighbors

    # TODO: rename to "determine_components"
    def _connected_components(self, adjacency_list: dict) -> list:
        """
        Return list of connected components (represented as a list).
        """
        visited    = set()
        components = []
        for root in adjacency_list.keys():
            if root not in visited:
                component = self._bfs(adjacency_list, root)
                visited.update(component)
                components.append(list(component))
        return components

    def _bfs(self, adjacency_list: dict, root: str) -> set:
        queue   = [root]
        visited = set([root])
        while queue:
            node = queue.pop()
            for adjacent in adjacency_list[node]:
                if adjacent not in visited:
                    visited.add(adjacent)
                    queue.append(adjacent)
        return visited

    def _relabel_busmap(self, busmap: pd.Series, next_label: int) -> (pd.Series, int):
        local_labels  = busmap.unique()
        last_label    = next_label + local_labels.size
        global_labels = dict(zip(local_labels, range(next_label, last_label)))
        return (busmap.map(global_labels), last_label)

    def _get_feature_data(self,
                          network: pypsa.Network,
                          buses: list,
                          add_missing_rows=True) -> pd.DataFrame:
        """
        Return feature data for 'buses'.
        """
        if self._feature_data is None:
            self._feature_data = self._feature.get_data(network, add_missing_rows)
        return self._feature_data.reindex(buses).dropna()

    def _correct_bus_names(self, busmap: pd.Series) -> pd.Series:
        """
        Some reduction algorithms return the new bus indices in the wrong form,
        e.g. as floats. Convert them to PyPSA's string indices.
        """
        return busmap.astype(int).astype(str)

    def _get_country_buses(self, network: pypsa.Network, country: str) -> np.ndarray:
        return network.buses.loc[network.buses.country == country].index.values

class ZeroBusesAllocator(ConcreteReducer):
    """
    'Zero-buses' do not have renewable generation capacity due to constraints
    on the usable terrain or for other reasons. 
    This causes Agglomerative Clustering to create small 'zero-clusters' 
    containing mostly zero-buses. To prevent this, ZeroBusesAllocator
    aggregates them with a close, reachable non-zero-bus.
    """
    def __init__(self, feature: NetworkFeature):
        super().__init__(feature, aggr_nom_power=np.sum)
        
    def _get_weighted_adj_matrix(self, buses, network):
        matrix       = np.zeros((buses.size, buses.size))
        line_weights = NetworkFeature.LINE_LENGTH_DIV_CAP.get_data(network)

        for _, row in line_weights.iterrows():
            if row.bus0 not in buses or row.bus1 not in buses:
                continue
            indices = buses.get_indexer([row.bus0, row.bus1])
            x, y    = indices[0], indices[1]
            weight  = row.value
            
            matrix[x][y] = weight
            matrix[y][x] = weight
        return sparse.csr_matrix(matrix)
    
    def _apply_algorithm(self, network: pypsa.Network, component_id: int) -> pd.Series:
        buses = pd.Index(self._components[component_id]).sort_values()
        data  = self._feature.get_data(network, 
                                       apply_std_scaler=False,
                                       add_missing_rows=False).reindex(buses).dropna()
        
        weighted_adj_matrix = self._get_weighted_adj_matrix(buses, network)
        zero_buses = None
        
        if data.columns.size == 1:
            non_zero_buses = set(data.index)
            zero_buses     = buses.reindex(set(buses) - non_zero_buses)[0]            
        elif data.columns.size == 2:            
            # 2 kinds of zero-buses: 2 zero-columns or 1 zero-column.            
            contains_zero     = data.eq(0).any(axis=1)
            single_zero_buses = set(contains_zero[contains_zero].index)
            non_zero_buses    = set(contains_zero[~contains_zero].index)
            double_zero_buses = set(buses) - non_zero_buses - single_zero_buses            
            zero_buses        = buses.reindex(single_zero_buses.union(double_zero_buses))[0]
       
        zero_indices = buses.get_indexer(zero_buses)
        distances    = dijkstra(weighted_adj_matrix, directed=False, indices=zero_indices)
        distances[:, zero_indices] = np.inf
            
        assign_to = distances.argmin(axis=1)
        busmap    = buses.to_series()
        busmap.loc[zero_buses] = buses.values[assign_to]
            
        return busmap 

class FixedKReducer(ConcreteReducer):
    """
    A Reducer using an algorithm which expects the
    number of clusters k as an argument.
    """
    def __init__(self, feature: NetworkFeature, k, aggr_nom_power=np.sum):
        super().__init__(feature, aggr_nom_power)
        self._k              = k
        self._component_to_k = None

    def _apply_algorithm(self, network: pypsa.Network, component_id: int) -> pd.Series:
        """
        Use a clustering or regionalisation algorithm to determine a mapping
        (bus -> aggregated bus) for the buses belonging to the respective component.
        """
        raise NotImplementedError('This method should be implemented by a subclass.')

    def _get_component_to_k(self, network: pypsa.Network, component_id: int) -> int:
        if self._component_to_k is None:
            self._component_to_k = self._distribute_k(network)
        return self._component_to_k[component_id]

    def _distribute_k(self, network):
        if len(self._components) == 1:
            return pd.Series(index=self._components.keys(), data=self._k)

        loads      = network.loads_t.p_set.mean().groupby(network.loads.bus).sum()
        weights    = pd.Series([])
        n_of_buses = pd.Series([])
        for key, component in self._components.items():
            weights[key]    = loads.reindex(component).sum()
            n_of_buses[key] = len(component)
        weights = weights.pipe(lambda x: (x / x.sum()).fillna(0.))

        solver_name = 'gurobi'

        if self._k < len(n_of_buses) or n_of_buses.sum() < self._k:
            raise ValueError(f'Choose k so that {len(n_of_buses)} <= k <= {n_of_buses.sum()}.')
        elif not np.isclose(weights.sum(), 1.0, rtol=1e-3):
            raise ValueError(f'Component weights must sum up to 1.0 when distributing clusters. Is {weights.sum()}.')

        def bounds(model, n_id):
            return (1, n_of_buses[n_id])
        
        m           = po.ConcreteModel()
        m.network   = po.Var(self._components.keys(), bounds=bounds, domain=po.Integers)
        m.tot       = po.Constraint(expr=(po.summation(m.network) == self._k))
        m.objective = po.Objective(expr=sum((m.network[i] - weights.loc[i] * self._k) ** 2 for i in weights.index),
                                    sense=po.minimize)

        opt = po.SolverFactory(solver_name)
        if not opt.has_capability('quadratic_objective'):
            opt = po.SolverFactory('ipopt')

        results = opt.solve(m)
        if not results['Solver'][0]['Status'] == 'ok':
            raise RuntimeError(f"Solver returned non-optimally: {results}")

        return pd.Series(m.network.get_values(), index=weights.index).astype(int)


class KmeansReducer(FixedKReducer):
    """
    Clusters buses in given network to k aggregated buses
    by applying the K-means++ clustering algorithm.
    """
    def _apply_algorithm(self, network: pypsa.Network, component_id: int) -> pd.Series:
        buses      = self._components[component_id]
        data       = self._get_feature_data(network, buses)
        n_clusters = self._get_component_to_k(network, component_id)

        kmeans = KMeans(init='k-means++', n_clusters=n_clusters)
        kmeans.fit(data.values)
        labels = kmeans.predict(data)
        return pd.Series(data=labels, index=data.index)

class MaxPRegionsReducer(ConcreteReducer):
    """
    A NetworkReducer using the max-p-regions algorithm.
    """
    def __init__(self,
                 feature: NetworkFeature,
                 spat_ext_attr: NetworkFeature,
                 threshold,
                 aggr_nom_power=np.sum):
        super().__init__(feature, aggr_nom_power)
        self._threshold          = threshold
        self._spat_ext_attr      = spat_ext_attr
        self._spat_ext_attr_data = None

    def _apply_algorithm(self, network: pypsa.Network, component_id: int) -> pd.Series:
        buses = self._components[component_id]
        if len(buses) == 1:
            return pd.Series({buses[0] : '0'})

        spat_ext_attr    = self._get_spat_ext_attr_data(network, buses)
        homogeneity_attr = self._get_feature_data(network, buses)

        if spat_ext_attr.sum().values <= self._threshold:
            return pd.Series(index=buses, data='0')

        local_search     = pr.azp.AZPBasicTabu(tabu_length=85)
        maxpr_heu        = maxpr.heuristics.MaxPRegionsHeu(local_search=local_search)

        neighbors        = self._get_reduced_adj_list(buses)
        spat_ext_attr    = NetworkFeature.to_dict(spat_ext_attr)
        homogeneity_attr = NetworkFeature.to_dict(homogeneity_attr)
        index            = sorted(neighbors)

        maxpr_heu.fit_from_dict(neighbors, homogeneity_attr, spat_ext_attr, self._threshold)

        busmap = pd.Series(data=maxpr_heu.labels_, index=index)
        return self._correct_bus_names(busmap)

    def _get_spat_ext_attr_data(self,
                                network: pypsa.Network,
                                buses) -> pd.DataFrame:
        if self._spat_ext_attr_data is None:
            self._spat_ext_attr_data = self._spat_ext_attr.get_data(network)
        return self._spat_ext_attr_data.reindex(buses)

class AgglomerativeReducer(FixedKReducer):
    """
    """
    def _apply_algorithm(self, network: pypsa.Network, component_id: int) -> pd.Series:
        buses = self._components[component_id]
        if len(buses) == 1:
            return pd.Series(index=buses, data='0')

        data     = self._get_feature_data(network, buses)
        index    = data.index
        affinity = 'euclidean'
        linkage  = 'ward'

        if self._feature is NetworkFeature.COORDINATES:
            data     = haversine_distances(data)
            affinity = 'precomputed'
            linkage  = 'complete'

        n_clusters   = self._get_component_to_k(network, component_id)
        connectivity = self._get_adjacency_matrix(index)

        labels = AgglomerativeClustering(n_clusters=n_clusters,
                                         connectivity=connectivity,
                                         affinity=affinity,
                                         linkage=linkage).fit_predict(data)
        busmap = pd.Series(data=labels, index=index)
        return self._correct_bus_names(busmap)
    
    def _get_adjacency_matrix(self, index):
        neighbors    = np.zeros((index.size, index.size))
        bus_to_index = dict(zip(index, range(index.size)))
        cmp_adj_list = self._get_reduced_adj_list(index)

        for bus in index:
            bus_index = bus_to_index[bus]
            for adjacent in cmp_adj_list[bus]:
                adj_index = bus_to_index[adjacent]
                neighbors[bus_index][adj_index] = 1.
                neighbors[adj_index][bus_index] = 1.

        return sparse.csr_matrix(neighbors)
