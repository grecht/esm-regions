import sys
from enum import Enum
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import pandas as pd
import numpy as np
import pypsa

sys.path.insert(0, '../scripts/')

class NetworkFeature(Enum):
    """
    Encapsulates extraction of network quantities.
    """
    COORDINATES           = "bus coordinates"
    AVG_LOAD              = "average load"
    SOLAR_CAP_FACTORS     = "solar capacity factor"
    SOLAR_TIMESERIES      = "solar availability timeseries"
    SOLAR_FLH             = "solar full load hours"
    WIND_CAP_FACTORS      = "wind capacity factor"
    WIND_TIMESERIES       = "wind availability timeseries"
    WIND_FLH              = "wind full load hours"
    SOLAR_WIND_TIMESERIES = "solar and wind timeseries"
    SOLAR_WIND_FLH        = "solar and wind full load hours"
    SOLAR_WIND_CAP        = "solar and wind capacity factors"
    SOLAR_WIND_FLH_MEAN   = "mean of solar and wind full load hours"
    SOLAR_WIND_CAP_MEAN   = "mean of solar and wind capacity factors"
    LINE_LENGTH_DIV_CAP   = "km/MW values for each line"

    def get_data(self,
                 network: pypsa.Network,
                 add_missing_rows=True,
                 fill_value=0.,
                 apply_std_scaler=True) -> pd.DataFrame:
        """
        Extract data from network.
        
        Standardisation is only applied to SOLAR_WIND_FLH.
        """
        data = pd.DataFrame([])
        
        if self is NetworkFeature.COORDINATES:
            data = self._get_coordinates(network)
        elif self is NetworkFeature.AVG_LOAD:
            data = self._get_avg_load(network)
        elif self is NetworkFeature.SOLAR_CAP_FACTORS:
            data = self._get_solar_cap_factors(network)
        elif self is NetworkFeature.WIND_CAP_FACTORS:
            data = self._get_wind_cap_factors(network)
        elif self is NetworkFeature.SOLAR_TIMESERIES:
            data = self._get_solar_timeseries(network)
        elif self is NetworkFeature.WIND_TIMESERIES:
            data = self._get_wind_timeseries(network)
        elif self is NetworkFeature.SOLAR_FLH:
            data = self._get_solar_flh(network)
        elif self is NetworkFeature.WIND_FLH:
            data = self._get_wind_flh(network)
        elif self is NetworkFeature.SOLAR_WIND_TIMESERIES:
            data = self._get_solar_wind_ts(network)
        elif self is NetworkFeature.SOLAR_WIND_CAP:
            data = self._get_solar_wind_cap(network)
        elif self is NetworkFeature.SOLAR_WIND_FLH:
            data = self._get_solar_wind_flh(network)
        elif self is NetworkFeature.SOLAR_WIND_CAP_MEAN:
            data = self._get_solar_wind_cap_mean(network)
        elif self is NetworkFeature.SOLAR_WIND_FLH_MEAN:
            data = self._get_solar_wind_flh_mean(network)
        elif self is NetworkFeature.LINE_LENGTH_DIV_CAP:
            data = self._get_line_length_div_cap(network)
            # TODO: should split NetworkFeature in BusFeature and LineFeature.
            add_missing_rows = False
        
        if apply_std_scaler and self in [NetworkFeature.SOLAR_WIND_FLH, NetworkFeature.SOLAR_WIND_CAP]:
            data = self._apply_std_scaler(data)
        if add_missing_rows:
            data = self._add_missing_rows(data, network, fill_value)
        return data

    def _get_coordinates(self, network: pypsa.Network) -> pd.DataFrame:
        return network.buses[['x', 'y']]

    def _get_avg_load(self, network: pypsa.Network) -> pd.DataFrame:
        avg_load = network.loads_t.p_set.mean().to_frame()
        return avg_load.rename(index=lambda x: x.split(' ')[0])

    def _get_solar_cap_factors(self, network: pypsa.Network) -> pd.DataFrame:
        so_cap_factors = network.generators_t.p_max_pu.filter(like='solar').mean().to_frame()
        return so_cap_factors.rename(index=lambda x: x.split(' ')[0])

    def _get_solar_timeseries(self, network: pypsa.Network) -> pd.DataFrame:
        year = network.generators_t.p_max_pu.filter(like='solar')
        data = year.T
        return data.rename(index=lambda x: x.split(' ')[0])
    
    def _get_solar_flh(self, network: pypsa.Network) -> pd.DataFrame:
        solar_flh = (network.generators_t.p_max_pu.filter(like='solar')).sum().to_frame()
        return solar_flh.rename(index=lambda x: x.split(' ')[0])

    def _get_wind_cap_factors(self, network: pypsa.Network) -> pd.DataFrame:
        wind_ts = self._get_wind_timeseries(network)
        return wind_ts.mean(axis=1).to_frame()

    def _get_wind_timeseries(self, network: pypsa.Network) -> pd.DataFrame:
        """
        For each bus with wind capacity, calculate combined availability timeseries
        for on- and offwind.
        For each bus, a weight is applied to each carrier equal to its contribution 
        to the summed capacity factor.
        """
        hourly_cap_factors = network.generators_t.p_max_pu.filter(like='wind').T
        yearly_cap_factors = hourly_cap_factors.mean(axis=1)
        summed_cap_factors = yearly_cap_factors.groupby(by=lambda x: x.split(' ')[0]).sum()
        
        weights = pd.Series([])
        for i in yearly_cap_factors.index:
            bus = i.split(' ')[0]
            weights[i] = np.divide(yearly_cap_factors[i], summed_cap_factors[bus])
        
        weighted_nominal_p = network.generators.p_nom_max.filter(like='wind').mul(weights, axis=0)
        weighted_active_p  = hourly_cap_factors.mul(weighted_nominal_p, axis=0)
        
        summed_active_p  = weighted_active_p.groupby(by=lambda x: x.split(' ')[0]).sum()
        summed_nominal_p = weighted_nominal_p.groupby(by=lambda x: x.split(' ')[0]).sum()
        
        return summed_active_p.div(summed_nominal_p, axis=0)

    def _get_wind_flh(self, network: pypsa.Network) -> pd.DataFrame:
        wind_ts  = self._get_wind_timeseries(network)
        return wind_ts.sum(axis=1).to_frame()

    def _get_solar_wind_ts(self, network: pypsa.Network) -> pd.DataFrame:
        solar_ts = self._get_solar_timeseries(network)
        wind_ts  = self._get_wind_timeseries(network)
        return pd.concat([solar_ts, wind_ts], axis=1, keys=['solar', 'wind'])
    
    def _get_solar_wind_cap(self, network:pypsa.Network) -> pd.DataFrame:
        solar_wind_ts = self._get_solar_wind_ts(network)
        return solar_wind_ts.mean(axis=1, level=0)
    
    def _get_solar_wind_cap_mean(self, network: pypsa.Network) -> pd.DataFrame:
        solar_wind_ts = self._get_solar_wind_ts(network)
        return solar_wind_ts.mean(axis=1, level=0).mean(axis=1).to_frame()
    
    def _get_solar_wind_flh(self, network: pypsa.Network) -> pd.DataFrame:
        solar_wind_ts = self._get_solar_wind_ts(network)
        return solar_wind_ts.sum(axis=1, level=0)
    
    def _get_solar_wind_flh_mean(self, network: pypsa.Network) -> pd.DataFrame:
        return self._get_solar_wind_flh(network).mean(axis=1).to_frame()
    
    def _get_line_length_div_cap(self, network: pypsa.Network) -> pd.DataFrame:
        columns = ['v_nom', 's_nom', 'num_parallel', 'under_construction']
        
        caps = [network.lines[(network.lines.v_nom == 220) & (network.lines.s_nom > 0)].s_nom.min(),
                network.lines[(network.lines.v_nom == 300) & (network.lines.s_nom > 0)].s_nom.min(),
                network.lines[(network.lines.v_nom == 380) & (network.lines.s_nom > 0)].s_nom.min()]
        
        data = [[220, caps[0], 1, False],[300, caps[1], 1, False],[380, caps[2], 1, False]]
        linedata = pd.DataFrame(index=[220, 300, 380], columns=columns, data=data)
        lines_copy = network.lines.copy()
        lines_copy[columns] = (network.lines[columns].apply(lambda b: linedata.loc[b.v_nom] if b.under_construction else b, axis=1))
        
        lines = lines_copy.length.combine(lines_copy.s_nom, lambda x, y: np.divide(x, y))
        lines = lines.to_frame('value').join(lines_copy[['bus0', 'bus1']])
        
        # TODO: add links?        
        return lines
        

    def _apply_std_scaler(self, data: pd.DataFrame) -> pd.DataFrame:
        std_scaler = StandardScaler().fit(data)
        return pd.DataFrame(std_scaler.transform(data),
                            index=data.index,
                            columns=data.columns)
    
    def _add_missing_rows(self,
                          data: pd.DataFrame,
                          network: pypsa.Network,
                          fill_value=np.NAN) -> pd.DataFrame:
        """
        Buses which do not have potential for renewable generation capacity are omitted in these
        datasets in PyPSA. To include them when clustering, join 'data' with the missing
        buses and a filler value.
        """
        missing   = list((set(network.buses.index) - set(data.index)))
        new_index = data.index.values.tolist() + missing
        return data.reindex(index=new_index, fill_value=fill_value)
    
    @staticmethod
    def to_dict(data: pd.DataFrame) -> dict:
        """
        Return data as dict with form {index -> value}.
        """
        # {index -> {column -> value}}
        data = data.to_dict(orient='index')
        # flatten to {index -> ndarray of value}
        return {k:np.array(list(v.values())) for (k,v) in data.items()}
