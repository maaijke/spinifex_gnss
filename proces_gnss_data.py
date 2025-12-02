from parse_gnss import DCBdata, GNSSData
from astropy.time import Time
import numpy as np
from spinifex.geometry import IPP
from gnss_geometry import get_stat_sat_ipp, get_sat_pos
from astropy.constants import c as speed_light
import astropy.units as u
from spinifex.ionospheric import get_density_ionex_single_layer, tec_data
from spinifex.ionospheric.iri_density import get_profile
from astropy.coordinates import EarthLocation
from concurrent.futures import as_completed, ProcessPoolExecutor

DISTANCE_DEGREES_CUT = 10
ELEVATION_CUT = 35


euref_station_file = "data/data_euref_pos.ssc2"
gnss_station_file = "data/data_gnss_pos.txt"
# TODO: get more gnss stations/databases
gnss_pos_dict = {}
with open(gnss_station_file) as myf:
    for line in myf:
        pos = [float(i) for i in line.strip().split()[1:]]
        gnss_pos_dict[line[:9]] = EarthLocation.from_geocentric(*pos, unit=u.m)


FREQ = {
    "G": {"f1": 1575.42e6, "f2": 1227.60e6},  # GPS L1/L2
    "R": {
        "f1": 1602.00e6 + 9 * 0.5625e6,
        "f2": 1246.00e6 + 9 * 0.4375e6,
    },  # nominal GLONASS; per-slot ideally
    "E": {"f1": 1575.42e6, "f2": 1191.795e6},  # Galileo E1/E5
    "C": {"f1": 1561.098e6, "f2": 1207.14e6},  # BeiDou B1/B2
    "J": {"f1": 1575.42e6, "f2": 1227.60e6},  # QZSS same as GPS
}


def get_transmission_time(
    c2: np.ndarray,
    times: Time,
    dcb_sat: float,
    dcb_stat: float,
) -> Time:
    """correct receiver times to satellite times at time of transmission using measured pathlength (approximation)

    Parameters
    ----------
    c2 : np.ndarray
        path length from pseudorange
    times : Time
        receiver times
    dcb_sat : float
        satellite bias (if known)
    dcb_stat : float
        station bias (if known)

    Returns
    -------
    Time
        transmission time
    """    
    distance = np.copy(c2)
    distance[np.isnan(distance)] = 0
    return times - (distance - (dcb_sat + dcb_stat)) * u.m / speed_light


def getpseudorange_tec(
    c1: np.ndarray,
    c2: np.ndarray,
    dcb_sat: float,
    dcb_stat: float,
    constellation: str = "G",
) -> np.ndarray:
    """Calculate stec based on pseudoranges 

    Parameters
    ----------
    c1 : np.ndarray
        measured pseudorange for "f1" frequency
    c2 : np.ndarray
        measured pseudorange for "f2" frequency
    dcb_sat : float
        satellite bias (if known)
    dcb_stat : float
        station bias (if known)
    constellation : str, optional
        satellite constellation, must be any of [G,R,E,C,J], by default G

    Returns
    -------
    np.ndarray
        pseudorange stec values for all times, nan if satellite not in sight
    """    
    C12 = 1e-16 / (
        40.3
        * (1.0 / FREQ[constellation]["f1"] ** 2 - 1.0 / FREQ[constellation]["f2"] ** 2)
    )
    return C12 * (
        c1
        - c2
        - (
            dcb_sat + dcb_stat
        )  # according to rinex3 definition there should be a minus sign here, but the data shows otherwise
    )


def getphase_tec(
    l1: np.ndarray, l2: np.ndarray, constellation: str = "G"
) -> np.ndarray:
    """Calculate stec based on carrier phases

    Parameters
    ----------
    l1 : np.ndarray
        measured phase for "f1" frequency
    l2 : np.ndarray
        measured phase for "f2" frequency
    constellation : str, optional
        satellite constellation, must be any of [G,R,E,C,J], by default G

    Returns
    -------
    np.ndarray
        carrier phase stec values for all times, nan if satellite not in sight
    """     
    C12 = 1e-16 / (
        40.3
        * (1.0 / FREQ[constellation]["f1"] ** 2 - 1.0 / FREQ[constellation]["f2"] ** 2)
    )
    WL1 = speed_light.value / FREQ[constellation]["f1"]
    WL2 = speed_light.value / FREQ[constellation]["f2"]
    return -C12 * (l1 * WL1 - l2 * WL2)


def _get_cycle_slips(phase_tec: np.ndarray) -> np.ndarray:
    """estimate cycle slip and data gaps based on the carrier phase stec data

    Parameters
    ----------
    phase_tec : np.ndarray
        stec calculated from carrier phases

    Returns
    -------
    np.ndarray
        array with indices of continues data
    """    
    diff = np.abs(np.diff(phase_tec, prepend=phase_tec[0]))
    slips = diff > 3 * np.nanmedian(diff)
    gaps = np.isnan(phase_tec)
    gaps = np.diff(gaps, prepend=0) > 0
    seg_id = np.cumsum(np.logical_or(slips, gaps).astype(int))
    return seg_id


def _get_phase_corrected(phase_tec: np.ndarray, pseudo_tec: np.ndarray) -> np.ndarray:
    """correct carrier phase calculated stec using pseudorange stec

    Parameters
    ----------
    phase_tec : np.ndarray
        stec calculated from carrier phases
    pseudo_tec : np.ndarray
        stec calculated from pseudoranges

    Returns
    -------
    np.ndarray
        bias corrected stec values
    """    
    cycle_slips = _get_cycle_slips(phase_tec=phase_tec)
    phase_bias = np.zeros_like(phase_tec)
    for seg in np.unique(cycle_slips):
        seg_idx = np.where(cycle_slips == seg)

        phase_bias[seg_idx] = np.nanmean(pseudo_tec[seg_idx] - phase_tec[seg_idx])
    return phase_tec + phase_bias


def get_gim_correction(
    stec_data: list[np.ndarray[float]], ipp_sat_stat: list[IPP], timestep: int = 10
) -> float:
    """
    Determine station biases based on GIM values
    Parameters
    ----------
    stec_data : list[np.ndarray[float]]
        measured stec data of a single gnss receiver many satellites of the same constellation
    ipp_sat_stat : list[IPP]
        list of the ipps of the station satellite pairs at fixed altitude
    timestep : int, optional
        to reduce cpu usage only calculate gim every timestep times, by default 10

    Returns
    -------
    float
        the estimated station bias
    """    
    dtec = []
    default_options = tec_data.IonexOptions()
    for ipp, stec in zip(ipp_sat_stat, stec_data):
        if type(ipp) == bool and not ipp:
            continue
        select = ipp.altaz.alt.deg[::timestep] > 65  # only use elevations above 65
        h_idx = np.argmin(
            np.abs(
                ipp.loc[0].height.to(u.km).value - default_options.height.to(u.km).value
            )
        )
        if not np.sum(select):
            continue
        ipp_select = IPP(
            altaz=ipp.altaz[::timestep][select],
            airmass=ipp.airmass[::timestep][select],
            loc=ipp.loc[::timestep][select],
            times=ipp.times[::timestep][select],
            los=ipp.los[::timestep][select],
            station_loc=ipp.station_loc,
        )
        gim_tec = get_density_ionex_single_layer(ipp=ipp_select)
        dtec.append(
            np.sum(gim_tec.electron_density, axis=1)
            - stec.tec_phase[::timestep][select] / ipp_select.airmass[:, h_idx]
        )
    # TODO : flag station if variance of dtec is too large, improve on gim based bias corrections, convert gps_time to utc?
    return np.nanmean(np.concatenate(dtec))


def _get_distance_degrees(loc1: EarthLocation, loc2: EarthLocation) -> float:
    return (
        (loc1.lon.deg - loc2.lon.deg) ** 2 + (loc1.lat.deg - loc2.lat.deg) ** 2
    ) ** 0.5  # TODO: make more correct (but as fast) estimate


def _get_distance_ipp(
    stec_values: np.ndarray,
    ipp_sat_stat: list[IPP],
    ipp_target: IPP,
    timeselect: np.ndarray,
    profiles: np.ndarray,  # normalized profile for times
) -> list[list[np.ndarray]]:
    """Select satellite data based on elevation and distance to the target ipp, return selected data for all times and heights

    Parameters
    ----------
    stec_values : np.ndarray
        array with stec values :satellites x times
    ipp_sat_stat : list[IPP]
        list of length satellites of ipp
    ipp_target : IPP
        ipp of the target
    timeselect : np.ndarray
        selected time indices (based on nearest neighbour)
    profiles : np.ndarray
        average normalized electron density profile for all times of ipp_target

    Returns
    -------
    list[list[np.ndarray]]
        list (times) of list (heights) of array with vtec, dlongitude (deg), dlatitude (deg) of selected satellites
    """    
    # stec_values : prn x gnss_times
    vtec_dlong_dlat_list = []
    for loc, timeidx, profile in zip(ipp_target.loc, timeselect, profiles):
        # loop over times
        vtec_dlong_dlat_list.append([])
        el_select = [
            i
            for i, ipp in enumerate(ipp_sat_stat)
            if ipp.altaz[timeidx].alt.deg > ELEVATION_CUT
        ]
        weighted_am = [
            np.sum(profile * ipp_sat_stat[i].airmass) for i in prn_select
        ]  # this gives the best (?) estimate of stec to vtec correction,
        # making sure that the sum over all vtec_i * am_i gives the measured stec
        for hidx, hloc in enumerate(loc):
            # loop over altitudes
            prn_select = [
                (i, am_weight)
                for i, am_weight in zip(el_select, weighted_am)
                if _get_distance_degrees(ipp_sat_stat[i].loc[timeidx, hidx], hloc)
                < DISTANCE_DEGREES_CUT
            ]

            vtec_dlong_dlat_list[-1].append(
                np.concatenate(
                    [
                        np.array(
                            stec_values[i, timeidx]
                            * profiles[timeidx, hidx]
                            / am_weight,
                            ipp_sat_stat[i].loc[timeidx, hidx].lon.deg - hloc.lon.deg,
                            ipp_sat_stat[i].loc[timeidx, hidx].lat.deg - hloc.lat.deg,
                        )[np.newaxis]
                        for i, am_weight in prn_select
                    ],
                    axis=0,
                )
            )
    return vtec_dlong_dlat_list


def get_interpolated_tec(
    input_data: list[list[np.ndarray]],
) -> np.ndarray:
    """The interpolation function, current implementation:
    fit a 2d linear gradient on inverse distance weighted data and return the data in the origin (location of the target ipp)

    Parameters
    ----------
    input_data : list[list[np.ndarray]]
      list (times) of list (heights) of array with vtec, dlongitude (deg), dlatitude (deg) of selected satellites-station combinations  

    Returns
    -------
    np.ndarray
        partial integrated electron density at the piercepoints
    """    
    fitted_density = np.zeros((len(input_data), len(input_data[0])))
    for timeidx, input_time in enumerate(input_data):
        for hidx, vtec_dlong_dlat in enumerate(input_time):
            A = np.ones_like(vtec_dlong_dlat)
            A[:, 1:] = vtec_dlong_dlat[:, 1:]
            # linear_fit
            w = 1.0 / np.linalg.norm(A[:, 1:], axis=1)  # weight with inverse distance
            Aw = w[:, np.newaxis] * A
            par = np.linalg.inv(Aw.T @ Aw) @ (Aw.T @ vtec_dlong_dlat[:, :1])
            fitted_density[timeidx, hidx] = par[0] # We need the offset at the origin
    return fitted_density
# TODO: Add error bars


def _get_dcb_value(dcbdata, c1_str, c2_str, prn) -> float:
    """Helper function to read dcb data from DCBData object
    """    
    dcb = dcbdata.dcb
    dcbkeys = dcbdata.station_code_combination
    if f"{str(prn)}_{c1_str}_{c2_str}" in dcbkeys:
        index = dcbkeys.index(f"{str(prn)}_{c1_str}_{c2_str}")
        return dcb[index][0]
    else:
        # print warning
        return 0.0


def get_gnss_station_density(
    gnss_data: GNSSData, dcb: DCBdata, ipp_target: IPP, profiles: np.ndarray, sat_pos_object
) -> list[list[np.ndarray]]:
    """For a given GNSS receiver get the bias corrected vtec data and distance in longitude and latitude to the desired target ipp

    Parameters
    ----------
    gnss_data : GNSSData
        receiver data and times (single constellation)
    dcb : DCBdata
        bias data
    ipp_target : IPP
        ionospheric piercepoints of the target
    profiles : np.ndarray
        average normalized density profiles for all times 

    Returns
    -------
    list[list[np.ndarray]]
        list (times) of list (heights) of array with vtec, dlongitude (deg), dlatitude (deg) of selected satellites
    """    
    dcb_stat = _get_dcb_value(dcb, gnss_data.c1_str, gnss_data.c2_str, gnss_data.station)
    prns = sorted(gnss_data.gnss.keys())
    stec_values = []
    ipp_sat_stat = []
    timeselect = np.argmin(
        np.abs(ipp_target.times.mjd - gnss_data.times.mjd[:, np.newaxis]),
        axis=0,
    )  # nearest neighbour interpolation in time, TODO: correct gps_time to UTC
    for prn in prns:
        sat_data = gnss_data.gnns[prn]
        dcb_sat = _get_dcb_value(dcb, gnss_data.c1_str, gnss_data.c2_str, prn)
        transmission_time = get_transmission_time(
            sat_data.c2, gnss_data.times, dcb_sat=dcb_sat, dcb_stat=dcb_stat
        )
        pseudo_stec = getpseudorange_tec(
            sat_data.c1,
            sat_data.c2,
            dcb_sat=dcb_sat,
            dcb_stat=dcb_stat,
            constellation=gnss_data.constellation,
        )
        phase_stec = getphase_tec(
            sat_data.l1, sat_data.l2, constellation=gnss_data.constellation
        )
        phase_stec = _get_phase_corrected(
            phase_stec, pseudo_stec
        )  # correct bias per cycle slip
        sat_pos = get_sat_pos(
                sat_pos_object, transmission_time, prn
        )
        ipp_sat_stat.append(get_stat_sat_ipp(
                satpos=sat_pos,
                gnsspos=gnss_pos_dict[gnss_data.station],
                times=gnss_data.times,
                height_array=ipp_target.loc[0].height,
        ) ) 
        stec_values.append(phase_stec)
    # correction probably needs to be per satellite constellation, this can be solved by creating a gnss_data object
    # per constellation
    correction = get_gim_correction(
        stec_data=stec_values, ipp_sat_stat=ipp_sat_stat, timestep=10
    )
    # Now make a subselection in time, elevation and distance to ipp. returns approximate vtec,dlong and dlat list for every h x time combination
    return _get_distance_ipp(
        stec_values=np.array(stec_values) + correction,
        ipp_sat_stat=ipp_sat_stat,
        ipp_target=ipp_target,
        timeselect=timeselect,
        profiles=profiles,
    )  # list of list of stec, airmass, dlong, dlat values per height and time of ipp_target


def get_ipp_density(
    ipp_target: IPP, gnss_data_list: list[GNSSData], dcb: DCBdata, sat_pos_object
) -> tec_data.ElectronDensity:
    """Get the partial integrated electron density at the ipps 

    Parameters
    ----------
    ipp_target : IPP
        ionospheric piercepoint of target
    gnss_data_list : list[GNSSData]
        list with all available GNSSData objects
    dcb : DCBdata
        bias data

    Returns
    -------
    tec_data.ElectronDensity
        partial integrated electron density at the piercepoints
    """    
    profiles = get_profile(ipp_target)
    Ntimes = ipp_target.times.shape[0]
    Nheights = ipp_target.loc.shape[1]
    all_data = [[[] for _ in range(Nheights)] for _ in range(Ntimes)] # Ntimes x Nheights
    stec_gnss_data = {}
    with ProcessPoolExecutor(max_workers=6) as executor:
        # Submit all tasks
        future_to_station_constellation = {
                executor.submit(get_gnss_station_density, gnss_data, dcb, ipp_target, profiles, sat_pos_object) : gnss_data.station+gnss_data.constellation
                for gnss_data in gnss_data_list
         }

        # Collect results as they complete
        for future in as_completed(future_to_station_constellation):
            station = future_to_station_constellation[future]
            try:
                result = future.result()
                stec_gnss_data[station] = result
            except Exception as e:
                stec_gnss_data[station] = f"Error: {e}"
    
    for station,station_data in stec_gnss_data.items:
        if  type(station_data) == str:
            print ("error for", station, station_data)
            continue
        for itm in range(Ntimes):
            for hidx in range(Nheights):
                all_data[itm][hidx].append(station_data[itm][hidx])        
    for itm in range(Ntimes):
        for hidx in range(Nheights):
            all_data[itm][hidx] = np.concatenate(all_data[itm][hidx], axis=0)
    electron_density = get_interpolated_tec(all_data)
    return tec_data.ElectronDensity(electron_density=electron_density, electron_density_error=np.zeros_like(electron_density))
