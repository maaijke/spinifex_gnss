from gnss_tec import select_gnss_stations
from astropy.coordinates import EarthLocation, AltAz
import astropy.units as u
from spinifex.geometry import get_ipp_from_altaz
from astropy.time import Time
import numpy as np
from pathlib import Path
from gnss_geometry import get_sat_pos_object, get_stat_sat_ipp, get_sat_pos
import glob
from proces_gnss_data import gnss_pos_dict
import matplotlib.pyplot as plt


datapath = Path("./data/")
ska = EarthLocation.from_geocentric(-2565115.544092137,5085629.234035365,-2861196.7160952426, unit=u.m)
n16 = EarthLocation.from_geocentric(-2562938.284486403,5104952.106135408,-2828733.076250165, unit=u.m)
e16 = EarthLocation.from_geocentric(-2600277.002781707,5060626.574694427,-2873673.335992341, unit=u.m)
s16 = EarthLocation.from_geocentric(-2534438.384733236,5087834.4671821175,-2884260.496429776, unit=u.m)

times  = Time("2025-06-26T00:00:00") + np.arange(60*24) * 1 * u.min
zenith = AltAz(alt = 90 *u.deg, az=0* u.deg, obstime=times[0], location=ska)

ipp = get_ipp_from_altaz(altaz=zenith, height_array=np.array([350.,])*u.km, loc=ska)
gnss = select_gnss_stations(ipp_location=ipp.loc)
sp3_files = [Path(i) for i in sorted(glob.glob(datapath.as_posix() +  "/*SP3.gz"))]
sat_pos_object = get_sat_pos_object(sp3_files = sp3_files)
ipps = []
ipps_ska = []
ipps_n16 = []
ipps_s16 = []
ipps_e16 = []

for prn in sat_pos_object.sv:
    sat_pos = get_sat_pos(sat_pos_object, times, prn)
    for key in gnss:
        ipps.append(
                get_stat_sat_ipp(
                    satpos=sat_pos,
                    gnsspos=gnss_pos_dict[key],
                    times=times,
                    height_array=np.array([350.,])*u.km,
                )
            )
    ipps_ska.append(get_stat_sat_ipp(
                    satpos=sat_pos,
                    gnsspos=ska,
                    times=times,
                    height_array=np.array([350.,])*u.km,
                ))
    ipps_n16.append(get_stat_sat_ipp(
                    satpos=sat_pos,
                    gnsspos=n16,
                    times=times,
                    height_array=np.array([350.,])*u.km,
                ))
    ipps_s16.append(get_stat_sat_ipp(
                    satpos=sat_pos,
                    gnsspos=s16,
                    times=times,
                    height_array=np.array([350.,])*u.km,
                ))
    ipps_s16.append(get_stat_sat_ipp(
                    satpos=sat_pos,
                    gnsspos=s16,
                    times=times,
                    height_array=np.array([350.,])*u.km,
                ))    
    ipps_e16.append(get_stat_sat_ipp(
                    satpos=sat_pos,
                    gnsspos=e16,
                    times=times,
                    height_array=np.array([350.,])*u.km,
                ))

lonlat = np.zeros((len(ipps), 1440,2), dtype=float)
lonlat_ska = np.zeros((len(ipps_ska), 1440,2), dtype=float)
lonlat_n16 = np.zeros((len(ipps_n16), 1440,2), dtype=float)
lonlat_s16 = np.zeros((len(ipps_s16), 1440,2), dtype=float)
lonlat_e16 = np.zeros((len(ipps_e16), 1440,2), dtype=float)

for idx,ip in enumerate(ipps):
    tselect = ip.altaz.alt.deg>35
    lonlat[idx,tselect,:] = np.array((ip.loc[tselect].to_geodetic().lon.deg, ip.loc[tselect].to_geodetic().lat.deg)).T
for idx,ip in enumerate(ipps_ska):
    tselect = ip.altaz.alt.deg>35
    lonlat_ska[idx,tselect,:] = np.array((ip.loc[tselect].to_geodetic().lon.deg, ip.loc[tselect].to_geodetic().lat.deg)).T
for idx,ip in enumerate(ipps_n16):
    tselect = ip.altaz.alt.deg>35
    lonlat_n16[idx,tselect,:] = np.array((ip.loc[tselect].to_geodetic().lon.deg, ip.loc[tselect].to_geodetic().lat.deg)).T
for idx,ip in enumerate(ipps_s16):
    tselect = ip.altaz.alt.deg>35
    lonlat_s16[idx,tselect,:] = np.array((ip.loc[tselect].to_geodetic().lon.deg, ip.loc[tselect].to_geodetic().lat.deg)).T    
for idx,ip in enumerate(ipps_e16):
    tselect = ip.altaz.alt.deg>35
    lonlat_e16[idx,tselect,:] = np.array((ip.loc[tselect].to_geodetic().lon.deg, ip.loc[tselect].to_geodetic().lat.deg)).T    

for itm in range(0,1440,10):
    plt.cla()
    plt.scatter(lonlat[:,itm,0],lonlat[:,itm,1])
    plt.scatter(lonlat_ska[:,itm,0],lonlat_ska[:,itm,1],s=50,c='r')
    plt.scatter(lonlat_n16[:,itm,0],lonlat_n16[:,itm,1],s=50,c='k')
    plt.scatter(lonlat_s16[:,itm,0],lonlat_s16[:,itm,1],s=50,c='g')
    plt.scatter(lonlat_e16[:,itm,0],lonlat_e16[:,itm,1],s=50,c='m')    
    plt.pause(.1)
