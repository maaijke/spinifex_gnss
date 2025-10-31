import asyncio
import gps_time
from datetime import datetime, timedelta
from spinifex.download import download_or_copy_url
from spinifex.asyncio_wrapper import sync_wrapper
import subprocess
from pathlib import Path
from astropy.time import Time
from typing import NamedTuple
import requests


class RinexFiles(NamedTuple):
    """Object containing all necessary information to dowload rinex data"""

    filenames: list[list[str]]
    """list of list with online file names, one per date"""
    times: list[Time]
    """list of times for unique days"""
    dates: Time
    """unique days"""
    url: str
    """url of the data"""
    stations: list[list[str]]
    """list of available 4 letter stations"""


def get_gps_week(date: datetime):
    gpstime = gps_time.GPSTime.from_datetime(date)
    return gpstime.week_number, int(gpstime.time_of_week / (24 * 3600))


async def _download_satpos_files_coro(
    date: datetime,
    url: str = "https://cddis.nasa.gov/archive/gnss/products/",
    datapath: Path = Path("/home/mevius/IONO/GPS/data/"),
) -> list[Path]:

    sp3_names = []
    yesterday = date - timedelta(days=1)
    tomorrow = date + timedelta(days=1)
    gpsweek, _ = get_gps_week(date)
    yesterweek, _ = get_gps_week(yesterday)
    tomorrowweek, _ = get_gps_week(tomorrow)
    sp3_names.append(
        f"{url}{yesterweek}/GFZ0OPSFIN_{yesterday.year}{yesterday.timetuple().tm_yday:03d}0000_01D_05M_ORB.SP3.gz"
    )
    sp3_names.append(
        f"{url}{gpsweek}/GFZ0OPSFIN_{date.year}{date.timetuple().tm_yday:03d}0000_01D_05M_ORB.SP3.gz"
    )
    sp3_names.append(
        f"{url}{tomorrowweek}/GFZ0OPSFIN_{tomorrow.year}{tomorrow.timetuple().tm_yday:03d}0000_01D_05M_ORB.SP3.gz"
    )
    clk_name = f"{url}{gpsweek}/GFZ0OPSFIN_{date.year}{date.timetuple().tm_yday:03d}0000_01D_30S_CLK.CLK.gz"
    sp3_names.append(clk_name)
    coros = []
    for url in sp3_names:
        coros.append(download_or_copy_url(url, output_directory=datapath))

    return await asyncio.gather(*coros)


def download_satpos_files(
    date: datetime,
    url: str = "https://cddis.nasa.gov/archive/gnss/products/",
    datapath: Path = Path("/home/mevius/IONO/GPS/data/"),
) -> list[Path]:
    """Get the sp3 position files and corresponding clock errors for a specific date, the day before and the day after for interpoaltion purposes

    Parameters
    ----------
    date : datetime
        requested date
    url : str, optional
        server from where to download the data, by default "https://cddis.nasa.gov/archive/gnss/products/"
    datapath : Path, optional
        output directory, by default Path("/home/mevius/IONO/GPS/data/")

    Returns
    -------
    list[Path] : list with the filenames of respectively three sorted sp3 files and clock errors of date

    """
    sp3_files = _download_satpos_files(date, url, datapath)
    for (
        fname
    ) in sp3_files:  # need to unpack because georinex fails on reading zipped sp3 data
        subprocess.run(["gunzip", "-f", str(fname)])
    return [i.parent / i.stem for i in sp3_files]


async def download_dcb_coro(
    date: datetime, datapath: Path = Path("/home/mevius/IONO/GPS/data/")
) -> Path:
    """Download differential code biases for a given date

    Parameters
    ----------
    date : datetime
        requested date
    datapath : Path, optional
        output directory, by default Path("/home/mevius/IONO/GPS/data/")

    Returns
    -------
    Path
        location of the data
    """
    # TODO: find optional servers for these dcb files
    server = "https://data.bdsmart.cn/pub/product/bias/"
    doy = date.timetuple().tm_yday
    url = f"{server}{date.year}/CAS0MGXRAP_{date.year}{doy:03d}0000_01D_01D_DCB.BSX.gz"
    dcb = download_or_copy_url(url, output_directory=datapath)
    return await asyncio.gather(dcb)


def check_url(url: str):
    """
    Check if a given url exists
    """
    response = requests.head(url)
    return response.status_code == 200


async def download_rinex_coro(
    date: datetime,
    stations: list[str],
    server: str = "https://www.epncb.oma.be/pub/obs/",
    datapath: Path = Path("/home/mevius/IONO/GPS/data/"),
):
    # TODO: get naming format for dates and servers (like we do for ionex data)
    urls = []
    year = date.year
    doy = date.timetuple().tm_yday
    url = f"{server}/{year}/{doy:03d}/"
    for station in stations:
        fname = f"{station}_R_{year}{doy:03d}0000_01D_30S_MO.crx.gz"
        if check_url(f"{url}{fname}"):
            urls.append(f"{url}{fname}")
        else:
            print(f"{url}{fname} not existing")

    coros = []
    for url in urls:
        coros.append(download_or_copy_url(url, output_directory=datapath))
    return await asyncio.gather(*coros)


download_rinex = sync_wrapper(download_rinex_coro)
download_dcb = sync_wrapper(download_dcb_coro)
_download_satpos_files = sync_wrapper(_download_satpos_files_coro)
