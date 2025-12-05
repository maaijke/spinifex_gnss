from pathlib import Path
from spinifex.vis_tools.ms_tools import get_metadata_from_ms

ms_metadata = get_metadata_from_ms(
    Path("/home/mevius/spinifex_scripts/ms_data/nen_obs/3c380_SB222.MS")
)
from gnss_tec import *
from spinifex.geometry import get_ipp_from_skycoord
import matplotlib.pyplot as plt
from spinifex.ionospheric.iri_density import get_profile
from spinifex.magnetic import magnetic_models
HEIGHT_ARRAY = np.arange(50, 2000, 10) * u.km
select_st = 0

times = ms_metadata.times[::60]
ipps = [get_ipp_from_skycoord(
    source=ms_metadata.source,
    height_array=HEIGHT_ARRAY,
    loc=loc,
    times=times,
) for loc in ms_metadata.locations]
edensity = get_electron_density_gnss(ipps[select_st])
profile = get_profile(ipps[select_st])[:90]
field = magnetic_models.ppigrf(ipps[select_st]).magnetic_field.to(u.nT).value[:90]
sorted_keys = [i for i in sorted(list(edensity.keys())) if not type(edensity[i]) == str]


distance = np.concatenate([np.array([i for i in edensity[key][1]]) for key in sorted_keys], axis=0)
am = np.concatenate([np.array([i for i in edensity[key][2]]) for key in sorted_keys], axis=0)
tec = np.concatenate([np.array([i + edensity[key][4] for i in edensity[key][0]]) for key in sorted_keys], axis=0)
elevation = np.concatenate([np.array([i for i in edensity[key][3]]) for key in sorted_keys], axis=0)

matec = np.ma.array(tec, mask=elevation<35)
madist = np.ma.array(distance, mask=np.broadcast_to(elevation[...,np.newaxis],distance.shape)<35)
sigma = 10e5
weights = np.ma.exp(-(madist/sigma)**2)
sum_weights= np.ma.sum(weights, axis=0, keepdims=True)

weighted_am = np.sum(profile*am, axis=-1, keepdims=True)
weighted_tec = matec[...,np.newaxis]*weights*profile/weighted_am
wtec=np.ma.sum(weighted_tec,axis=0)/sum_weights[0]
b_field_to_rm = -2.62e-6
rm = np.ma.sum(
        b_field_to_rm
        * wtec
        * field
        * ipps[select_st].airmass[:90],
        axis=1,
    )

if False:
    tec = [
        [i.tec_phase + edensity[j][2] for i, k in zip(edensity[j][0], edensity[j][1])]
        for j in sorted_keys
    ]


    ipp_1 = edensity[sorted_keys[0]][1]

    lons = np.zeros((len(sorted_keys), 32, ipp_1[0].times.shape[0]))
    lats = np.zeros((len(sorted_keys), 32, ipp_1[0].times.shape[0]))
    tecs_time = np.zeros((len(sorted_keys), 32, ipp_1[0].times.shape[0]))
    lons[:] = np.nan
    lats[:] = np.nan
    tecs_time[:] = np.nan
    el = [
        [i.altaz.alt.deg if not type(i) == bool else None for i in edensity[j][1]]
        for j in sorted_keys
    ]
    am = [
        [i.airmass if not type(i) == bool else None for i in edensity[j][1]]
        for j in sorted_keys
    ]
    for i in range(len(sorted_keys)):
        ipp_1 = edensity[sorted_keys[i]][1]
        el1 = el[i]
        am1 = am[i]
        for j in range(len(ipp_1)):
            if type(ipp_1[j]) == bool:
                continue
            # plt.scatter(ipp_1[j].loc[:,31].lon.deg[el1[j]>35], ipp_1[j].loc[:,31].lat.deg[el1[j]>35], c=tec[i][j][el1[j]>35] + np.mean(np.concatenate(dtecs[i])), vmin=0, vmax=40)
            lons[i, j, el1[j] > 35] = ipp_1[j].loc[:, 29].lon.deg[el1[j] > 35]
            lats[i, j, el1[j] > 35] = ipp_1[j].loc[:, 29].lat.deg[el1[j] > 35]
            tecs_time[i, j, el1[j] > 35] = tec[i][j][el1[j] > 35]/am1[j][:,29][el1[j] > 35]
    for itm in range(lats.shape[-1]):
        plt.cla()
        plt.scatter(
            lons[..., itm],
            lats[..., itm],
            c=tecs_time[..., itm],
            vmin=20,
            vmax=45,
            cmap="plasma",
            s=100,
            alpha=0.5,
        )
        plt.pause(0.1)
