"""
Collocate one MODIS granule's ship-track mask + L1B reflectances + MYD03
geometry + MYD06 cloud retrieval into a single per-pixel table.

All four sources share the native (2030, 1354) swath grid -- no resampling
required, direct index correspondence confirmed in verify_mask_alignment.py.

Usage: python collocate_real_data.py <region> [--limit N]
Output: data/modis_real/<region>_collocated.parquet
"""
import sys, os, argparse, json
import numpy as np
import pandas as pd
from PIL import Image
from pyhdf.SD import SD, SDC
from scipy import ndimage, stats
from scipy.spatial import cKDTree

BASE = os.path.dirname(os.path.abspath(__file__))
MASK_DIR = os.path.join(BASE, 'data', 'ship_track_masks', 'extracted',
                        'Full_Sized_Images', 'MaskedImages')
DATA_DIR = os.path.join(BASE, 'data', 'modis_real')
VIRIDIS_FLOOR = np.array([68, 1, 84])

# Segrin et al. (2007)-style background-validity parameters
BG_WINDOW = 15   # +/- pixels (~15 km at 1 km sampling) searched for background
BG_MIN_N = 20    # minimum valid background (liquid, non-track) pixels required

EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def local_background_stats(track, valid_liquid, re_mod06, window=BG_WINDOW, min_n=BG_MIN_N):
    """
    For every pixel, compute the mean MOD06 re and count of valid background
    (liquid-phase, non-track) pixels in a (2*window+1) square window --
    the observational proxy for 'nearby out-of-track pixels' (Segrin et al.,
    2007), evaluated at every pixel rather than per manually-drawn segment.
    """
    bg_candidate = valid_liquid & ~track
    re_bg = np.where(bg_candidate, np.nan_to_num(re_mod06, nan=0.0), 0.0)
    size = 2 * window + 1
    bg_count = ndimage.uniform_filter(bg_candidate.astype(np.float64), size=size,
                                      mode='constant') * size * size
    bg_sum = ndimage.uniform_filter(re_bg, size=size, mode='constant') * size * size
    with np.errstate(invalid='ignore', divide='ignore'):
        bg_mean = bg_sum / bg_count
    bg_valid = bg_count >= min_n
    return bg_mean, bg_count, bg_valid


MIN_COMPONENT_PX = 30  # a connected track component smaller than this can't
                       # support a reliable PCA axis or orientation test
MIN_ORIENT_PX = 15     # minimum finite re_mod06 samples needed to orient the axis


def compute_along_track_position(track, lat, lon, re_mod06,
                                 min_component_px=MIN_COMPONENT_PX,
                                 min_orient_px=MIN_ORIENT_PX):
    """
    Intrinsic along-track distance, replacing the heads.json-based absolute
    matching (heads.json is in a different, unusable coordinate frame -- see
    alignment_test.py). For each connected component of the track mask:
    project pixels onto the component's own PCA principal axis (in a local
    tangent-plane km frame), then orient that axis using the MOD06 re_mod06
    Twomey signature (head = smaller-re end) -- an OBSERVATIONAL variable,
    not the model's own inferred S_hat, so the later decay-of-S_hat test
    isn't circular. Returns dist_along_track (km, 0 at the low-re end),
    NaN where a component is too small to support a component axis or an
    orientation decision.
    """
    labeled, n_comp = ndimage.label(track, structure=np.ones((3, 3)))
    dist_along_track = np.full(track.shape, np.nan)
    if n_comp == 0:
        return dist_along_track

    for comp_id in range(1, n_comp + 1):
        mask_c = labeled == comp_id
        ys, xs = np.where(mask_c)
        if len(ys) < min_component_px:
            continue  # too small for a reliable axis

        clat, clon = lat[ys, xs].mean(), lon[ys, xs].mean()
        dy = (lat[ys, xs] - clat) * (np.pi / 180.0) * EARTH_R_KM
        dx = (lon[ys, xs] - clon) * (np.pi / 180.0) * EARTH_R_KM * np.cos(np.radians(clat))
        coords_c = np.column_stack([dx, dy])
        coords_c -= coords_c.mean(axis=0)

        _, _, vt = np.linalg.svd(coords_c, full_matrices=False)
        t = coords_c @ vt[0]   # signed position along the component's own axis

        re_c = re_mod06[ys, xs]
        finite = np.isfinite(re_c)
        if finite.sum() < min_orient_px:
            continue  # can't reliably orient this component

        rho, _ = stats.spearmanr(t[finite], re_c[finite])
        if not np.isfinite(rho):
            continue
        if rho < 0:
            t = -t   # orient so distance increases with re_mod06 (away from head)

        dist_along_track[ys, xs] = t - t.min()

    return dist_along_track


def read_sds(sd, name, apply_scaling=True):
    ds = sd.select(name)
    arr = ds[:].astype(np.float64)
    attrs = ds.attributes()
    fill = attrs.get('_FillValue')
    if fill is not None:
        arr = np.where(arr == fill, np.nan, arr)
    if apply_scaling and 'scale_factor' in attrs:
        scale = attrs.get('scale_factor', 1.0)
        offset = attrs.get('add_offset', 0.0)
        arr = (arr - offset) * scale
    return arr


def read_l1b_band(sd, dataset_name, band_index):
    ds = sd.select(dataset_name)
    attrs = ds.attributes()
    raw = ds[:][band_index].astype(np.float64)
    fill = attrs.get('_FillValue')
    if fill is not None:
        raw = np.where(raw == fill, np.nan, raw)
    scale = attrs.get('reflectance_scales')[band_index]
    offset = attrs.get('reflectance_offsets')[band_index]
    return (raw - offset) * scale


def collocate_granule(granule_key, gdir, region):
    fname_key = granule_key.replace('.', '_')
    mask_path = os.path.join(MASK_DIR, f'{fname_key}_Masked.png')
    if not os.path.exists(mask_path):
        return None
    mask = np.array(Image.open(mask_path))
    track = ~np.all(mask[..., :3] == VIRIDIS_FLOOR, axis=-1)

    l1b_file = next((f for f in os.listdir(gdir) if f.startswith('MYD021KM')), None)
    m03_file = next((f for f in os.listdir(gdir) if f.startswith('MYD03')), None)
    m06_file = next((f for f in os.listdir(gdir) if f.startswith('MYD06')), None)
    if not all([l1b_file, m03_file, m06_file]):
        return None

    l1b = SD(os.path.join(gdir, l1b_file), SDC.READ)
    m03 = SD(os.path.join(gdir, m03_file), SDC.READ)
    m06 = SD(os.path.join(gdir, m06_file), SDC.READ)

    refl_086 = read_l1b_band(l1b, 'EV_250_Aggr1km_RefSB', 1)   # band 2 = 0.86um
    # Band 7 (2.1um) is a native-500m band aggregated to 1km, so it lives in
    # EV_500_Aggr1km_RefSB (band_names "3,4,5,6,7" -> index 4), NOT in
    # EV_1KM_RefSB (whose bands are 8,9,10,...,26 -- index 4 there is band 12,
    # a ~0.55um ocean-color channel). Using EV_1KM_RefSB was a bug: it fed the
    # VAE band 12 reflectance labelled as "2.1um" for every real pixel.
    refl_213 = read_l1b_band(l1b, 'EV_500_Aggr1km_RefSB', 4)   # band 7 = 2.1um

    lat = read_sds(m03, 'Latitude', apply_scaling=False)
    lon = read_sds(m03, 'Longitude', apply_scaling=False)
    # read_sds() already applies each SDS's own scale_factor (0.01 deg/count for
    # these MYD03 fields) -- do NOT rescale again here (that was a double-scaling
    # bug: it silently shrank every angle by 100x and broke the raz wraparound,
    # which operates on true degrees).
    solz = read_sds(m03, 'SolarZenith')
    satz = read_sds(m03, 'SensorZenith')
    sola = read_sds(m03, 'SolarAzimuth')
    sata = read_sds(m03, 'SensorAzimuth')
    raz = np.abs(((sola - sata + 180) % 360) - 180)

    # MODIS L1B "reflectance" (reflectance_scales product) is rho*cos(solz),
    # whereas the DISORT training data (libRadtran output_quantity
    # 'reflectivity') is the BRF rho = pi*L/(mu0*E0). Divide by cos(solz) so
    # real and synthetic inputs share the BRF convention. Verified empirically
    # 16/07/2026: thick overcast MODIS refl correlates with mu0 at Spearman
    # +0.77 (killed by this division), while the training refl does not.
    mu0 = np.cos(np.deg2rad(solz))
    refl_086 = refl_086 / mu0
    refl_213 = refl_213 / mu0

    re_mod06 = read_sds(m06, 'Cloud_Effective_Radius')
    tau_mod06 = read_sds(m06, 'Cloud_Optical_Thickness')
    phase = read_sds(m06, 'Cloud_Phase_Optical_Properties', apply_scaling=False)

    shape = track.shape
    for arr, name in [(refl_086, 'refl_086'), (refl_213, 'refl_213'), (lat, 'lat'),
                       (solz, 'solz'), (satz, 'satz'), (re_mod06, 're_mod06'),
                       (tau_mod06, 'tau_mod06'), (phase, 'phase')]:
        if arr.shape != shape:
            raise ValueError(f'{name} shape {arr.shape} != mask shape {shape} for {granule_key}')

    # liquid-phase only (phase == 2), valid reflectances, valid MOD06 retrieval
    valid = ((phase == 2) & (refl_086 > 0) & (refl_086 < 1.5) &
             (refl_213 > 0) & (refl_213 < 1.5) & ~np.isnan(re_mod06) & ~np.isnan(tau_mod06))
    if valid.sum() == 0:
        return None

    bg_mean, bg_count, bg_valid = local_background_stats(track, valid, re_mod06)
    dist_along_track = compute_along_track_position(track, lat, lon, re_mod06)

    df = pd.DataFrame({
        'granule_key': granule_key, 'region': region,
        'lat': lat[valid], 'lon': lon[valid],
        'track': track[valid],
        'refl_086': refl_086[valid], 'refl_213': refl_213[valid],
        'solz': solz[valid], 'satz': satz[valid], 'raz': raz[valid],
        're_mod06': re_mod06[valid], 'tau_mod06': tau_mod06[valid],
        'bg_re_mean': bg_mean[valid], 'bg_n': bg_count[valid], 'bg_valid': bg_valid[valid],
        'dist_along_track_km': dist_along_track[valid],
    })
    return df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('region')
    ap.add_argument('--limit', type=int, default=None)
    args = ap.parse_args()

    region_dir = os.path.join(DATA_DIR, args.region)
    granule_dirs = sorted(os.listdir(region_dir)) if os.path.isdir(region_dir) else []
    granule_dirs = [g for g in granule_dirs if os.path.isdir(os.path.join(region_dir, g))]
    if args.limit:
        granule_dirs = granule_dirs[:args.limit]

    frames = []
    for i, gkey_fs in enumerate(granule_dirs):
        # directory names and mask filenames both use the YYYYDDD_HHMM form
        try:
            df = collocate_granule(gkey_fs, os.path.join(region_dir, gkey_fs), args.region)
        except Exception as e:
            print(f'  [{i+1}/{len(granule_dirs)}] {gkey_fs}: FAILED ({e})')
            continue
        if df is None:
            print(f'  [{i+1}/{len(granule_dirs)}] {gkey_fs}: no valid pixels / missing files')
            continue
        n_track = df['track'].sum()
        print(f'  [{i+1}/{len(granule_dirs)}] {gkey_fs}: {len(df)} valid liquid pixels, '
              f'{n_track} track ({100*n_track/len(df):.2f}%)')
        frames.append(df)

    if frames:
        out = pd.concat(frames, ignore_index=True)
        out_path = os.path.join(DATA_DIR, f'{args.region}_collocated.parquet')
        out.to_parquet(out_path, index=False)
        n_track = out['track'].sum()
        track_rows = out[out['track']]
        n_track_valid_bg = track_rows['bg_valid'].sum()
        n_track_positioned = track_rows['dist_along_track_km'].notna().sum()
        print(f'\n{len(out):,} total pixels from {len(frames)} granules -> {out_path}')
        print(f'  track pixels: {n_track:,} ({100*n_track/len(out):.2f}%)')
        print(f'  track pixels with valid background (Analysis 1/2 eligible): '
              f'{n_track_valid_bg:,} ({100*n_track_valid_bg/max(n_track,1):.1f}% of track pixels)')
        print(f'  track pixels with an oriented along-track position (Analysis 3 eligible): '
              f'{n_track_positioned:,} ({100*n_track_positioned/max(n_track,1):.1f}% of track pixels)')
    else:
        print('\nNo granules collocated successfully.')
