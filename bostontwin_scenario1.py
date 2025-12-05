"""
BostonTwin + Sionna RT Network Deployment:
- Cellular uses REAL antenna locations from BostonTwin dataset
- Other services use random deployment inside tile bounds
"""


import os
import numpy as np
import drjit as dr
import mitsuba as mi
import geopandas as gpd
from pathlib import Path

# Enable GPU execution
# mi.set_variant("cuda_ad_rgb")   # required for Sionna RT


# Import Sionna RT
try:
    import sionna.rt
except ImportError:
    os.system("pip install sionna-rt")
    import sionna.rt

from sionna.rt import (
    load_scene,
    ITURadioMaterial,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver,
    Camera,
)

# BostonTwin library
from src.classes.BostonTwin import BostonTwin


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
# Adjust this path to where your BostonTwin dataset actually is
DATASET_DIR = Path("neu_ms35xx11z/bostontwin")
SCENE_NAME  = "BOS_G_5"    # choose valid tile name from tiles_dict

no_preview  = True  # Set to False for GUI preview


# ------------------------------------------------------------
# SERVICE DEFINITIONS
# ------------------------------------------------------------
services = [
    {"name": "Sensor",   "Ntx": 1, "Nrx": 10, "H_tx": 25, "Pt_dBm": 40},
    {"name": "Radionav", "Ntx": 1, "Nrx": 10, "H_tx": 40, "Pt_dBm": 60},
    {"name": "Radioloc", "Ntx": 1, "Nrx": 10, "H_tx": 60, "Pt_dBm": 70},
    # Cellular will be overwritten using real BostonTwin antennas
    {"name": "Cellular", "Ntx": 0, "Nrx": 20, "H_tx": 25, "Pt_dBm": 46},
]


# ------------------------------------------------------------
# LOAD BOSTONTWIN TILE INTO SIONNA RT
# ------------------------------------------------------------
def load_bostontwin_scene(dataset_dir: Path, scene_name: str):
    bostwin = BostonTwin(dataset_dir)
    print(f"[INFO] Loading BostonTwin tile: {scene_name}")

    scene, scene_antennas = bostwin.load_bostontwin(scene_name)

    # Tile bounds for deployment
    tileinfo_path = bostwin.boston_model.tiles_dict[scene_name]["tileinfo_path"]
    gdf = gpd.GeoDataFrame.from_file(tileinfo_path)

    minx, miny, maxx, maxy = gdf.total_bounds

    # Convert ALL to Python float
    minx = float(minx)
    maxx = float(maxx)
    miny = float(miny)
    maxy = float(maxy)

    cx = float(0.5 * (minx + maxx))
    cy = float(0.5 * (miny + maxy))

    bounds = (minx, maxx, miny, maxy)

    # Camera must use pure python floats
    cam = Camera(
        position=[cx, cy, float(450.0)],
        look_at=[cx, cy, float(0.0)],
    )

    if no_preview:
        scene.render(camera=cam)
    else:
        scene.preview()

    return scene, cam, bounds, scene_antennas


# ------------------------------------------------------------
# RANDOM DEPLOYMENT FOR NON-CELLULAR SERVICES
# ------------------------------------------------------------
def deploy_random(services, bounds, z_rx=1.5):
    minx, maxx, miny, maxy = bounds
    deployments = []

    for svc in services:
        if svc["name"] == "Cellular":
            # Skip here, replaced by real antenna positions later
            continue

        name = svc["name"]
        Ntx  = svc["Ntx"]
        Nrx  = svc["Nrx"]
        Htx  = svc["H_tx"]

        # TX random
        tx = np.column_stack([
            np.random.uniform(minx, maxx, Ntx),
            np.random.uniform(miny, maxy, Ntx),
            np.ones(Ntx) * Htx
        ])

        # RX random
        rx = np.column_stack([
            np.random.uniform(minx, maxx, Nrx),
            np.random.uniform(miny, maxy, Nrx),
            np.ones(Nrx) * z_rx
        ])

        deployments.append({"name": name, "tx": tx, "rx": rx})
        print(f"[DEPLOY] Random for {name}: {Ntx} TX, {Nrx} RX")

    return deployments


# ------------------------------------------------------------
# REAL CELLULAR ANTENNAS FROM BOSTONTWIN
# ------------------------------------------------------------
def extract_xyz(df):
    # --------------------------------------------------------
    # 1. Extract from geometry (BEST OPTION)
    # --------------------------------------------------------
    if "geometry" in df.columns:
        print("[INFO] Extracting XYZ from geometry (Shapely Points)")

        xs = df.geometry.x.to_numpy()
        ys = df.geometry.y.to_numpy()

        # Assign default antenna height (25 m)
        zs = np.ones_like(xs) * 25.0

        return np.column_stack([xs, ys, zs])

    # --------------------------------------------------------
    # 2. Extract from Lat/Long if geometry missing
    # --------------------------------------------------------
    if "Lat" in df.columns and "Long" in df.columns:
        print("[INFO] Using Lat/Long as coordinates")

        xs = df["Long"].to_numpy()
        ys = df["Lat"].to_numpy()

        zs = np.ones_like(xs) * 25.0
        return np.column_stack([xs, ys, zs])

    # --------------------------------------------------------
    # 3. No valid columns -> fail
    # --------------------------------------------------------
    print("\n[ERROR] Cannot find geometry or Lat/Long columns.")
    print("Available columns:", df.columns.tolist())
    raise KeyError("Unable to extract antenna coordinates.")


def deploy_real_cellular(scene_antennas, services):
    df = scene_antennas

    if len(df) == 0:
        print("[WARNING] No antennas found in this tile.")
        return None

    # ---- Extract REAL antenna positions ----
    TX = extract_xyz(df)

    # Number of RX users to create
    cellular_spec = [s for s in services if s["name"] == "Cellular"][0]
    Nrx = cellular_spec["Nrx"]

    minx, miny = TX[:, 0].min(), TX[:, 1].min()
    maxx, maxy = TX[:, 0].max(), TX[:, 1].max()

    # Random RX around antennas
    rx = np.column_stack([
        np.random.uniform(minx, maxx, Nrx),
        np.random.uniform(miny, maxy, Nrx),
        np.ones(Nrx) * 1.5
    ])

    print(f"[DEPLOY] REAL Cellular: {TX.shape[0]} BS, {Nrx} RX")

    return {"name": "Cellular", "tx": TX, "rx": rx}

# ------------------------------------------------------------
# INSERT TX/RX OBJECTS INTO SIONNA SCENE
# ------------------------------------------------------------
def insert_nodes(scene, deployments):
    for svc in deployments:
        name = svc["name"]

        # TX
        for i, p in enumerate(svc["tx"]):
            pos = [float(p[0]), float(p[1]), float(p[2])]
            scene.add(
                Transmitter(
                    f"{name}_tx_{i}",
                    position=pos,
                    display_radius=3
                )
            )

        # RX
        for j, p in enumerate(svc["rx"]):
            pos = [float(p[0]), float(p[1]), float(p[2])]
            scene.add(
                Receiver(
                    f"{name}_rx_{j}",
                    position=pos,
                    display_radius=2
                )
            )

    print("[INFO] Inserted all TX/RX nodes into the BostonTwin scene.")


# ------------------------------------------------------------
# PATH COMPUTATION
# ------------------------------------------------------------
def compute_paths(scene):
    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="tr38901",
        polarization="V"
    )

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(
        num_rows=1,
        num_cols=1,
        vertical_spacing=0.5,
        horizontal_spacing=0.5,
        pattern="dipole",
        polarization="cross"
    )

    solver = PathSolver()
    paths = solver(scene, max_depth=6)

    print("[INFO] Path computation complete.")
    return paths


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    np.random.seed(7)

    # Load BostonTwin tile
    scene, cam, bounds, scene_antennas = load_bostontwin_scene(
        DATASET_DIR, SCENE_NAME
    )

    # Deploy non-cellular services randomly
    deployments = deploy_random(services, bounds)

    # Add REAL Cellular antennas
    cellular_deploy = deploy_real_cellular(scene_antennas, services)
    if cellular_deploy is not None:
        deployments.append(cellular_deploy)

    # Add all nodes to the Sionna scene
    insert_nodes(scene, deployments)

    # Compute multipath
    paths = compute_paths(scene)

    # Visualize
    if no_preview:
        scene.render(camera=cam, paths=paths)
    else:
        scene.preview(paths=paths)

    print("\n[DONE] BostonTwin + SionnaRT network deployed successfully.\n")


if __name__ == "__main__":
    main()
