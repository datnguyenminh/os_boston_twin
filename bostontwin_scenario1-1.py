"""
BostonTwin + Sionna RT Network Deployment:
- Cellular uses REAL antenna locations from BostonTwin dataset
- Other services use random deployment inside tile bounds
"""

# ------------------------------------------------------------
# Mitsuba backend (AMD → CPU/LLVM)
# ------------------------------------------------------------
import mitsuba as mi
mi.set_variant("llvm_ad_rgb")   # AD-capable CPU backend for Sionna RT
print("Mitsuba variant:", mi.variant())

import os
import numpy as np
import drjit as dr
import geopandas as gpd
import matplotlib.pyplot as plt
from pathlib import Path

# Optional: background map for Lat/Lon plot
try:
    import contextily as ctx
except ImportError:
    ctx = None
    print("[INFO] 'contextily' not installed, Lat/Lon basemap will be skipped.")

# Import Sionna RT
try:
    import sionna.rt
except ImportError:
    os.system("pip install sionna-rt")
    import sionna.rt

from sionna.rt import (
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
DATASET_DIR = Path("neu_ms35xx11z/bostontwin")  # adjust to your path
SCENE_NAME  = "BOS_G_5"                         # valid tile name

no_preview  = True   # we now always skip 3D render to avoid Color3f error


# ------------------------------------------------------------
# SERVICE DEFINITIONS
# ------------------------------------------------------------
services = [
    {"name": "Sensor",   "Ntx": 2, "Nrx": 10, "H_tx": 25, "Pt_dBm": 40},
    {"name": "Radionav", "Ntx": 2, "Nrx": 10, "H_tx": 40, "Pt_dBm": 60},
    {"name": "Radioloc", "Ntx": 2, "Nrx": 10, "H_tx": 60, "Pt_dBm": 70},
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

    # IMPORTANT: we skip scene.render()/preview() to avoid Color3f error
    # If you really want a 3D preview later, we can build a custom RGB material.

    return scene, cam, bounds, scene_antennas, tileinfo_path


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
    # 1) Extract from geometry (preferred)
    if "geometry" in df.columns:
        print("[INFO] Extracting XYZ from geometry (Shapely Points)")
        xs = df.geometry.x.to_numpy()
        ys = df.geometry.y.to_numpy()
        zs = np.ones_like(xs) * 25.0   # default antenna height
        return np.column_stack([xs, ys, zs])

    # 2) Fallback: Lat/Long columns
    if "Lat" in df.columns and "Long" in df.columns:
        print("[INFO] Using Lat/Long as coordinates")
        xs = df["Long"].to_numpy()
        ys = df["Lat"].to_numpy()
        zs = np.ones_like(xs) * 25.0
        return np.column_stack([xs, ys, zs])

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
# PLOT ANTENNA LOCATIONS (Lat/Lon + XY[m])
# ------------------------------------------------------------
def plot_cellular_deployment(df, TX, tileinfo_path, scene_name):
    """
    Left: Antennas on Lat/Lon map
    Right: Antennas in local XY [m] over BostonTwin tile
    """
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    plt.suptitle("Antenna Location", fontsize=18)

    # ---------------- LEFT: Lat/Lon plot --------------------
    if "Lat" in df.columns and "Long" in df.columns:
        lon = df["Long"].to_numpy()
        lat = df["Lat"].to_numpy()
        gdf_geo = gpd.GeoDataFrame(
            df.copy(),
            geometry=gpd.points_from_xy(lon, lat),
            crs="EPSG:4326"
        )

        gdf_geo.plot(ax=axs[0], marker="o",
                     color="dodgerblue", markersize=10, alpha=0.8)

        axs[0].set_title(scene_name)
        axs[0].set_xlabel("Longitude")
        axs[0].set_ylabel("Latitude")

        if ctx is not None:
            try:
                ctx.add_basemap(
                    axs[0],
                    crs=gdf_geo.crs.to_string(),
                    source=ctx.providers.CartoDB.Positron
                )
            except Exception as e:
                print("[WARNING] Could not load basemap:", e)
    else:
        axs[0].text(0.5, 0.5,
                    "Lat/Long not available",
                    ha="center", va="center", fontsize=12)
        axs[0].set_axis_off()

    # --------------- RIGHT: Local XY plot -------------------
    tile_gdf = gpd.read_file(tileinfo_path)
    tile_gdf.plot(ax=axs[1], color="black", alpha=0.5)

    axs[1].scatter(TX[:, 0], TX[:, 1], s=25, color="red")
    axs[1].set_title(scene_name)
    axs[1].set_xlabel("X [m]")
    axs[1].set_ylabel("Y [m]")
    axs[1].grid(True)

    plt.tight_layout()
    plt.show()


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
    scene, cam, bounds, scene_antennas, tileinfo_path = load_bostontwin_scene(
        DATASET_DIR, SCENE_NAME
    )

    # Deploy non-cellular services randomly
    deployments = deploy_random(services, bounds)

    # Add REAL Cellular antennas
    cellular_deploy = deploy_real_cellular(scene_antennas, services)
    if cellular_deploy is not None:
        deployments.append(cellular_deploy)

        # Plot antenna locations (Lat/Lon and XY[m])
        plot_cellular_deployment(
            df=scene_antennas,
            TX=cellular_deploy["tx"],
            tileinfo_path=tileinfo_path,
            scene_name=SCENE_NAME
        )

    # Add all nodes to the Sionna scene
    insert_nodes(scene, deployments)

    # Compute multipath (no 3D render)
    paths = compute_paths(scene)

    print("\n[DONE] BostonTwin + SionnaRT network deployed successfully.\n")


if __name__ == "__main__":
    main()
