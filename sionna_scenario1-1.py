"""
Computes: Received Power, Interference, SINR, SE
"""

import os
import numpy as np
import drjit as dr
import mitsuba as mi
import matplotlib.pyplot as plt

# Import Sionna RT
try:
    import sionna.rt
except ImportError:
    os.system("pip install sionna-rt")
    import sionna.rt

from sionna.rt import (
    load_scene,
    SceneObject,
    ITURadioMaterial,
    Transmitter,
    Receiver,
    PlanarArray,
    PathSolver,
    Camera,
    watt_to_dbm,
)

no_preview = True


# ============================================================
# Utility functions for performance
# ============================================================
def dbm_to_watt(Pdbm):
    return 10 ** ((Pdbm - 30) / 10)

def watt_to_db(W):
    return 10 * np.log10(W)

def spectral_efficiency(sinr):
    return np.log2(1 + sinr)


# ============================================================
# Compute received power per TX-RX link
# ============================================================
def compute_link_budget(paths, scene, deployments, services):
    """
    Returns:
        performance[service_name] = {
            'rx_pwr': list of received signal power per link
            'sinr': list of sinr per link
            'se': list of spectral efficiency per link
        }
    """

    # Extract path amplitudes & delays
    a, tau = paths.cir(out_type="numpy")  # complex amplitudes

    # Shapes
    # a: [num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
    a = a[..., 0]  # remove time dimension → shape [RX, ant1, TX, ant2, paths]

    num_rx = a.shape[0]
    num_tx = a.shape[2]

    performance = {}

    # Global noise (assume B = 400 MHz)
    noise_dbm = -174 + 10*np.log10(400e6)
    noise_watt = dbm_to_watt(noise_dbm)

    # Map service - list of its TX and RX indices
    tx_map = {}
    rx_map = {}
    tx_counter = 0
    rx_counter = 0

    for svc in deployments:
        name = svc["name"]
        Ntx = len(svc["tx"])
        Nrx = len(svc["rx"])
        tx_map[name] = list(range(tx_counter, tx_counter + Ntx))
        rx_map[name] = list(range(rx_counter, rx_counter + Nrx))
        tx_counter += Ntx
        rx_counter += Nrx

    # Loop over services
    for idx, svc in enumerate(services):
        svc_name = svc["name"]
        Pt = dbm_to_watt(svc["Pt_dBm"])

        svc_tx_indices = tx_map[svc_name]
        svc_rx_indices = rx_map[svc_name]

        rx_powers = []
        sinrs = []
        ses = []

        # Compute link-by-link
        for rx_idx in svc_rx_indices:
            for tx_idx in svc_tx_indices:

                # signal: sum |a|^2
                signal = np.sum(np.abs(a[rx_idx, 0, tx_idx, 0, :]) ** 2) * Pt

                # interference: all TX from other services
                interference = 0
                for other_svc in services:
                    if other_svc["name"] == svc_name:
                        continue

                    for other_tx in tx_map[other_svc["name"]]:
                        interference += np.sum(
                            np.abs(a[rx_idx, 0, other_tx, 0, :]) ** 2
                            * dbm_to_watt(other_svc["Pt_dBm"])
                        )

                sinr = signal / (interference + noise_watt)
                se = spectral_efficiency(sinr)

                rx_powers.append(signal)
                sinrs.append(sinr)
                ses.append(se)

        performance[svc_name] = {
            "rx_pwr": np.array(rx_powers),
            "sinr": np.array(sinrs),
            "se": np.array(ses),
        }

    return performance


# ============================================================
# Plot performance
# ============================================================
def plot_performance(perf):
    services = list(perf.keys())

    # ---------------------- Histogram of SINR -------------------------
    plt.figure(figsize=(10,6))
    for svc in services:
        plt.hist(10*np.log10(perf[svc]["sinr"]), bins=20, alpha=0.5, label=svc)
    plt.xlabel("SINR (dB)")
    plt.ylabel("Count")
    plt.title("SINR Distribution Across Services")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------- Histogram of SE -------------------------
    plt.figure(figsize=(10,6))
    for svc in services:
        plt.hist(perf[svc]["se"], bins=20, alpha=0.5, label=svc)
    plt.xlabel("Spectral Efficiency (bits/s/Hz)")
    plt.ylabel("Count")
    plt.title("SE Distribution Across Services")
    plt.legend()
    plt.grid(True)
    plt.show()

    # ---------------------- Average values -------------------------
    avg_sinr = [np.mean(10*np.log10(perf[svc]["sinr"])) for svc in services]
    avg_se = [np.mean(perf[svc]["se"]) for svc in services]

    x = np.arange(len(services))

    plt.figure(figsize=(10,6))
    plt.bar(x - 0.2, avg_sinr, width=0.4, label="Avg SINR (dB)")
    plt.bar(x + 0.2, avg_se, width=0.4, label="Avg SE (bits/s/Hz)")
    plt.xticks(x, services)
    plt.title("Average Performance per Service")
    plt.legend()
    plt.grid(True)
    plt.show()

# ============================================================
# 1. Load scenes exactly as tutorial
# ============================================================
def load_street_canyon_examples():
    print("\n=== Loading simple street canyon, merge=False ===")
    scene_A = load_scene(sionna.rt.scene.simple_street_canyon,
                         merge_shapes=False)

    for name, obj in scene_A.objects.items():
        print(f"{name:<15}{obj.radio_material.name}")

    print("\n=== Loading simple street canyon, merge=True ===")
    scene_B = load_scene(sionna.rt.scene.simple_street_canyon,
                         merge_shapes=True)

    for name, obj in scene_B.objects.items():
        print(f"{name:<15}{obj.radio_material.name}")

    print("\n=== Excluding some buildings from merge ===")
    scene_C = load_scene(
        sionna.rt.scene.simple_street_canyon,
        merge_shapes=True,
        merge_shapes_exclude_regex=r"building_[0-2]$"
    )

    for name, obj in scene_C.objects.items():
        print(f"{name:<15}{obj.radio_material.name}")

    return scene_C  # we return a scene to modify if needed


# ============================================================
# 2. Load etoile scene for editing (tutorial example)
# ============================================================
def load_etoile_scene():
    scene = load_scene(sionna.rt.scene.etoile)
    cam = Camera(position=[-360,145,400], look_at=[-115,33,1.5])

    if no_preview:
        scene.render(camera=cam)
    else:
        scene.preview()

    return scene, cam


# ============================================================
# 3. Add cars around the monument 
# ============================================================
def add_tutorial_cars(scene):
    num_cars = 5

    car_material = ITURadioMaterial(
        "car-material", 
        "metal", 
        thickness=0.01, 
        color=(0.8, 0.1, 0.1)
    )

    # Instantiate car objects
    cars = [
        SceneObject(
            fname=sionna.rt.scene.low_poly_car,
            name=f"car-{i}",
            radio_material=car_material
        )
        for i in range(num_cars)
    ]

    # Add all cars at origin
    scene.edit(add=cars)

    # === Circle placement ===
    center = mi.Point3f(-127, 37, 1.5)
    radius = 100

    # Vector of angles
    thetas = dr.linspace(mi.Float, 0., dr.two_pi, num_cars, endpoint=False)

    # Compute circle positions (vectorized)
    car_positions = center + mi.Point3f(dr.cos(thetas), dr.sin(thetas), 0.) * radius

    # Tangent directions → look_at points
    d = dr.normalize(car_positions - center)
    tangents = mi.Vector3f(d.y, -d.x, 0.)
    look_points = car_positions + tangents

    # === Assign per-car positions (SCALAR mode!) ===
    for i in range(num_cars):
        cars[i].position = mi.Point3f(
            float(car_positions.x[i]),
            float(car_positions.y[i]),
            float(car_positions.z[i])
        )

        cars[i].look_at(
            mi.Point3f(
                float(look_points.x[i]),
                float(look_points.y[i]),
                float(look_points.z[i])
            )
        )

    # Scale first car
    cars[0].scaling = 3.0

    return cars, car_positions



# ============================================================
# 4. Wireless network service definitions
# ============================================================
services = [
    {"name": "Sensor",   "Ntx": 2, "Nrx": 5, "H_tx": 25, "Pt_dBm": 40},
    {"name": "Radionav", "Ntx": 2, "Nrx": 5, "H_tx": 40, "Pt_dBm": 60},
    {"name": "Radioloc", "Ntx": 2, "Nrx": 5, "H_tx": 60, "Pt_dBm": 70},
    {"name": "Cellular", "Ntx": 2, "Nrx": 5, "H_tx": 20, "Pt_dBm": 46},
]


# ============================================================
# 5. Deploy network inside etoile bounding box
# ============================================================
def deploy_network(scene, services, xlim=(-300,100), ylim=(-200,200)):
    deployments = []

    for svc in services:
        name  = svc["name"]
        Ntx   = svc["Ntx"]
        Nrx   = svc["Nrx"]
        Htx   = svc["H_tx"]

        # TX positions (random)
        tx_x = np.random.uniform(xlim[0], xlim[1], Ntx)
        tx_y = np.random.uniform(ylim[0], ylim[1], Ntx)
        tx_z = np.ones(Ntx)*Htx
        tx   = np.column_stack([tx_x, tx_y, tx_z])

        # RX positions (random)
        rx_x = np.random.uniform(xlim[0], xlim[1], Nrx)
        rx_y = np.random.uniform(ylim[0], ylim[1], Nrx)
        rx_z = np.ones(Nrx)*1.0
        rx   = np.column_stack([rx_x, rx_y, rx_z])

        deployments.append({"name":name, "tx":tx, "rx":rx})

    return deployments


# ============================================================
# 6. Add TX/RX to scene
# ============================================================
def insert_nodes(scene, deployments):
    for svc in deployments:
        name = svc["name"]

        # TX
        for i, p in enumerate(svc["tx"]):
            scene.add(Transmitter(f"{name}_tx_{i}", position=p.tolist(),
                                  display_radius=3))

        # RX
        for j, p in enumerate(svc["rx"]):
            scene.add(Receiver(f"{name}_rx_{j}", position=p.tolist(),
                               display_radius=2))

    print("[INFO] Wireless network nodes inserted.")


# ============================================================
# 7. Compute paths
# ============================================================
def compute_paths(scene):
    
    # Configure antenna array for all transmitters
    scene.tx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="tr38901",
                                polarization="V")

    # Configure antenna array for all receivers
    scene.rx_array = PlanarArray(num_rows=1,
                                num_cols=1,
                                vertical_spacing=0.5,
                                horizontal_spacing=0.5,
                                pattern="dipole",
                                polarization="cross")


    solver = PathSolver()
    paths = solver(scene, max_depth=5)

    print("[INFO] Path computation done.")
    return paths

# ============================================================
# 9. MAIN (updated)
# ============================================================
def main():
    np.random.seed(1)

    # Load base scene
    scene, cam = load_etoile_scene()

    # Add cars
    cars, car_positions = add_tutorial_cars(scene)

    # Deploy network
    deployments = deploy_network(scene, services)
    insert_nodes(scene, deployments)

    # Compute paths
    paths = compute_paths(scene)

    # Compute performance
    print("[INFO] Computing network performance …")
    perf = compute_link_budget(paths, scene, deployments, services)

    # Print simple summary
    for svc in perf:
        print(f"\n=== {svc} ===")
        print("Avg SINR (dB):", np.mean(10*np.log10(perf[svc]["sinr"])))
        print("Avg SE (bits/s/Hz):", np.mean(perf[svc]["se"]))

    # Plot results
    plot_performance(perf)

    # Render scene
    if no_preview:
        scene.render(camera=cam, paths=paths)
    else:
        scene.preview(paths=paths)

    print("\n[DONE] Full evaluation complete.")



if __name__ == "__main__":
    main()
