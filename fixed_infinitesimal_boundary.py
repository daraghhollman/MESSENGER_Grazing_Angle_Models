"""
In this script, we look at the distribution of grazing angles as would've been observed for crossings only occuring along the Winslow+ 2013 boundary models.

The average aberrated frame is used for computational speed.
"""

import datetime as dt
import multiprocessing
from typing import Iterable
import numpy as np
import pandas as pd
import shapely
import shapely.plotting
import shapely.vectorized
import matplotlib.pyplot as plt
from tqdm import tqdm
from hermpy import trajectory, utils, plotting


def main():
    plot = False
    resolution = 600 # seconds

    # This could be potentially vectorised, allowing for a roll operation on the regions to find crossings.
    time_steps, positions = Get_Times_And_Positions(resolution=resolution)

    # We define the boundaries as polygons (closing them at the end)
    magnetopause_points = Get_Magnetopause()
    magnetopause_region = shapely.Polygon(magnetopause_points)

    bow_shock_points = Get_Bow_Shock()
    bow_shock_region = shapely.Polygon(bow_shock_points)

    # We then check for the entire trajectory what region the spacecraft is in.
    print("Finding where points lie")
    regions = Check_Region_Vectorised(positions, bow_shock_region, magnetopause_region)

    # Now we iterrate through the regions and find the indices when the region changes
    # When the region changes, we mark a crossing and determine the grazing angle.
    bow_shock_crossing_times = []
    magnetopause_crossing_times = []

    previous_region = regions[0]
    for i, current_region in tqdm(
        enumerate(regions[1:], start=1),
        desc="Searching for crossings",
        total=len(regions) - 1,
    ):

        if previous_region != current_region:
            # Found a crossing
            if (
                previous_region == "solar wind" and current_region == "magnetosheath"
            ) or (
                previous_region == "magnetosheath" and current_region == "solar wind"
            ):
                bow_shock_crossing_times.append(
                    time_steps[i - 1] + (time_steps[i] - time_steps[i - 1]) / 2
                )
            else:
                magnetopause_crossing_times.append(
                    time_steps[i - 1] + (time_steps[i] - time_steps[i - 1]) / 2
                )

        previous_region = current_region

    # We must define the crossings in the same way as we would for a crossing list.
    # This is so that we can use our grazing angle code.
    bow_shock_positions = (
        trajectory.Get_Avg_Aberrated_Position("MESSENGER", bow_shock_crossing_times)
        / utils.Constants.MERCURY_RADIUS_KM
    )
    magnetopause_positions = (
        trajectory.Get_Avg_Aberrated_Position("MESSENGER", magnetopause_crossing_times)
        / utils.Constants.MERCURY_RADIUS_KM
    )

    bow_shock_crossings = {
        "Start Time": bow_shock_crossing_times,
        "End Time": bow_shock_crossing_times,
        "X MSM'": bow_shock_positions[:, 0],
        "Y MSM'": bow_shock_positions[:, 1],
        "Z MSM'": bow_shock_positions[:, 2],
    }
    magnetopause_crossings = {
        "Start Time": magnetopause_crossing_times,
        "End Time": magnetopause_crossing_times,
        "X MSM'": magnetopause_positions[:, 0],
        "Y MSM'": magnetopause_positions[:, 1],
        "Z MSM'": magnetopause_positions[:, 2],
    }

    if plot:
        for crossings in [bow_shock_crossings, magnetopause_crossings]:
            fig, ax = plt.subplots()

            ax.scatter(
                crossings["positions"][:, 0],
                np.sqrt(
                    crossings["positions"][:, 1] ** 2
                    + crossings["positions"][:, 2] ** 2
                ),
                alpha=0.1,
                color="indianred",
            )

            plotting.Plot_Magnetospheric_Boundaries(ax)
            plotting.Format_Cylindrical_Plot(ax)

            plt.show()

            fig, ax = plt.subplots()
            ax.scatter(
                crossings["positions"][:, 0],
                crossings["positions"][:, 1],
                alpha=0.1,
                color="indianred",
            )
            plotting.Plot_Magnetospheric_Boundaries(ax)
            plotting.Plot_Mercury(ax, shaded_hemisphere="left", frame="MSM", plane="xy")
            plotting.Square_Axes(ax, 6)
            plotting.Add_Labels(ax, plane="xy", frame="MSM")

            plt.show()

            fig, ax = plt.subplots()
            ax.scatter(
                crossings["positions"][:, 0],
                crossings["positions"][:, 2],
                alpha=0.1,
                color="indianred",
            )
            plotting.Plot_Magnetospheric_Boundaries(ax)
            plotting.Plot_Mercury(ax, shaded_hemisphere="left", frame="MSM", plane="xz")
            plotting.Square_Axes(ax, 6)
            plotting.Add_Labels(ax, plane="xz", frame="MSM")

            plt.show()

    # We then need to calculate grazing angles for each crossing
    bow_shock_crossings = pd.DataFrame(bow_shock_crossings)
    magnetopause_crossings = pd.DataFrame(magnetopause_crossings)

    grazing_angles = trajectory.Get_Grazing_Angle(bow_shock_crossings, function="bow shock", aberrate="average")

    print(grazing_angles)



def Get_Grazing_Angles(input):
    crossing, function = input
    return trajectory.Get_Grazing_Angle(crossing, function=function, aberrate="average")


def Get_Times_And_Positions(
    resolution: int = 600,
    cpus: int = multiprocessing.cpu_count() - 1,
):
    # Define the start and end time, along with a resolution
    mission_start = dt.datetime(2011, 3, 23, 15, 37)
    mission_end = dt.datetime(2012, 4, 30, 15, 8)

    mission_duration = int((mission_end - mission_start).total_seconds())

    time_steps = np.array(
        [
            mission_start + dt.timedelta(seconds=int(t))
            for t in np.arange(0, mission_duration, resolution)
        ]
    )

    time_chunks = np.array_split(time_steps, cpus)

    print("Time steps calculated")

    with multiprocessing.Pool(cpus) as pool:
        results = list(
            tqdm(
                pool.imap(Compute_Chunk_Positions, time_chunks),
                desc="Finding positions",
                total=len(time_chunks),
            )
        )

    positions = np.concatenate(results)

    print("Found positions")

    return time_steps, positions


def Compute_Chunk_Positions(time_chunk):

    positions = trajectory.Get_Avg_Aberrated_Position("MESSENGER", time_chunk)
    positions /= utils.Constants.MERCURY_RADIUS_KM

    return positions


def Check_Region_Vectorised(positions, bow_shock_region, magnetopause_region):

    x, yz = positions[:, 0], np.sqrt(positions[:, 1] ** 2 + positions[:, 2] ** 2)

    within_bow_shock = shapely.vectorized.contains(bow_shock_region, x, yz)

    within_magnetopause = shapely.vectorized.contains(magnetopause_region, x, yz)

    regions = np.full(len(positions), "unknown", dtype=object)

    regions[within_bow_shock & within_magnetopause] = "magnetosphere"
    regions[within_bow_shock & ~within_magnetopause] = "magnetosheath"
    regions[~within_bow_shock & ~within_magnetopause] = "solar wind"

    if np.any(regions == "unknown"):
        raise ValueError("Can't determine spacecraft region")

    return regions


def Get_Magnetopause():
    sub_solar_magnetopause = 1.45  # radii
    alpha = 0.5

    phi = np.linspace(0, 2 * np.pi, 1000)
    rho = sub_solar_magnetopause * (2 / (1 + np.cos(phi))) ** alpha

    x = rho * np.cos(phi)
    y = rho * np.sin(phi)

    return list(zip(x, y))


def Get_Bow_Shock():
    initial_x = 0.5
    psi = 1.04
    p = 2.75
    L = psi * p

    phi = np.linspace(0, 2 * np.pi, 1000)
    rho = L / (1 + psi * np.cos(phi))

    x = initial_x + rho * np.cos(phi)
    y = rho * np.sin(phi)

    points = list(zip(x, y))

    correct_points = []
    for point in points:
        if point[0] > 50:
            continue

        else:
            correct_points.append(point)

    return correct_points


if __name__ == "__main__":
    main()
