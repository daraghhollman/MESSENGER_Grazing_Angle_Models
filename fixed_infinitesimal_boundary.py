"""
In this script, we look at the distribution of grazing angles as would've been observed for crossings only occuring along the Winslow+ 2013 boundary models.

"""

import datetime as dt
import numpy as np
import shapely
import shapely.plotting
import matplotlib.pyplot as plt
from hermpy import trajectory, utils


def main():

    time_steps, positions = Get_Times_And_Positions(resolution=3600)

    # We define the boundaries as polygons (closing them at the end)
    magnetopause_points = Get_Magnetopause()
    magnetopause_region = shapely.Polygon(magnetopause_points)

    bow_shock_points = Get_Bow_Shock()
    bow_shock_region = shapely.Polygon(bow_shock_points)

    # Now we iterrate through the positions and determine which region the spacecraft is in.
    # If the point is in neither of the regions, we are in the solar wind.
    # If the point is in the bow shock region but not the magnetopause region, we are in the magnetosheath.
    # if the point is in both the bow shock region and the magnetopause region, we are in the magnetosphere.
    # When the region changes, we mark a crossing and determine the grazing angle.
    for position in positions:

        region = Check_Region(position, bow_shock_region, magnetopause_region)

        print(region)


def Get_Times_And_Positions(resolution: int = 600):
    # Define the start and end time, along with a resolution
    mission_start = dt.datetime(2011, 3, 23, 15, 37)
    mission_end = dt.datetime(2015, 4, 30, 15, 8)

    mission_duration = int((mission_end - mission_start).total_seconds())

    time_steps = np.array(
        [
            mission_start + dt.timedelta(seconds=int(t))
            for t in np.arange(0, mission_duration, resolution)
        ]
    )

    print("Time steps calculated")

    positions = trajectory.Get_Position(
        "MESSENGER", time_steps, frame="MSM", aberrate=False
    )
    positions /= utils.Constants.MERCURY_RADIUS_KM

    print("Found positions")

    return time_steps, positions


def Check_Region(position, bow_shock_region, magnetopause_region):
    x, y, z = position

    # First check if within bow shock
    if bow_shock_region.contains(shapely.Point(x, y)) and bow_shock_region.contains(
        shapely.Point(x, z)
    ):
        within_bow_shock = True
    else:
        within_bow_shock = False

    # The check if within magnetopause
    if magnetopause_region.contains(
        shapely.Point(x, y)
    ) and magnetopause_region.contains(shapely.Point(x, z)):
        within_magnetopause = True
    else:
        within_magnetopause = False

    # Finally, confirm which region we are in

    if within_bow_shock and within_magnetopause:
        region = "magnetosphere"

    elif within_bow_shock and not within_magnetopause:
        region = "magnetosheath"

    elif not within_bow_shock and not within_magnetopause:
        region = "solar wind"

    else:
        raise ValueError(f"Can't determine spacecraft region for position {position}")

    return region


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
