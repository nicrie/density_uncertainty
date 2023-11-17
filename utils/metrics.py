import math
import pandas as pd


def centroid_on_sphere(lons, lats):
    x = 0.0
    y = 0.0
    z = 0.0

    for lon, lat in zip(lons, lats):
        latitude = math.radians(lat)
        longitude = math.radians(lon)

        x += math.cos(latitude) * math.cos(longitude)
        y += math.cos(latitude) * math.sin(longitude)
        z += math.sin(latitude)

    total = len(lons)

    x = x / total
    y = y / total
    z = z / total

    central_longitude = math.atan2(y, x)
    central_square_root = math.sqrt(x * x + y * y)
    central_latitude = math.atan2(z, central_square_root)

    return pd.Series([math.degrees(central_longitude), math.degrees(central_latitude)])
