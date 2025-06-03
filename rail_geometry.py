import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from rolland.database.rail.db_rail import UIC60


def load_rail_geo(file_path):
    """Load rail geometry from pts file."""
    with open(file_path) as file:
        return [(float(parts[0]), float(parts[2])) for line in file if
                (parts := line.split()) and len(parts) == 3 and not line.startswith(':!') and line.strip()]

def mirror_at_z_axis(geometry):
    """Mirror the rail geometry at the z-axis."""
    return [(y, -z) for y, z in geometry]

def redefine_reference_point(geometry, new_reference):
    """Shift the rail geometry to a new reference point."""
    y_ref, z_ref = new_reference
    return [(y - y_ref, z - z_ref) for y, z in geometry]



# Plotting the rail geometry
def plot_rail_geometry(geometry):
    y, z = zip(*geometry)
    plt.figure(figsize=(10, 10))
    plt.plot(y, z, marker="x", linestyle="-", color="b")
    plt.title('Rail Geometry')
    plt.xlabel('Y [m]')
    plt.ylabel('Z [m]')
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    rl_geo = load_rail_geo(os.path.join(os.path.dirname(__file__), 'UIC60'))
    #plot_rail_geometry(rl_geo)

    rl_mirror = mirror_at_z_axis(rl_geo)

    rl_geo_new = redefine_reference_point(rl_mirror, (-80.7, (172-90.1-34.8)))
    #plot_rail_geometry(rl_geo_new)




    #############################################

    # Input: Original contour points
    contour = np.array(rl_geo_new)  # Ensure rl_geo_new is a NumPy array
    y = contour[:, 0] * 10 **-3
    z = contour[:, 1] * 10 **-3

    # Compute the cumulative distance along the contour
    dy = np.diff(y)
    dz = np.diff(z)
    distances = np.sqrt(dy**2 + dz**2)
    cumulative_distances = np.zeros(len(y))
    cumulative_distances[1:] = np.cumsum(distances)

    # Normalize distances to [0, 1]
    normalized_distances = cumulative_distances / cumulative_distances[-1]

    # Create interpolation functions for y and z
    fy = interp1d(normalized_distances, y, kind='linear', assume_sorted=True)
    fz = interp1d(normalized_distances, z, kind='linear', assume_sorted=True)

    # Define the number of new points
    num_new_points = 100  # Change this to your desired number of points


    # Generate new equally spaced distances
    new_normalized_distances = np.linspace(0, 1, num_new_points)

    # Interpolate new points
    new_y = fy(new_normalized_distances)
    new_z = fz(new_normalized_distances)

    # Combine into new contour
    new_contour = np.column_stack((new_y, new_z))

    # Plot the new contour
    plot_rail_geometry(new_contour)


    #import csv
    #with open('../../rolland/rolland/database/rail/UIC60.csv', 'w', newline='') as csvfile:
    #    csvwriter = csv.writer(csvfile)
    #    csvwriter.writerow(['Y', 'Z'])
    #    csvwriter.writerows(new_contour)