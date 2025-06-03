import unittest
import numpy as np
import rail_geometry as rg
import os
from scipy.interpolate import interp1d

class Crosssection():

        def __init__(self, rail_contour: np.ndarray, mobility_freq: np.ndarray = None):
            """
            :param rail_contour: 2D contour array in the spae of (n, 2)
            """
            self.rail_contour: np.ndarray = rail_contour
            self.tangential_length = None
            self.mobility_freq = mobility_freq

        def calculate_tangential_length(self, rail_contour: np.ndarray = None):
            if rail_contour is None:
                if self.rail_contour is None:
                    raise ValueError('Please provide the method with a rail contour')
                rail_contour = self.rail_contour
            #rail contour is a 2D array in the shape of (n, 2). n = num of discrpoints
            y_coordinates = rail_contour[:, 0]
            sum = 0
            for i in range(1, y_coordinates.shape[0]):
                sum = sum +  abs(y_coordinates[i] - y_coordinates[i - 1])

            tangential_length = sum
            self.tangential_length = tangential_length
            return tangential_length

        def calculate_nominal_cross_section_speed(self, frequency_range_mobility: np.ndarray = None,
                                                  tan_length: float = None):
            """
            :param frequency_range_mobility: mobility for every frequency band
            numpy array in shape of (n, ) or (n, 1)
            :param tan_length: tangential length of crosssection
            :return: nominal speed of the crossection
            """
            if frequency_range_mobility is None:
                if self.mobility_freq is None:
                    raise ValueError('please put in a mobility array for the frequency range')
                frequency_range_mobility = self.mobility_freq
            if tan_length is None:
                if self.tangential_length is None:
                    raise ValueError('Please put in a tangential length')
                tan_length = self.tangential_length
            nominal_speed_crosssection_frequency_range = frequency_range_mobility * tan_length
            return nominal_speed_crosssection_frequency_range


def normalize_and_interpolate(contour: np.ndarray,
                              num_discr_points: int = 100) -> np.ndarray:
    dy = np.diff(contour[:, 0])
    dz = np.diff(contour[:, 1])
    distances = np.sqrt(dy ** 2 + dz ** 2)
    cumulative_distances = np.zeros(len(dy) + 1)
    cumulative_distances[1:] = np.cumsum(distances)
    normalized_distances = cumulative_distances / cumulative_distances[-1]
    fy = interp1d(normalized_distances, contour[:, 0], kind='linear', assume_sorted=True)
    fz = interp1d(normalized_distances, contour[:, 1], kind='linear', assume_sorted=True)
    normalized_distances_in_range = np.linspace(0, 1, num_discr_points)
    y_in_range = fy(normalized_distances_in_range)
    z_in_range = fz(normalized_distances_in_range)
    new_contour = np.column_stack((y_in_range, z_in_range))

    return new_contour

def create_dummy_profile(head_width: int, head_height: int,
                         web_width: int, web_height: int,
                         foot_width: int, foot_height: int):
    head_x_max = head_width / 2
    head_y_max = foot_height + web_height + head_height
    web_x_max = web_width / 2
    web_y_max = foot_height + web_height
    foot_x_max = foot_width / 2
    foot_y_max = foot_height
    x = [0, foot_x_max, foot_x_max, web_x_max, web_x_max,
                              head_x_max, head_x_max]
    x_rev = [-val for val in reversed(x)]
    x += x_rev
    y = [0, 0, foot_y_max, foot_y_max, web_y_max,
                               web_y_max, head_y_max]
    y_rev = list(reversed(y))
    y += y_rev
    dummy_contour = np.array([x, y]) #* 1e-3
    dummy_contour = np.transpose(dummy_contour)
    #print(dummy_contour)
    new_dummy_contour = normalize_and_interpolate(dummy_contour)
    rg.plot_rail_geometry(new_dummy_contour)
    return new_dummy_contour


if __name__ == '__main__':
    rl_geo = rg.load_rail_geo(os.path.join(os.path.dirname(__file__), 'UIC60'))
    rl_mirror = rg.mirror_at_z_axis(rl_geo)
    rl_geo = rg.redefine_reference_point(rl_mirror, (-80.7, (172-90.1-34.8)))
    contour = np.array(rl_geo) * 10 ** -3
    new_contour = normalize_and_interpolate(contour)
    rg.plot_rail_geometry(new_contour)

    #create_dummy_profile(10, 10, 3, 15, 20, 5)



    #print(contour)

    head_width = 10
    head_height = 10
    web_width = 3
    web_height = 15
    foot_width = 20
    foot_height = 5
    dummy = create_dummy_profile(head_width, head_height, web_width, web_height, foot_width, foot_height)
    expected = (head_width + foot_width - web_width) * 2
    dummy_profile = Crosssection(dummy)
    print(f"expected: {expected}, calculated: {dummy_profile.calculate_tangential_length()}")
