import numpy as np
from matplotlib import pyplot as plt





class CrossSection():
    def __init__(self, rail_contour: np.ndarray,
                 observation_point: tuple[float, float, float],
                 mobility_freq: np.ndarray = None,):
        """
        :param rail_contour: 2D contour array in the spae of (n, 2)
        """
        self.rail_contour: np.ndarray = rail_contour
        self.tangential_length = None
        self.mobility_freq = mobility_freq
        self.observation_point = observation_point

class Radiator():
    def __init__(self, coordinates: tuple[float, float, float],
                 v_tan: float, area,
                 cross_section: CrossSection = None,
                 ):
        self.coordinates = coordinates
        self.cross_section = cross_section
        self.v_tan = v_tan
        self.area = area


    def calculate_k(self, freq: float|int, c: float|int = 343):
        return 2 * np.pi * freq/c

    def radiate_frequency_monopole(self, v_tan: float,
                observation_point: np.ndarray,
                          frequency: int|float,
                coordinates: np.ndarray = None,
                area: float | int = None,
                          rho: int|float = 1.225
                )-> float:
        """
        Calculates the sound pressure in observation point
        :param coordinates: coordinates of that monopole
        :param mobility: mobility in that frequency
        :param observation_point: the coordinates of the observation point
        :return: sound pressure level calculated in the observation point
        """
        if area is None:
            if self.area is not None:
                area = self.area
            else:
                raise ValueError('Please input a v_tan')

        if coordinates is None:
            if self.coordinates is not None:
                coordinates = self.coordinates
            else:
                raise ValueError('Please input the coordinates')
        coordinates = np.array(coordinates)
        observation_point =np.array(observation_point)
        q = area * v_tan
        k = self.calculate_k(frequency)
        r = np.linalg.norm((coordinates - observation_point))
        #p = 1j * ((frequency * rho)/(2)) * q * np.exp(-(1j * k * r))/ r
        p = ( (-1j * 2 * np.pi * frequency * rho * v_tan * area) * np.exp(-1j * k * r)/(4 * np.pi * r) )

        return p

    # def radiate_frequency_dipole(self, v_tan: float,
    #                              observation_point: np.ndarray,
    #                              frequency: int|float,
    #                              p_acoustic: int|float = 0,
    #                              coordinates: np.ndarray = None,
    #                              area: float|int = None,
    #                              ro: int|float = 1.225):
    #     if area is None:
    #         if self.area is not None:
    #             area = self.area
    #         else:
    #             raise ValueError('Please input a v_tan')
    #
    #     if coordinates is None:
    #         if self.coordinates is not None:
    #             coordinates = self.coordinates
    #         else:
    #             raise ValueError('Please input the coordinates')
    #     coordinates = np.array(coordinates)
    #     observation_point = np.array(observation_point)
    #     k = self.calculate_k(frequency)
    #     r = np.linalg.norm((coordinates - observation_point))
    #     d_green = ((1j* k * r) * np.exp(-1j* r * k))/ 4 * np.pi *r
    #     cos_theta = np.dot(coordinates, observation_point) / (np.linalg.norm(coordinates) * np.linalg.norm(observation_point))
    #     p = (p_acoustic * d_green * area * cos_theta)
    #     return p








if __name__ == '__main__':
    radiations = []
    monopole = Radiator([1, 0, 0], 5, 0.05)
    freqs = np.logspace(np.log10(100), np.log10(18000), num=300)
    freqs = np.linspace(100, 18000, num=5000)
    for freq in freqs:
        radiation = monopole.radiate_frequency_monopole(5, (7, 0, 0), freq, area = 0.05)
        radiations.append(radiation)
    plt.plot(freqs, np.abs(radiations))
    plt.show()
    # radiations = []
    # for freq in freqs:
    #     radiation = monopole.radiate_frequency_dipole(5, (7, 0, 0), freq, area = 0.05)
    #     radiations.append(radiation)
    # plt.plot(freqs, np.abs(radiations))
    # plt.show()








