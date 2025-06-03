import unittest
import BEM
import numpy as np

class TestBEM(unittest.TestCase):

    def test_tangential_length(self):
        head_width = 10
        head_height = 10
        web_width = 3
        web_height = 15
        foot_width = 20
        foot_height = 5
        dummy = BEM.create_dummy_profile(head_width, head_height, web_width, web_height, foot_width, foot_height)
        expected = (head_width + foot_width - web_width) * 2
        dummy_profile = BEM.Crosssection(dummy)
        self.assertAlmostEqual(dummy_profile.calculate_tangential_length(), expected, 3)

    def test_calculate_nominal_speed_cross_section(self):
        tailored_frequency_mobility_range = [
            1.2e-6,
            3.5e-6,
            7.2e-6,
            2.8e-5,
            4.5e-5,
            1.1e-4,
            3.5e-4
        ]
        tailored_frequency_mobility_range = np.array(tailored_frequency_mobility_range)
        head_width = 10
        head_height = 10
        web_width = 3
        web_height = 15
        foot_width = 20
        foot_height = 5
        dummy = BEM.create_dummy_profile(head_width, head_height,
                                         web_width, web_height,
                                         foot_width, foot_height)
        dummy_profile = BEM.Crosssection(dummy)
        tangential_length = dummy_profile.calculate_tangential_length()
        nominal_speed = dummy_profile.calculate_nominal_cross_section_speed(
            tailored_frequency_mobility_range
        )
        expected = np.array([6.48e-05, 0.000189, 0.0003888, 0.001512, 0.00243, 0.00594, 0.0189])
        for nom_spd1, nom_spd2 in zip(nominal_speed, expected):
            self.assertAlmostEqual(nom_spd1, nom_spd2, 3)
