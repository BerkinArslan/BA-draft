import numpy as np
import matplotlib.pyplot as plt

class RailDiscr():
    def __init__(self,
                 rail_contour_point_x: np.ndarray,
                 observation_point: np.ndarray,
                 deflections: np.ndarray):
        """
        :param rail_contour_point: coordinates of the points in 3D room
        :param observation_point: coordinate of the observation point
        :param deflections: deflection of each x coordinate
        """
        self.rail_contour_point_x = rail_contour_point_x #shape (nx, )
        self.observation_point = observation_point #shape (1, 3)
        self.deflections = deflections #shape(nx, n_freqs)

    def generate_radiators(self):
        pass

class Radiators():
    #for efficient computation I will create all of the radiators in one go in a matrix
    def __init__(self,
                 coordinates: np.ndarray,
                 v_normal: np.ndarray,
                 area: float,
                 rail: RailDiscr = None,
                 observation_point: np.ndarray = None,
                 ):
        self.rail = rail #RailDiscr element
        self.v_normal = v_normal #shape of (nx, n_freqs)
        self.coordinates = coordinates #shape of (n_cross, 2)
        self.area = area #int
        self.observation_point = observation_point


    def callculate_k(self, freqs: np.ndarray, c:int = 343):
        """
        :param freqs: array of shape (n_freqs, )
        :param c: speed of sound travel
        :return: wave number k (in the shape of (n_freqs, ))
        """
        return 2 * np.pi * freqs / c #array of (n_freq, )

    def radiate_monopole(self,
                         freqs: np.ndarray,  # this as well
                         v_normal: np.ndarray = None, #this as well
                         observation_point: np.ndarray = None, #this could be defined as attr as well
                         coords: np.ndarray = None,
                         area: float = None,
                         rho:float = 1.225) -> np.ndarray:
        if area is None:
            if self.area is not None:
                area = self.area
            else:
                raise ValueError('Please input a valid area')

        if coords is None:
            if self.coordinates is not None:
                coords = self.coordinates
            else:
                raise ValueError('Please input the coordinates')

        if v_normal is None:
            if self.v_normal is not None:
                v_normal = self.v_normal
            else:
                raise ValueError('Please input the v_normal')

        if observation_point is None:
            if self.observation_point is not None:
                observation_point = self.observation_point
            else:
                raise ValueError('Please input the observation_point')

        k = self.callculate_k(freqs) #(n_freqs)
        r = np.linalg.norm(coords - observation_point, axis=1)
        r = r.reshape(-1, 1)
        k = k.reshape(1, -1)
        #freqs (n_freqs, ), v_normal (nx, n_freqs), k (n_freqs,)
        p = ( (-1j * 2 * np.pi * freqs * v_normal * area * rho) * np.exp(-1j * k * r)/(4 * np.pi * r) )
        return p

    # def radiate_monopole(self,
    #                      freqs: np.ndarray,
    #                      v_normal: np.ndarray = None,
    #                      observation_point: np.ndarray = None,
    #                      coords: np.ndarray = None,
    #                      area: float = None,
    #                      rho: float = 1.225) -> np.ndarray:
    #
    #     if area is None:
    #         if self.area is not None:
    #             area = self.area
    #         else:
    #             raise ValueError('Please input a valid area')
    #
    #     if coords is None:
    #         if self.coordinates is not None:
    #             coords = self.coordinates
    #         else:
    #             raise ValueError('Please input the coordinates')
    #
    #     if v_normal is None:
    #         if self.v_normal is not None:
    #             v_normal = self.v_normal
    #         else:
    #             raise ValueError('Please input the v_normal')
    #
    #     if observation_point is None:
    #         if self.observation_point is not None:
    #             observation_point = self.observation_point
    #         else:
    #             raise ValueError('Please input the observation_point')
    #
    #     k = self.callculate_k(freqs)  # shape: (n_freqs,)
    #     r = np.linalg.norm(coords - observation_point, axis=1)  # shape: (nx,)
    #
    #     # Reshape to allow broadcasting: (nx, 1) * (1, n_freqs) → (nx, n_freqs)
    #     rk = r.reshape(-1, 1) * k.reshape(1, -1)  # (nx, n_freqs)
    #
    #     # omega = 2πf: shape (1, n_freqs) so it broadcasts with v_normal (nx, n_freqs)
    #     omega = 2 * np.pi * freqs.reshape(1, -1)
    #
    #     denom = 4 * np.pi * r.reshape(-1, 1)  # (nx, 1)
    #
    #     p = (-1j * omega * rho * area * v_normal) * np.exp(-1j * rk) / denom  # (nx, n_freqs)
    #
    #     return p

if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt

    # --- Test parameters ---
    nx = 10
    nt = 1024
    dt = 1e-4
    t = np.linspace(0, (nt - 1) * dt, nt)
    x_vals = np.linspace(0, 5, nx)  # (nx,)
    area = 1e-4
    obs_point = np.array([3.0, 4.0, 1.5])  # (3,)

    # --- Fake deflection signal ---
    f_signal = 200  # Hz
    rail_defl_time = np.sin(2 * np.pi * f_signal * t)[None, :] * np.hanning(nt)
    rail_defl_time = rail_defl_time * np.linspace(1, 0.5, nx)[:, None]  # shape (nx, nt)

    # --- FFT and velocity conversion ---
    U_all = np.fft.rfft(rail_defl_time, axis=1) * (2.0 / nt)  # (nx, n_freqs)
    freqs = np.fft.rfftfreq(nt, dt)
    f_min = 50
    f_max = 5000
    mask = (freqs > f_min) & (freqs <= f_max)
    freqs = freqs[mask]
    U = U_all[:, mask]
    omega = 2 * np.pi * freqs
    v_normal = 1j * omega * U  # shape (nx, n_freqs)

    # --- Coordinates (x axis) ---
    coords = np.stack([x_vals, np.zeros_like(x_vals), np.zeros_like(x_vals)], axis=1)  # (nx, 3)

    # --- Instantiate classes ---
    rail = RailDiscr(rail_contour_point_x=x_vals,
                     observation_point=obs_point,
                     deflections=None)

    radiators = Radiators(coordinates=coords,
                          v_normal=v_normal,
                          area=area,
                          rail=rail,
                          observation_point=obs_point)

    # --- Calculate SPL ---
    p = radiators.radiate_monopole(freqs)
    p_sum = np.sum(p, axis=0)
    spl = 20 * np.log10(np.abs(p_sum) / 20e-6)

    # --- Plot ---
    plt.figure(figsize=(8, 4))
    plt.plot(freqs, spl)
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("SPL [dB re 20 µPa]")
    plt.title("SPL from Simulated Monopole Line")
    plt.grid(True)
    plt.tight_layout()
    plt.show()







