from rolland import DiscrPad, Sleeper, Ballast
from rolland.database.rail.db_rail import UIC60
from rolland import SimplePeriodicBallastedSingleRailTrack
from rolland import (
      PMLRailDampVertic,
      GaussianImpulse,
      DiscretizationEBBVerticConst,
      DeflectionEBBVertic
  )
from rolland.postprocessing import Response as resp, TDR, PostProcessing
import BEM_draft_v2 as BEM
from matplotlib import pyplot as plt
import numpy as np

track = SimplePeriodicBallastedSingleRailTrack(
    rail=UIC60,  # Standard UIC60 rail profile
    pad=DiscrPad(
        sp=[180e6, 0],  # Stiffness properties [N/m]
        dp=[18000, 0]  # Damping properties [Ns/m]
    ),
    sleeper=Sleeper(ms=150),  # Sleeper mass [kg]
    ballast=Ballast(
        sb=[105e6, 0],  # Ballast stiffness [N/m]
        db=[48000, 0]  # Ballast damping [Ns/m]
    ),
    num_mount=243,  # Number of discrete mounting positions
    distance=0.6  # Distance between sleepers [m]
)

# # 1. TRACK DEFINITION ----------------------------------------------------------
# # Create a ballasted single rail track model with periodic supports
track = SimplePeriodicBallastedSingleRailTrack(
    rail=UIC60,  # Standard UIC60 rail profile
    pad=DiscrPad(
        sp=[180e6, 0],  # Stiffness properties [N/m]
        dp=[18000, 0]  # Damping properties [Ns/m]
    ),
    sleeper=Sleeper(ms=150),  # Sleeper mass [kg]
    ballast=Ballast(
        sb=[105e6, 0],  # Ballast stiffness [N/m]
        db=[48000, 0]  # Ballast damping [Ns/m]
    ),
    num_mount=243,  # Number of discrete mounting positions
    distance=0.6  # Distance between sleepers [m]
)

# 2. SIMULATION SETUP ---------------------------------------------------------
# Define boundary conditions (Perfectly Matched Layer absorbing boundary)
boundary = PMLRailDampVertic(l_bound=33.0)  # 33.0 m boundary domain

# Define excitation (Gaussian impulse between sleepers at 71.7m)
excitation = GaussianImpulse(x_excit=71.7)

# 3. DISCRETIZATION & SIMULATION ----------------------------------------------
# Set up numerical discretization parameters
discretization = DiscretizationEBBVerticConst(
    track=track,
    bound=boundary,
)

# Run the simulation and calculate deflection over time
deflection_results = DeflectionEBBVertic(
    discr=discretization,
    excit=excitation
)

# 4. POSTPROCESSING & VISUALIZATION -------------------------------------------


response = resp(results=deflection_results) #now response is a DeflectionEBBVert object.

#get the small variables that we will need to when computing
dt = response.results.discr.dt
nt = response.results.discr.nt
frequencies = response.freq
n_freq = frequencies.shape[0]

#Get deflection result
D = response.results.deflection     # shape: (2*nx, nt+1)


# the shape has all deflections from rail and the pad
nx = D.shape[0] // 2
rail_defl = D[0::2, :nt] #not taking the last because they are the dts

#computing the deflection in frequency domain
#for that there is a numpy function
U_all = np.fft.rfft(rail_defl, axis=1)     # shape: (nx, nt//2+1)
U_all *= 2.0 / nt                           # same normalization as fast_fourier_tr
# I dont understand this but appearently this is done because
#we want to get the amplidtues but numpy does not divide with N


# 5) build the full frequency axis and then mask it to [f_min, f_max]
full_freqs = np.fft.rfftfreq(nt, dt)             # shape: (nt//2+1,)
mask = (full_freqs > response.f_min) & (full_freqs <= response.f_max)
freqs = full_freqs[mask]                    # shape: (n_freq,) == response.freq
#I dont know how they have the same shape but they do

U_masked = U_all[:, mask]                      # shape: (nx, n_freq)


# 6) convert displacement → velocity: v_fft[i,k] = j·2πf_k · U_masked[i,k]
omega = 2 * np.pi * freqs                  # shape: (n_freq,)
# broadcast (1×n_freq) × (nx×n_freq) → (nx×n_freq)
rail_velFFT = (1j * omega)[None, :] * U_masked    # shape: (nx, n_freq)


# 7) compute SPL by superposing each node’s contribution
spl = np.zeros(n_freq)
spl_mono = np.zeros(n_freq)                    # shape: (n_freq,)
spl_di = np.zeros(n_freq)                    # shape: (n_freq,)

p_ref = 20e-6                               # reference pressure [Pa]

for k, f in enumerate(freqs):
    p_total = 0+0j
    p_total_mono = 0 + 0j
    p_total_di = 0 + 0j
    for i in range(nx):
        v = rail_velFFT[i, k]# scalar complex velocity
        monopole = BEM.Radiator([0,0,0], v, 1e-4)
        monopole.calculate_k(f)
        p = monopole.radiate_frequency_monopole(v, [3,4,0], f) + monopole.radiate_frequency_dipole(v, [3,4,0], f)
        p_mono = monopole.radiate_frequency_monopole(v, [3,4,0], f)
        p_di = monopole.radiate_frequency_dipole(v, [3,4,0], f)

        p_total += p
        p_total_mono += p_mono
        p_total_di += p_di

        """
        INSTEAD OF DOING A FOR LOOP CAN I DO A MATRIX EQUATION TO 
        HAVE FASTER CALCULATION?
        """

    spl[k] = 20 * np.log10(abs(p_total) / p_ref)
    spl_mono[k] = 20 * np.log10(abs(p_total_mono) / p_ref)
    spl_di[k] = 20 * np.log10(abs(p_total_di) / p_ref)


# 8) plot
plt.figure(figsize=(8,5))
plt.semilogx(freqs, spl)
plt.xlabel("Frequency [Hz]")
plt.ylabel("SPL [dB re 20 µPa]")
plt.title("Radiated SPL Spectrum Total")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()
plt.figure(figsize=(8,5))
plt.semilogx(freqs, spl_mono)
plt.xlabel("Frequency [Hz]")
plt.ylabel("SPL [dB re 20 µPa]")
plt.title("Radiated SPL Spectrum only monopole")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()
plt.figure(figsize=(8,5))
plt.semilogx(freqs, spl_di)
plt.xlabel("Frequency [Hz]")
plt.ylabel("SPL [dB re 20 µPa]")
plt.title("Radiated SPL Spectrum only dipole")
plt.grid(True, which="both", ls="--", lw=0.5)
plt.show()

