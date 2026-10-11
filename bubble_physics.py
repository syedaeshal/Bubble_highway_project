"""
bubble_physics.py - formulas SHARED by Energy Engineers 1, 2 and 3
====================================================================
The guide insists the water physics (EE1) and the controller (EE2) use the
same water-speed formula "so you never disagree", and the energy (EE3) must use
the same air numbers. So all three import these functions.

1. surface_speed(air)   - Bulson (1961): Vmax = C (g q)^(1/3)
2. push_shape(dx)       - how the push from one pipe varies along the corridor
3. depth_factor(depth)  - the push only reaches the top H/4
4. background_current() - the tide (EE1 Job B step 6); also read by the current meter
5. pipe_power_w(air)    - compressor power per pipe (guide formula, EE3 Job A)

Labelled as THEORY, not measurement: no measured plume speed exists for our
design (the team decided to use bubble-plume theory).
"""
import numpy as np


def air_si(air_lpm_per_m):
    """L/min per metre of pipe -> m^3/s per metre (free air)."""
    return np.asarray(air_lpm_per_m, float) / 1000.0 / 60.0


def surface_speed(air_lpm_per_m, s):
    """Fastest horizontal water speed a firing pipe makes at the surface.

    Bulson (1961), large-scale tests: Vmax ~ 1.46 (g q)^(1/3), q = air per metre.
    Cube root: doubling the air raises the speed by 2^(1/3) = 1.26, i.e. about a
    quarter - exactly the rule in the guide (EE1 Job B step 4)."""
    return s["plume_constant"] * (s["gravity"] * air_si(air_lpm_per_m)) ** (1.0 / 3.0)


def push_shape(dx, s):
    """Fraction (-1..1) of the full push felt at distance dx from a pipe
    (dx > 0 = on the sea side of the pipe).

    * Directly above the pipe the water rises, so the horizontal push is 0.
    * It is strongest about d/2 from the pipe (Ince's design guide), d = pipe depth.
    * Beyond that it fades away (assumed exponential).
    Shape used: f(r) = (r / r0) * exp(1 - r / r0), r0 = d/2, which is 0 at the
    pipe, exactly 1 at r0, and fades smoothly after.
    Optimistic version: only the sea side pushes. Realistic: both sides push away."""
    r0 = s["push_peak_distance_fraction"] * s["pipe_depth_m"]
    r = np.abs(dx)
    f = (r / r0) * np.exp(1.0 - r / r0)
    if s["water_version"] == "optimistic":
        return np.where(dx > 0, f, 0.0)
    return np.sign(dx) * f


def depth_factor(depth, s):
    """Bulson: the surface current is H/4 deep and its speed falls linearly from
    the surface to zero at that depth. With 5 m pipes: felt only in the top 1.25 m."""
    h_out = s["outflow_depth_fraction"] * s["pipe_depth_m"]
    return np.clip(1.0 - np.asarray(depth, float) / h_out, 0.0, 1.0)


def background_current(t, s):
    """Background current along the corridor (m/s, + = toward the sea).
    Optionally turns like a tide: a smooth cosine change over current_ramp_s."""
    u0 = s["current_mps"]
    t_rev = s["current_reverse_at_s"]
    if t_rev is None:
        return u0
    u1 = s["current_after_reverse_mps"]
    a = np.clip((t - (t_rev - s["current_ramp_s"] / 2)) / s["current_ramp_s"], 0.0, 1.0)
    return u0 + (u1 - u0) * 0.5 * (1 - np.cos(np.pi * a))


def pipe_positions(s):
    """Pipes across the corridor every pipe_spacing_m from entrance to exit."""
    n = int(round(s["corridor_length_m"] / s["pipe_spacing_m"])) + 1
    return np.linspace(0.0, s["corridor_length_m"], n)


def pipe_pressure_pa(s):
    """EE3 Job A step 1: absolute pressure the compressor must reach =
    atmosphere + weight of water above the pipe + hose/nozzle loss."""
    return s["p_atm_pa"] + s["water_density"] * s["gravity"] * s["pipe_depth_m"] + s["hose_loss_pa"]


def pipe_power_w(air_lpm_per_m, s):
    """EE3 Job A step 2: electrical power for ONE pipe (adiabatic compression):
        P = Q p1 (k/(k-1)) [ (p2/p1)^((k-1)/k) - 1 ] / efficiency
    Q = free air for the whole pipe = air per metre x corridor width.
    Check (guide): 5 m deep, 20 m wide, 374 L/min/m -> about 9 kW."""
    k, p1 = s["gamma_air"], s["p_atm_pa"]
    Q = air_si(air_lpm_per_m) * s["corridor_width_m"]
    ratio = pipe_pressure_pa(s) / p1
    return Q * p1 * k / (k - 1) * (ratio ** ((k - 1) / k) - 1) / s["compressor_efficiency"]
