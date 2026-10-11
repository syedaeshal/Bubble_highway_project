"""
energy.py - ENERGY ENGINEER 3, JOB A: electricity used and saved  [STAND-IN until EE3's version]
================================================================================================
1. Pressure the compressor must reach: atmosphere + water above the pipe (+ hose loss = 0).
2. Power for one pipe: adiabatic compressor formula, efficiency 60 % (bubble_physics.pipe_power_w).
3. Check against the guide: 5 m deep, 20 m wide -> ~9 kW per pipe, ~450 kW for 51 pipes.
4. Energy over a bloom: for every controller decision, power per pipe x pipes firing
   x time until the next decision. (We use the AVERAGE number of pipes that fired
   during each 5 s interval, recorded by the joining program, because wave fronts
   move between decisions.)
5. Ordinary curtain: same air per pipe, ALL pipes, for the same time the controller was on.
6. Saving %, and the headline: kWh per 1,000 real jellyfish moved out to sea.
7. Never energy on its own: always reported next to the jellyfish moved.
"""
import bubble_physics as bp


def check_against_guide(s):
    n = len(bp.pipe_positions(s))
    p = bp.pipe_power_w(s["air_nominal_lpm_per_m"], s)
    return {"pressure_bar_gauge": (bp.pipe_pressure_pa(s) - s["p_atm_pa"]) / 1e5,
            "kw_per_pipe": p / 1000, "pipes": n, "kw_all_pipes": n * p / 1000}


def summarise(decisions, s, moved_out_real):
    n_all = len(bp.pipe_positions(s))
    dt = s["controller_dt_s"]
    used = curtain = on_time = 0.0
    for d in decisions:
        if not d["on"]:
            continue
        p = bp.pipe_power_w(d["air_lpm_per_m"], s)
        used += p * d.get("n_pipes_avg", d["n_pipes_now"]) * dt
        curtain += p * n_all * dt
        on_time += dt
    kwh, kwh_c = used / 3.6e6, curtain / 3.6e6
    out = {"energy_kwh": kwh, "curtain_equivalent_kwh": kwh_c,
           "saving_vs_curtain": 1 - kwh / kwh_c if kwh_c else None,
           "on_time_s": on_time, "moved_out_real": moved_out_real,
           "kwh_per_1000_moved": 1000 * kwh / moved_out_real if moved_out_real else None}
    if s["electricity_price_aed_per_kwh"]:
        out["cost_aed"] = kwh * s["electricity_price_aed_per_kwh"]
    return out
