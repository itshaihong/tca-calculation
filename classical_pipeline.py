import os

import pandas as pd
import numpy as np
import time
from scipy.optimize import least_squares, minimize_scalar, root_scalar
import orekit
from orekit.pyhelpers import download_orekit_data_curdir, setup_orekit_curdir


vm = orekit.initVM()
if not os.path.exists("orekit-data.zip"):
    download_orekit_data_curdir()
setup_orekit_curdir("orekit-data.zip")

from org.orekit.orbits import CartesianOrbit, OrbitType
from org.orekit.utils import PVCoordinates, Constants, IERSConventions
from org.orekit.frames import FramesFactory, TopocentricFrame
from org.orekit.time import AbsoluteDate, TimeScalesFactory
from org.orekit.propagation.numerical import NumericalPropagator
from org.orekit.propagation import SpacecraftState
from org.hipparchus.ode.nonstiff import DormandPrince853Integrator
from org.orekit.forces.gravity.potential import GravityFieldFactory
from org.orekit.forces.gravity import HolmesFeatherstoneAttractionModel
from org.orekit.forces.drag import DragForce, IsotropicDrag
from org.orekit.models.earth.atmosphere import HarrisPriester
from org.orekit.bodies import OneAxisEllipsoid, CelestialBodyFactory, GeodeticPoint
from org.hipparchus.geometry.euclidean.threed import Vector3D

EARTH_MU = Constants.WGS84_EARTH_MU
EARTH_RADIUS = Constants.WGS84_EARTH_EQUATORIAL_RADIUS

def build_simplified_propagator(state_vector, epoch, inertial_frame):
    """
    THE FLAWED MODEL: 10x10 Gravity + Fixed Harris-Priester Drag.
    This simulates an operational, but imperfect, numerical propagator.
    """
    pos = Vector3D(float(state_vector[0]), float(state_vector[1]), float(state_vector[2]))
    vel = Vector3D(float(state_vector[3]), float(state_vector[4]), float(state_vector[5]))
    initial_orbit = CartesianOrbit(PVCoordinates(pos, vel), inertial_frame, epoch, EARTH_MU)
    
    min_step, max_step, pos_tol = 0.001, 300.0, 0.1
    tolerances = NumericalPropagator.tolerances(pos_tol, initial_orbit, OrbitType.CARTESIAN)
    integrator = DormandPrince853Integrator(min_step, max_step, 
                                            orekit.JArray_double.cast_(tolerances[0]), 
                                            orekit.JArray_double.cast_(tolerances[1]))
    
    propagator = NumericalPropagator(integrator)
    propagator.setOrbitType(OrbitType.CARTESIAN)
    
    # 1. Add 10x10 Gravity
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    gravity_provider = GravityFieldFactory.getNormalizedProvider(10, 10)
    propagator.addForceModel(HolmesFeatherstoneAttractionModel(itrf, gravity_provider))
    
    # 2. Add Fixed Drag (Assuming 100kg mass, 1m^2 cross-section, Cd=2.2)
    earth = OneAxisEllipsoid(EARTH_RADIUS, Constants.WGS84_EARTH_FLATTENING, itrf)
    sun = CelestialBodyFactory.getSun()
    atmosphere = HarrisPriester(sun, earth)
    isotropic_drag = IsotropicDrag(1.0, 2.2)
    propagator.addForceModel(DragForce(atmosphere, isotropic_drag))
    
    spacecraft_state = SpacecraftState(initial_orbit, 100.0)
    propagator.setInitialState(spacecraft_state)
    
    return propagator

def compute_residuals(state_guess, elapsed_times, observed_ranges, station_frame, guess_epoch, sim_start_epoch, inertial_frame):
    """
    Cost Function for Batch Least Squares OD.
    Dynamically anchors the guess to the start of the radar tracklet.
    """
    try:
        propagator = build_simplified_propagator(state_guess, guess_epoch, inertial_frame)
        residuals = []
        
        # Propagate through the measurement arc
        for i, t_elapsed in enumerate(elapsed_times):
            # Calculate absolute date from the simulation start (t=0)
            target_date = sim_start_epoch.shiftedBy(float(t_elapsed))
            
            pred_state = propagator.propagate(target_date)
            pred_pos = pred_state.getPVCoordinates().getPosition()
            computed_range = station_frame.getRange(pred_pos, inertial_frame, target_date)
            residuals.append(observed_ranges[i] - computed_range)
            
        return np.array(residuals)
        
    except orekit.JavaError:
        # Penalty wall if the optimizer throws the orbit into the atmosphere
        return np.full(len(elapsed_times), 1e9)
            

def distance_at_time(t_offset, deb_initial_state, obs_state_5400, start_epoch, collision_epoch, inertial_frame):
    """
    Calculates the 3D distance between Debris and Observer at a specific time.
    Handles dual-epoch anchoring: Debris at t=0, Observer at t=5400.
    """
    # Debris anchored at t=0 (start_epoch)
    deb_propagator = build_simplified_propagator(deb_initial_state, start_epoch, inertial_frame)
    
    # Observer anchored at t=5400 (collision_epoch)
    obs_propagator = build_simplified_propagator(obs_state_5400, collision_epoch, inertial_frame) 
    
    # Target date is calculated relative to the start of the simulation
    target_date = start_epoch.shiftedBy(float(t_offset))
    
    # Propagate both to the exact same clock time
    deb_pos = deb_propagator.propagate(target_date).getPVCoordinates().getPosition()
    obs_pos = obs_propagator.propagate(target_date).getPVCoordinates().getPosition()
    
    return Vector3D.distance(deb_pos, obs_pos)


def orthogonality_condition(t_offset, deb_initial_state, obs_state, deb_epoch, obs_epoch, simulation_start_epoch, inertial_frame):
    """
    Evaluates the orthogonality condition. 
    t_offset is the absolute time in seconds since the simulation started.
    """
    # Initialize propagators anchored to their specific dynamic timestamps
    deb_propagator = build_simplified_propagator(deb_initial_state, deb_epoch, inertial_frame)
    obs_propagator = build_simplified_propagator(obs_state, obs_epoch, inertial_frame) 
    
    # Target date is calculated relative to the simulation start time (t=0)
    target_date = simulation_start_epoch.shiftedBy(float(t_offset))
    
    # Propagate
    r_deb, v_deb = deb_propagator.propagate(target_date).getPVCoordinates().getPosition(), deb_propagator.propagate(target_date).getPVCoordinates().getVelocity()
    r_obs, v_obs = obs_propagator.propagate(target_date).getPVCoordinates().getPosition(), obs_propagator.propagate(target_date).getPVCoordinates().getVelocity()
    
    r_rel = r_deb.subtract(r_obs)
    v_rel = v_deb.subtract(v_obs)
    
    return Vector3D.dotProduct(r_rel, v_rel)

def calculate_precise_tca(deb_initial_state, obs_state, deb_epoch, obs_epoch, simulation_start_epoch, inertial_frame):
    # Same as before, but pass the new dynamic epoch arguments to root_scalar
    try:
        res = root_scalar(
            orthogonality_condition, 
            args=(deb_initial_state, obs_state, deb_epoch, obs_epoch, simulation_start_epoch, inertial_frame),
            bracket=[5200.0, 5600.0], # Search near the 5400s mark
            method='brentq', 
            xtol=1e-4
        )
        
        tca_exact_s = res.root
        
        # Calculate miss distance at this exact time
        deb_propagator = build_simplified_propagator(deb_initial_state, deb_epoch, inertial_frame)
        obs_propagator = build_simplified_propagator(obs_state, obs_epoch, inertial_frame) 
        tca_date = simulation_start_epoch.shiftedBy(float(tca_exact_s))
        
        r_deb_tca = deb_propagator.propagate(tca_date).getPVCoordinates().getPosition()
        r_obs_tca = obs_propagator.propagate(tca_date).getPVCoordinates().getPosition()
        
        return tca_exact_s, Vector3D.distance(r_deb_tca, r_obs_tca)
    except ValueError:
        return None, None

def main():
    print("1. Ingesting Ground Radar Tracklet...")
    
    df_ground = pd.read_csv("ground_radar_dataset.csv") # Use your actual path
    ep = df_ground['episode_id'].unique()[2] 
    ep_data = df_ground[df_ground['episode_id'] == ep].copy()
        
    times = ep_data['time_elapsed_s'].values
    ranges = ep_data['noisy_ground_range_m'].values
    
    # Set the overarching simulation time framework
    utc = TimeScalesFactory.getUTC()
    collision_epoch = AbsoluteDate(2026, 5, 6, 12, 10, 0.0, utc)
    simulation_start_epoch = collision_epoch.shiftedBy(-5400.0) # t=0
    inertial_frame = FramesFactory.getEME2000()
    itrf = FramesFactory.getITRF(IERSConventions.IERS_2010, True)
    earth = OneAxisEllipsoid(EARTH_RADIUS, Constants.WGS84_EARTH_FLATTENING, itrf)
    
    lat_rad = np.radians(ep_data['station_lat_deg'].iloc[0])
    lon_rad = np.radians(ep_data['station_lon_deg'].iloc[0])
    station_frame = TopocentricFrame(earth, GeodeticPoint(float(lat_rad), float(lon_rad), 0.0), "Radar")

    # --- DYNAMIC EPOCH SETUP ---
    # 1. DEBRIS ANCHOR (Start of the tracklet)
    t_deb_start = float(times[0]) # e.g., 5095.0
    deb_epoch = simulation_start_epoch.shiftedBy(t_deb_start)
    
    # 2. OBSERVER ANCHOR (Start of the tracklet)
    t_obs_start = float(times[0]) 
    obs_epoch = simulation_start_epoch.shiftedBy(t_obs_start)
    obs_state = ep_data.iloc[0][['obs_x', 'obs_y', 'obs_z', 'obs_vx', 'obs_vy', 'obs_vz']].values


    # --- ORBIT DETERMINATION (OD) ---
    print("\n2. Executing Batch Least Squares OD...")
    
    # Get the true debris state at t=5095 to simulate an IOD guess
    true_initial = ep_data.iloc[0][['true_deb_x', 'true_deb_y', 'true_deb_z', 'true_deb_vx', 'true_deb_vy', 'true_deb_vz']].values
    
    # Inject operational error
    x0 = true_initial.copy()
    x0[0] += 300.0  
    x0[1] -= 300.0  
    x0[2] += 200.0  
    x0[3] += 0.5    
    x0[4] -= 0.5    
    x0[5] += 0.5    
    
    # TIGHT BOUNDS: Prevents the optimizer from exploiting the velocity vector 
    # to find phantom orbits on range-only data.
    pos_bound, vel_bound = 5000.0, 10.0 
    lower_bounds = x0 - np.array([pos_bound, pos_bound, pos_bound, vel_bound, vel_bound, vel_bound])
    upper_bounds = x0 + np.array([pos_bound, pos_bound, pos_bound, vel_bound, vel_bound, vel_bound])
    physical_scale = np.array([1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0])

    t0 = time.time()
    res = least_squares(
        compute_residuals, x0=x0, 
        args=(times, ranges, station_frame, deb_epoch, simulation_start_epoch, inertial_frame),
        bounds=(lower_bounds, upper_bounds),
        method='trf', 
        ftol=1e-3, 
        xtol=1e-3, 
        diff_step=1e-4, 
        max_nfev=30, 
        x_scale=physical_scale, 
        loss='linear'
    )
    od_state = res.x
    
    pos_rmse = np.linalg.norm(od_state[:3] - true_initial[:3])
    vel_rmse = np.linalg.norm(od_state[3:] - true_initial[3:])
    print(f"   OD Complete in {time.time()-t0:.1f}s.")
    print(f"   Position RMSE at {t_deb_start}s : {pos_rmse:.1f} m")
    print(f"   Velocity RMSE at {t_deb_start}s : {vel_rmse:.2f} m/s")


    # --- ORBIT PREDICTION & TCA CALCULATION ---
    print("\n3. Calculating Time of Closest Approach (TCA)...")
    
    tca_time, tca_miss_dist = calculate_precise_tca(
        deb_initial_state=od_state,           # The optimized state from OD 
        obs_state=obs_state,           # Known target state
        deb_epoch=deb_epoch,                  # Anchored at 5095.0
        obs_epoch=obs_epoch,                  # Anchored at 5120.0
        simulation_start_epoch=simulation_start_epoch, 
        inertial_frame=inertial_frame
    )
    
    if tca_time is not None:
        print("\n================ CA METRICS ================")
        print(f"Predicted TCA     : {tca_time:.2f} seconds")
        print(f"True Encounter    : 5400.00 seconds")
        print(f"TCA Timing Error  : {abs(tca_time - 5400.0):.2f} seconds")
        print(f"Miss Dist at TCA  : {tca_miss_dist / 1000:.2f} km")
        print("============================================")
    print(f"Whole process complete in {time.time()-t0:.1f}s.")

if __name__ == "__main__":
    main()