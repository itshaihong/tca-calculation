import os
import time
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import joblib
from scipy.optimize import root_scalar

# ---------------------------------------------------------
# 0. ENVIRONMENT & ARCHITECTURE SETUP
# ---------------------------------------------------------
# Suppress the OpenMP Intel collision warning common in Windows ML environments
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class MethodBPropagator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=32, output_dim=6):
        super(MethodBPropagator, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 128),
            nn.GELU(),  
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, tracklet_seq, delta_t):
        lstm_out, (h_n, c_n) = self.lstm(tracklet_seq)
        latent_vec = self.latent_proj(h_n[-1])
        decoder_input = torch.cat([latent_vec, delta_t], dim=1)
        return self.decoder(decoder_input)


# ---------------------------------------------------------
# 1. THE ML ROOT SOLVER (REPLACES OREKIT PROPAGATORS)
# ---------------------------------------------------------
def orthogonality_condition_ml(t_guess_s, tracklet_tensor, ml_model, scaler_time, scaler_state):
    """
    Evaluates the dot product of Relative Position and Velocity at time t_guess_s.
    t_guess_s is the delta_t in seconds from the end of the radar tracklet.
    """
    # 1. Scale the target time guess
    t_np = np.array([[t_guess_s]])
    t_scaled = scaler_time.transform(t_np)
    t_tensor = torch.tensor(t_scaled, dtype=torch.float32)
    
    # 2. Ask the ML model for the RTN Relative State (O(1) Inference)
    with torch.no_grad():
        pred_scaled = ml_model(tracklet_tensor, t_tensor)
        
    # 3. Inverse transform back to physical meters and m/s
    pred_physical = scaler_state.inverse_transform(pred_scaled.numpy())[0]
    
    r_rel = pred_physical[:3]  # [rel_x, rel_y, rel_z]
    v_rel = pred_physical[3:]  # [rel_vx, rel_vy, rel_vz]
    
    # 4. Return the dot product (When 0, we are at TCA)
    return np.dot(r_rel, v_rel)


# ---------------------------------------------------------
# 2. MAIN EXECUTION PIPELINE
# ---------------------------------------------------------
def main():
    print("1. Loading ML Edge Engine & Scalers...")
    try:
        scaler_track = joblib.load('scaler_track.pkl')
        scaler_time  = joblib.load('scaler_time.pkl')
        scaler_state = joblib.load('scaler_state.pkl')
    except FileNotFoundError:
        print("CRITICAL ERROR: Scaler .pkl files not found.")
        return

    # Initialize and load model
    ml_model = MethodBPropagator(input_dim=3, output_dim=6)
    ml_model.load_state_dict(torch.load('onboard_ml_propagator.pth', weights_only=True))
    ml_model.eval()

    print("\n2. Ingesting Radar Tracklet from Test Dataset...")
    PATH = os.getcwd()
    df = pd.read_csv(f"{PATH}\\..\\ml_testing_dataset.csv")
    
    # Grab the first episode to test
    ep = df['episode_id'].unique()[0]
    ep_df = df[df['episode_id'] == ep].sort_values('time_elapsed_s')
    
    # Extract the 25-row Tracklet
    tracklet_length = 25
    tracklet_cols = ['noisy_range_m', 'noisy_doppler_ms', 'eta_dot']
    tracklet_df = ep_df.iloc[:tracklet_length]
    t_end_tracklet = tracklet_df['time_elapsed_s'].iloc[-1]
    
    # Scale and package the tracklet for the PyTorch model
    raw_tracklet = tracklet_df[tracklet_cols].values
    scaled_tracklet = scaler_track.transform(raw_tracklet)
    # Add batch dimension: Shape becomes (1, 25, 3)
    tracklet_tensor = torch.tensor(np.expand_dims(scaled_tracklet, axis=0), dtype=torch.float32)


    print("\n3. Calculating Time of Closest Approach (TCA) via Continuous Inference...")
    t0 = time.time()
    
    # --- DYNAMIC BRACKET FINDER ---
    # We sweep forward from the end of the tracklet (delta_t = 0) up to 600 seconds 
    # to find the exact 10-second window where the dot product flips from negative to positive.
    search_steps = np.arange(0.0, 600.0, 10.0) 
    valid_bracket = None
    
    prev_dot = orthogonality_condition_ml(search_steps[0], tracklet_tensor, ml_model, scaler_time, scaler_state)
    
    for i in range(1, len(search_steps)):
        current_t = search_steps[i]
        curr_dot = orthogonality_condition_ml(current_t, tracklet_tensor, ml_model, scaler_time, scaler_state)
        
        # If the sign flips (multiplication of signs is negative), we found the crossing!
        if np.sign(prev_dot) != np.sign(curr_dot):
            valid_bracket = [search_steps[i-1], current_t]
            break
        prev_dot = curr_dot
        
    if valid_bracket is None:
        print("CRITICAL: Debris does not pass the observer within the 600-second search window.")
        return

    # --- EXACT ROOT ISOLATION ---
    try:
        # Now we pass the mathematically verified bracket to Brent's Method
        res = root_scalar(
            orthogonality_condition_ml, 
            args=(tracklet_tensor, ml_model, scaler_time, scaler_state),
            bracket=valid_bracket, 
            method='brentq', 
            xtol=1e-4
        )
        tca_delta_t = res.root
        
        # Calculate Absolute TCA Time
        absolute_tca = t_end_tracklet + tca_delta_t
        
        # Fetch the final Miss Distance at the exact TCA
        t_np = np.array([[tca_delta_t]])
        t_scaled = scaler_time.transform(t_np)
        t_tensor = torch.tensor(t_scaled, dtype=torch.float32)
        
        with torch.no_grad():
            final_pred_scaled = ml_model(tracklet_tensor, t_tensor)
            
        final_state = scaler_state.inverse_transform(final_pred_scaled.numpy())[0]
        miss_distance = np.linalg.norm(final_state[:3]) 
        
        inference_time = time.time() - t0
        
        print("\n================ ML OP METRICS ================")
        print(f"Predicted TCA     : {absolute_tca:.2f} seconds")
        print(f"True Encounter    : 5400.00 seconds")
        print(f"TCA Timing Error  : {abs(absolute_tca - 5400.0):.2f} seconds")
        print(f"Miss Dist at TCA  : {miss_distance / 1000:.2f} km")
        print(f"TCA Compute Time  : {inference_time:.4f} seconds")
        print("===============================================")
        
    except ValueError as e:
        print(f"Root Solver Failed: {e}")

if __name__ == "__main__":
    main()