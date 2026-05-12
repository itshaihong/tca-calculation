import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class ConjunctionDataset(Dataset):
    def __init__(self, tracklets, target_times, target_states):
        self.tracklets = torch.tensor(tracklets, dtype=torch.float32)
        self.target_times = torch.tensor(target_times, dtype=torch.float32)
        self.target_states = torch.tensor(target_states, dtype=torch.float32)
        
    def __len__(self):
        return len(self.tracklets)
    
    def __getitem__(self, idx):
        return self.tracklets[idx], self.target_times[idx], self.target_states[idx]

def load_and_preprocess_data(csv_path, tracklet_length=25, save_scalers=True):
    print(f"Loading dataset from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    # Define our feature columns
    tracklet_cols = ['noisy_range_m', 'noisy_doppler_ms', 'eta_dot']
    target_cols = ['rel_deb_x_rtn', 'rel_deb_y_rtn', 'rel_deb_z_rtn', 
                   'rel_deb_vx_rtn', 'rel_deb_vy_rtn', 'rel_deb_vz_rtn']
    
    raw_tracklets = []
    raw_delta_ts = []
    raw_targets = []
    
    # 1. Parse the Episodes
    episodes = df['episode_id'].unique()
    for ep in episodes:
        ep_df = df[df['episode_id'] == ep].sort_values('time_elapsed_s')
        
        # Split tracklet vs target window based on the time gap
        # Assuming tracklet is 1Hz, a gap > 10s means we've jumped to the collision window
        time_diffs = ep_df['time_elapsed_s'].diff()
        gap_idx = time_diffs[time_diffs > 10.0].index
        
        if len(gap_idx) == 0:
            # Fallback if no gap: take first 'tracklet_length' rows as tracklet
            tracklet_df = ep_df.iloc[:tracklet_length]
            target_df = ep_df.iloc[tracklet_length:]
        else:
            split_loc = ep_df.index.get_loc(gap_idx[0])
            tracklet_df = ep_df.iloc[:split_loc]
            target_df = ep_df.iloc[split_loc:]
            
        # Ensure tracklet is a fixed length for the LSTM (pad or truncate if necessary)
        t_seq = tracklet_df[tracklet_cols].values
        if len(t_seq) > tracklet_length:
            t_seq = t_seq[:tracklet_length]
        elif len(t_seq) < tracklet_length:
            # Pad with the last known value (Zero-Order Hold)
            pad_length = tracklet_length - len(t_seq)
            padding = np.repeat(t_seq[-1:], pad_length, axis=0)
            t_seq = np.vstack([t_seq, padding])
            
        t_end_tracklet = tracklet_df['time_elapsed_s'].iloc[-1]
        
        # For every row in the target window, create a training sample
        for _, row in target_df.iterrows():
            delta_t = row['time_elapsed_s'] - t_end_tracklet
            
            raw_tracklets.append(t_seq)
            raw_delta_ts.append([delta_t])
            raw_targets.append(row[target_cols].values)

    # Convert to Numpy Arrays
    X_track = np.array(raw_tracklets) # Shape: (Samples, Seq_Len, 3) Seq_Len = tracklet_length
    X_time = np.array(raw_delta_ts)   # Shape: (Samples, 1)
    Y_state = np.array(raw_targets)   # Shape: (Samples, 6)
    
    print(f"Extracted {len(Y_state)} samples across {len(episodes)} episodes.")

    # 2. Fit Scalers & Normalize
    scaler_track = StandardScaler()
    scaler_time = StandardScaler()
    scaler_state = StandardScaler()
    
    # Flatten tracklets for scaling (Samples * Seq_Len, 3)
    flat_track = X_track.reshape(-1, 3)
    scaled_flat_track = scaler_track.fit_transform(flat_track)
    X_track_scaled = scaled_flat_track.reshape(X_track.shape)
    
    X_time_scaled = scaler_time.fit_transform(X_time)
    Y_state_scaled = scaler_state.fit_transform(Y_state)
    
    # 3. Save the Scalers for Inference
    if save_scalers:
        joblib.dump(scaler_track, 'scaler_track.pkl')
        joblib.dump(scaler_time, 'scaler_time.pkl')
        joblib.dump(scaler_state, 'scaler_state.pkl')
        print("Scalers saved as .pkl files for inference engine.")
        
    return X_track_scaled, X_time_scaled, Y_state_scaled