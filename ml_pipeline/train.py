import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
from data_loader import load_and_preprocess_data, ConjunctionDataset
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import joblib

# =====================================================================
# 1. THE ARCHITECTURE (METHOD B)
# =====================================================================
class MethodBPropagator(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=64, latent_dim=32, output_dim=6):
        super(MethodBPropagator, self).__init__()
        
        # STAGE 1: The Sensor Encoder (LSTM)
        # Ingests the 25-second radar tracklet to understand the physics
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=2, batch_first=True)
        self.latent_proj = nn.Linear(hidden_dim, latent_dim)
        
        # STAGE 2: The Time-Conditioned Decoder (MLP)
        # Takes the physics understanding (latent_vector) + Target Time (delta_t)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim + 1, 128),
            nn.GELU(),  # GELU is smoother for physics gradients than ReLU
            nn.Linear(128, 128),
            nn.GELU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, tracklet_seq, delta_t):
        # tracklet_seq shape: (Batch, Seq_Len, Features)
        # delta_t shape: (Batch, 1)
        
        # 1. Encode the tracklet
        lstm_out, (h_n, c_n) = self.lstm(tracklet_seq)
        
        # Get the final hidden state of the top LSTM layer
        last_hidden = h_n[-1] 
        latent_vec = self.latent_proj(last_hidden)
        
        # 2. Condition on Time
        # Concatenate [Latent Physics Context] + [Target Delta T]
        decoder_input = torch.cat([latent_vec, delta_t], dim=1)
        
        # 3. Predict Future State
        predicted_state = self.decoder(decoder_input)
        return predicted_state



# =====================================================================
# 2. TRAINING & EVALUATION PIPELINE
# =====================================================================
def train_model(model, train_loader, epochs=50, lr=0.001, save_path="ml_propagator.pth"):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    print(f"\n--- Starting Training ({epochs} Epochs) ---")
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for tracklets, delta_ts, true_states in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(tracklets, delta_ts)
            
            # Loss and Backprop
            loss = criterion(predictions, true_states)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")
            
    # Save the Model
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")



# =====================================================================
# EXECUTION
# =====================================================================
if __name__ == "__main__":
    # 1. Load and Scale Data
    PATH = os.getcwd()
    X_track, X_time, Y_state = load_and_preprocess_data(f"{PATH}/../ml_training_dataset.csv")

    X_tr, X_te, T_tr, T_te, Y_tr, Y_te = train_test_split(
        X_track, X_time, Y_state, test_size=0.2, random_state=42
    )
    
    train_dataset = ConjunctionDataset(X_tr, T_tr, Y_tr)
    test_dataset = ConjunctionDataset(X_te, T_te, Y_te)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # 2. Instantiate Model
    ml_propagator = MethodBPropagator(input_dim=3, output_dim=6)
    
    # 3. Train and Save
    model_save_file = "onboard_ml_propagator.pth"
    train_model(ml_propagator, train_loader, epochs=50, save_path=model_save_file)
    
    # 4. Load and Evaluate
    evaluate_model(model_save_file, test_loader)