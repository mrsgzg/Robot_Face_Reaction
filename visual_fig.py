#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
import torch.nn as nn
from data_process import get_dataloader
from model import get_model
from collections import OrderedDict

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    return state_dict, checkpoint.get('epoch', None)

def inverse_transform_predictions(predictions, scaler):
    """
    Inverse-transform a tensor (or numpy array) of shape [B, T, F] using the provided scaler.
    """
    B, T, F = predictions.shape
    predictions_flat = predictions.reshape(B * T, F)
    unscaled_flat = scaler.inverse_transform(predictions_flat)
    return unscaled_flat.reshape(B, T, F)

def main():
    parser = argparse.ArgumentParser(description="Test the trained model for listener face feature generation.")
    
    # Dataset and scaler parameters
    parser.add_argument("--mapping_csv", type=str, default='Robot_dataset/test.csv',
                        help="Path to the test CSV mapping file.")
    parser.add_argument("--scaler_path", type=str, default="Robot_Face_Reaction/data_process/Face_Scaler.pkl",
                        help="Path to the face features scaler.")
    parser.add_argument("--audio_scaler_path", type=str, default="Robot_Face_Reaction/data_process/Audio_Scaler.pkl",
                        help="Path to the audio features scaler.")
    
    # DataLoader parameters
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for testing.")
    parser.add_argument("--sequence_length", type=int, default=100, help="Length of each sequence window.")
    parser.add_argument("--stride", type=int, default=10, help="Stride for sequence creation.")
    parser.add_argument("--num_workers", type=int, default=0, help="Number of workers for DataLoader.")
    parser.add_argument("--num_select", type=int, default=1, help="Number of sequences to randomly select per video.")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="gru_encoderdecoder", help="Model type to use.")
    parser.add_argument("--checkpoint", type=str, default='checkpoints/checkpoint_epoch_50.pt', help="Path to the model checkpoint file.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run testing on.")
    
    # Output plot file path
    parser.add_argument("--plot_file", type=str, default="test_results.png",
                        help="File path to save the result plot.")
    
    args = parser.parse_args()
    device = args.device
    
    # Create the test DataLoader
    test_loader = get_dataloader(
        mapping_csv=args.mapping_csv,
        batch_size=args.batch_size,
        stride=args.stride,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length,
        scaler_path=args.scaler_path,
        audio_scaler_path=args.audio_scaler_path,
        num_select=args.num_select
    )
    
    # Initialize the model using our model factory.
    model = get_model(args.model, speaker_input_dim=161, audio_input_dim=39, output_dim=161, hidden_dim=128, num_layers=1)
    model.to(device)
    
    # Load the checkpoint.
    state_dict, epoch = load_checkpoint(args.checkpoint, device)
    # If keys have "module." prefix, wrap the model.
    if list(state_dict.keys())[0].startswith("module."):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load the face scaler for inverse transformation.
    face_scaler = joblib.load(args.scaler_path)
    
    # Get a batch of test data.
    batch = next(iter(test_loader))
    # Expected batch: speaker_expr, listener_expr, speaker_mfcc, listener_mfcc, speaker_mel, listener_mel
    speaker_expr, listener_expr, speaker_mfcc, _, _, _ = batch
    
    # Flatten the tensors if they have shape [B, num_select, T, F].
    if speaker_expr.ndim == 4:
        B, N, T, F_expr = speaker_expr.shape
        speaker_expr = speaker_expr.view(-1, T, F_expr)
        listener_expr = listener_expr.view(-1, T, listener_expr.shape[-1])
        speaker_mfcc = speaker_mfcc.view(-1, T, speaker_mfcc.shape[-1])
    
    # Prepare decoder inputs using teacher forcing.
    batch_size, seq_len, out_dim = listener_expr.shape
    decoder_inputs = torch.zeros(batch_size, seq_len, out_dim, device=device)
    decoder_inputs[:, 1:, :] = listener_expr[:, :-1, :]
    
    # Move tensors to device.
    speaker_expr = speaker_expr.to(device).float()
    speaker_mfcc = speaker_mfcc.to(device).float()
    decoder_inputs = decoder_inputs.to(device).float()
    
    with torch.no_grad():
        outputs = model(speaker_expr, speaker_mfcc, decoder_inputs)  # Shape: [B, T, 161]
        test_loss = nn.MSELoss()(outputs, listener_expr.to(device).float())
        print("Test Loss (normalized):", test_loss.item())
    
    # Convert outputs and ground truth to numpy arrays for inverse transformation.
    outputs_np = outputs.cpu().numpy()
    listener_np = listener_expr.cpu().numpy()
    
    # Inverse-transform to get original scale.
    unscaled_outputs = inverse_transform_predictions(outputs_np, face_scaler)
    unscaled_listener = inverse_transform_predictions(listener_np, face_scaler)
    
    # Compute MSE on unscaled data.
    mse_unscaled = np.mean((unscaled_outputs - unscaled_listener) ** 2)
    print("Test MSE on unscaled data:", mse_unscaled)
    
    # Visualization: Plot the first feature over time for the first sample.
    plt.figure(figsize=(10, 5))
    
    plt.plot(unscaled_outputs[0, 0, 0:68],unscaled_outputs[0, 0, 68:136], "o",label="Predicted")
    plt.plot(unscaled_listener[0, 0, 0:68],unscaled_listener[0, 0, 68:136], "o",label="Ground Truth")
    plt.xlabel("Time Step")
    plt.ylabel("Feature Value")
    plt.title("Listener Face Feature (Feature 0) Over Time")
    plt.legend()
    plt.gca().invert_yaxis()
    # Save the figure to a file.
    plt.savefig(args.plot_file)
    print(f"Plot saved to {args.plot_file}")

if __name__ == "__main__":
    main()
