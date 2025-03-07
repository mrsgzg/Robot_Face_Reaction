#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
import torch.nn as nn
import matplotlib.animation as animation

from data_process import get_dataloader
from model import get_model
def visualize_face_sequence(ground_face,predict_face,Speaker_face, save_path):
    """
    Visualize a sequence of face features as an animated plot.
    
    Args:
        face_sequence (ndarray): Array of shape [T, 161] where each row
            contains 68 x-coordinates, 68 y-coordinates, then additional features.
        save_path (str): Path to save the output animation (e.g. 'face_animation.gif').
    """
    T, D = ground_face.shape
    
    # Extract the first 136 features (68 x and 68 y)
    #landmarks = ground_face[:, :136]  # shape: [T, 136]
    
    # Set up the plot.
    fig, (ax1,ax2,ax3) = plt.subplots(1, 3,figsize=(10, 5))
    ax1.set_xlim(0, 250)
    ax1.set_ylim(250, 0)  # 反转 y 轴使得面部显示正常
    
    
    ax2.set_xlim(0, 250)
    ax2.set_ylim(250, 0)  # 反转 y 轴使得面部显示正常
    
    ax3.set_xlim(0, 250)
    ax3.set_ylim(250, 0)  # 反转 y 轴使得面部显示正常
    

    scat1, = ax1.plot([],[], 'ro', markersize=2)
    scat2, = ax2.plot([],[], 'bo', markersize=2)
    scat3, = ax3.plot([],[], 'go', markersize=2)
    def update(frame):
         # shape: [68, 2]
        scat1.set_data(Speaker_face[frame,0:68],Speaker_face[frame,68:136])
        ax1.set_title(f"Frame {frame+1}/{T} Speaker Face: ")

        scat2.set_data(ground_face[frame,0:68],ground_face[frame,68:136])
        ax2.set_title(f" Grount Truth: ")
        scat3.set_data(predict_face[frame,0:68],predict_face[frame,68:136])
        ax3.set_title(f"Generated Reaction: ")
        return scat1,scat2
    
    ani = animation.FuncAnimation(fig, update, frames=T, blit=True, interval=100)
    ani.save(save_path, writer='pillow')
    plt.close(fig)
    print(f"Animation saved to {save_path}")

def load_checkpoint(checkpoint_path, device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    # Optionally, if the keys start with "module.", we can leave it if we wrap the model.
    return state_dict, checkpoint.get('epoch', None)

def inverse_transform_predictions(predictions, scaler):
    """
    Inverse-transform a tensor of shape [B, T, F] using the provided scaler.
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
    parser.add_argument("--sequence_length", type=int, default=50, help="Length of each sequence window.")
    parser.add_argument("--stride", type=int, default=10, help="Stride for sequence creation.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for DataLoader.")
    parser.add_argument("--num_select", type=int, default=1, help="Number of sequences to randomly select per video.")
    
    # Model parameters
    parser.add_argument("--model", type=str, default="gru_encoderdecoder", help="Model type to use.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_100.pt", help="Path to the model checkpoint file.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run testing on.")
    parser.add_argument("--plot_file", type=str, default="genreated2.gif",
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
    # We assume fixed dimensions: speaker_input_dim=161, audio_input_dim=39, output_dim=161, hidden_dim=128, num_layers=1.
    model = get_model(args.model, speaker_input_dim=161, audio_input_dim=39, output_dim=161, hidden_dim=128, num_layers=1)
    model.to(device)
    
    # Load the checkpoint. If the checkpoint was saved using DataParallel, we need to either strip the "module." prefix or wrap the model.
    state_dict, epoch = load_checkpoint(args.checkpoint, device)
    if list(state_dict.keys())[0].startswith("module."):
        # If keys start with "module.", we wrap the model with DataParallel.
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load the face scaler for inverse transformation.
    face_scaler = joblib.load(args.scaler_path)
    
    # Get a batch of test data.
    batch = next(iter(test_loader))
    # Expected batch: speaker_expr, listener_expr, speaker_mfcc, listener_mfcc, speaker_mel, listener_mel
    speaker_expr, listener_expr, speaker_mfcc, _, _, _ = batch
    print(speaker_expr.shape)
    # The dataset returns tensors in shape [B, num_select, T, F]. Flatten them so each sequence is independent.
    if speaker_expr.ndim == 4:
        B, N, T, F_expr = speaker_expr.shape
        speaker_expr = speaker_expr.view(-1, T, F_expr)
        listener_expr = listener_expr.view(-1, T, listener_expr.shape[-1])
        speaker_mfcc = speaker_mfcc.view(-1, T, speaker_mfcc.shape[-1])
    
    # Prepare decoder inputs with teacher forcing: shift listener_expr by one time step.
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
    
    # Move outputs and ground truth to CPU for inverse transformation.
    outputs_np = outputs.cpu().numpy()
    listener_np = listener_expr.cpu().numpy()
    speaker_np = speaker_expr.cpu().numpy()
    # Inverse-transform to get original-scale outputs.
    unscaled_outputs = inverse_transform_predictions(outputs_np, face_scaler)
    unscaled_listener = inverse_transform_predictions(listener_np, face_scaler)
    unscaled_speaker = inverse_transform_predictions(speaker_np, face_scaler)
    

    mse_unscaled = np.mean((unscaled_outputs - unscaled_listener) ** 2)
    print("Test MSE on unscaled data:", mse_unscaled)
    print(unscaled_outputs.shape)
    
    visualize_face_sequence(unscaled_listener[0],unscaled_outputs[0],unscaled_speaker[0] ,args.plot_file)
    #visualize_face_sequence(unscaled_listener[0],"ground.gif")
if __name__ == "__main__":
    main()
