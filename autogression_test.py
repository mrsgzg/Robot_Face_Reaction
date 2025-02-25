#!/usr/bin/env python
import argparse
import os
import torch
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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
    Inverse-transform a numpy array of shape [B, T, F] using the provided scaler.
    """
    B, T, F = predictions.shape
    predictions_flat = predictions.reshape(B * T, F)
    unscaled_flat = scaler.inverse_transform(predictions_flat)
    return unscaled_flat.reshape(B, T, F)

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

def main():
    parser = argparse.ArgumentParser(description="Test the trained model and visualize face sequence using autoregressive decoding.")
    
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
    
    # Autoregressive flag
    parser.add_argument("--autoregressive", default=True, help="Use autoregressive decoding (model feeds its own previous output).")
    
    # Visualization output
    parser.add_argument("--plot_file", type=str, default="face_animation.gif",
                        help="File path to save the face animation (e.g., face_animation.gif).")
    
    args = parser.parse_args()
    device = args.device
    
    # Create test DataLoader.
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
    
    # Initialize model (using fixed dimensions as in training).
    model = get_model(args.model, speaker_input_dim=161, audio_input_dim=39, output_dim=161, hidden_dim=128, num_layers=1)
    model.to(device)
    
    # Load checkpoint; if keys are prefixed with "module.", wrap model in DataParallel.
    state_dict, _ = load_checkpoint(args.checkpoint, device)
    if list(state_dict.keys())[0].startswith("module."):
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)
    model.eval()
    
    # Load face scaler.
    face_scaler = joblib.load(args.scaler_path)
    
    # Get a batch of test data.
    batch = next(iter(test_loader))
    
    # Expected batch: speaker_expr, listener_expr, speaker_mfcc, listener_mfcc, speaker_mel, listener_mel
    speaker_expr, listener_expr, speaker_mfcc, _, _, _ = batch
    
    # Flatten tensors if shape is [B, num_select, T, F].
    if speaker_expr.ndim == 4:
        B, N, T, F_expr = speaker_expr.shape
        speaker_expr = speaker_expr.view(-1, T, F_expr)
        listener_expr = listener_expr.view(-1, T, listener_expr.shape[-1])
        speaker_mfcc = speaker_mfcc.view(-1, T, speaker_mfcc.shape[-1])
    
    with torch.no_grad():
        print(f"State of Auto:{args.autoregressive}")
        if args.autoregressive:
            # Use autoregressive decoding.
            # First, obtain the encoder output from the model.
            # If model is DataParallel, use model.module.
            encoder = model.module if isinstance(model, torch.nn.DataParallel) else model
            speaker_expr = speaker_expr.to(device).float()
            speaker_mfcc = speaker_mfcc.to(device).float()
            # Run speaker and audio encoders.
            _, speaker_hidden = encoder.speaker_encoder(speaker_expr)
            speaker_hidden = speaker_hidden[-1]
            _, audio_hidden = encoder.audio_encoder(speaker_mfcc)
            audio_hidden = audio_hidden[-1]
            fused = torch.cat((speaker_hidden, audio_hidden), dim=-1)
            decoder_init = torch.tanh(encoder.fusion_fc(fused))
            decoder_init = decoder_init.unsqueeze(0).repeat(encoder.decoder.num_layers, 1, 1)
            
            # Autoregressive loop.
            B_ar, seq_len, out_dim = listener_expr.shape
            input_t = torch.zeros(B_ar, 1, out_dim, device=device)  # start token (can be zeros)
            predictions = []
            hidden_state = decoder_init
            for t in range(seq_len):
                # One step at a time: input_t shape [B, 1, out_dim].
                out_step, hidden_state = encoder.decoder(input_t, hidden_state)
                pred_t = encoder.out_fc(out_step)  # shape: [B, 1, out_dim]
                predictions.append(pred_t)
                input_t = pred_t  # feed prediction as next input
            outputs = torch.cat(predictions, dim=1)  # shape: [B, seq_len, out_dim]
            test_loss = nn.MSELoss()(outputs, listener_expr.to(device).float())
            print("Test Loss (autoregressive, normalized):", test_loss.item())
        else:
            # Use teacher forcing.
            batch_size, seq_len, out_dim = listener_expr.shape
            decoder_inputs = torch.zeros(batch_size, seq_len, out_dim, device=device)
            decoder_inputs[:, 1:, :] = listener_expr[:, :-1, :]
            outputs = model(speaker_expr, speaker_mfcc, decoder_inputs)
            test_loss = nn.MSELoss()(outputs, listener_expr.to(device).float())
            print("Test Loss (teacher forcing, normalized):", test_loss.item())
    
    # Convert outputs and ground truth to numpy arrays.
    outputs_np = outputs.cpu().numpy()
    listener_np = listener_expr.cpu().numpy()
    speaker_np = speaker_expr.cpu().numpy()
    # Inverse-transform predictions and ground truth.
    unscaled_outputs = inverse_transform_predictions(outputs_np, face_scaler)
    unscaled_listener = inverse_transform_predictions(listener_np, face_scaler)
    unscaled_speaker = inverse_transform_predictions(speaker_np, face_scaler)

    mse_unscaled = np.mean((unscaled_outputs - unscaled_listener) ** 2)
    print("Test MSE on unscaled data:", mse_unscaled)
    
    # Save the first sample's predicted face sequence to a file.
    sample_face_sequence = unscaled_outputs[0]  # shape: [T, 161]
    np.save("sample_face_sequence.npy", sample_face_sequence)
    print("Saved sample face sequence to sample_face_sequence.npy")
    
    # Visualize the face sequence as an animated GIF.
    visualize_face_sequence(unscaled_listener[0],unscaled_outputs[0],unscaled_speaker[0] ,args.plot_file)
    
if __name__ == "__main__":
    main()
