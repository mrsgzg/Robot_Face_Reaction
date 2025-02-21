import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

def sample_sequences(tensor, num_select=8):
    """
    Randomly sample 'num_select' sequences from each sample in the batch.
    
    Args:
        tensor (Tensor): Shape [B, N, T, F]
        num_select (int): Number of sequences to select per sample.
    
    Returns:
        Tensor: Sampled tensor of shape [B, num_select, T, F]
    """
    B, N, T, F = tensor.size()
    # For each sample in the batch, randomly pick num_select indices from [0, N)
    indices = torch.randint(low=0, high=N, size=(B, num_select))
    sampled = torch.stack([tensor[b, indices[b], :, :] for b in range(B)], dim=0)
    return sampled

def train_model(model, train_loader, val_loader, optimizer, device, epochs, checkpoint_dir, logs_dir, checkpoint_interval):
    """
    Train the given model.

    Args:
        model (nn.Module): The neural network to train.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader or None): DataLoader for validation data (can be None).
        optimizer (Optimizer): Optimizer.
        device (str): Device to train on ('cuda' or 'cpu').
        epochs (int): Total number of training epochs.
        checkpoint_dir (str): Directory to save model checkpoints.
        logs_dir (str): Directory to save training logs.
        checkpoint_interval (int): Save a checkpoint every N epochs.
    """
    # Mean Squared Error loss for regression
    criterion = nn.MSELoss()
    
    # Set up TensorBoard logging
    writer = SummaryWriter(log_dir=logs_dir)
    
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        num_batches = 0
        start_time = time.time()
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")

        for batch in progress_bar:
            # Unpack the batch
            # Expected: speaker_expr, listener_expr, speaker_mfcc, speaker_mel, listener_mfcc, listener_mel
            # For the GRU-EncoderDecoder model we use speaker_expr, speaker_mfcc and listener_expr.
            if batch is None:
                continue
            speaker_expr, listener_expr, speaker_mfcc, _, _, _ = batch
            
            # Move data to device
            speaker_expr = speaker_expr.to(device).float()
            listener_expr = listener_expr.to(device).float()
            speaker_mfcc = speaker_mfcc.to(device).float()
            
            speaker_expr = sample_sequences(speaker_expr, num_select=1)
            listener_expr = sample_sequences(listener_expr, num_select=1)
            speaker_mfcc = sample_sequences(speaker_mfcc, num_select=1)

            # Prepare decoder inputs using teacher forcing:
            # For simplicity, we create a decoder input sequence by shifting the listener face features by one time step.
            B, N, T, output_dim = listener_expr.size()
            speaker_expr = speaker_expr.view(-1, T, speaker_expr.size(-1))      # shape: [B*N, T, speaker_dim]
            listener_expr = listener_expr.view(-1, T, output_dim)                 # shape: [B*N, T, output_dim]
            speaker_mfcc = speaker_mfcc.view(-1, T, speaker_mfcc.size(-1))         # shape: [B*N, T, audio_dim]
            # Now, update batch_size to the new flattened size
            new_batch_size = speaker_expr.size(0)
            decoder_inputs = torch.zeros(new_batch_size, T, output_dim, device=device)
            decoder_inputs[:, 1:, :] = listener_expr[:, :-1, :]

            # Forward pass
            outputs = model(speaker_expr, speaker_mfcc, decoder_inputs)

            decoder_inputs = torch.zeros(batch_size, seq_len, output_dim, device=device)
            decoder_inputs[:, 1:, :] = listener_expr[:, :-1, :]
            
            optimizer.zero_grad()
            outputs = model(speaker_expr, speaker_mfcc, decoder_inputs)
            loss = criterion(outputs, listener_expr)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item())
        
        avg_loss = running_loss / num_batches if num_batches > 0 else float('inf')
        epoch_time = time.time() - start_time
        print(f"Epoch [{epoch}/{epochs}] - Training Loss: {avg_loss:.6f} - Time: {epoch_time:.2f}s")
        writer.add_scalar("Loss/train", avg_loss, epoch)
        
        # Validation loop if validation loader is provided
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    if batch is None:
                        continue
                    speaker_expr, listener_expr, speaker_mfcc, _, _, _ = batch
                    speaker_expr = speaker_expr.to(device).float()
                    listener_expr = listener_expr.to(device).float()
                    speaker_mfcc = speaker_mfcc.to(device).float()
                    
                    batch_size, seq_len, output_dim = listener_expr.size()
                    decoder_inputs = torch.zeros(batch_size, seq_len, output_dim, device=device)
                    decoder_inputs[:, 1:, :] = listener_expr[:, :-1, :]
                    
                    outputs = model(speaker_expr, speaker_mfcc, decoder_inputs)
                    loss = criterion(outputs, listener_expr)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else float('inf')
            print(f"Epoch [{epoch}/{epochs}] - Validation Loss: {avg_val_loss:.6f}")
            writer.add_scalar("Loss/validation", avg_val_loss, epoch)
        
        # Save checkpoint every 'checkpoint_interval' epochs
        if epoch % checkpoint_interval == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    writer.close()
    print("Training complete.")
