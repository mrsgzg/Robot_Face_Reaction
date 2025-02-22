import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm



def train_model(model, train_loader, val_loader, optimizer, device, epochs, checkpoint_dir, logs_dir, checkpoint_interval,num_select):
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
        num_select (int): Number of sequences selected per video (used in flattening).

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
        print(start_time)
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")

        for batch in progress_bar:
            # Unpack the batch
            # Expected: speaker_expr, listener_expr, speaker_mfcc, speaker_mel, listener_mfcc, listener_mel
            # For the GRU-EncoderDecoder model we use speaker_expr, speaker_mfcc and listener_expr.
            if batch is None:
                continue
            speaker_expr, listener_expr, speaker_mfcc, _, _, _ = batch
            print("1")
            B, N, T, F_expr = speaker_expr.shape
            speaker_expr = speaker_expr.view(-1, T, F_expr)
            listener_expr = listener_expr.view(-1, T, listener_expr.shape[-1])
            speaker_mfcc = speaker_mfcc.view(-1, T, speaker_mfcc.shape[-1])
            print("2")
            # Prepare decoder inputs using teacher forcing: first time step is zeros.
            batch_size, seq_len, out_dim = listener_expr.shape
            decoder_inputs = torch.zeros(batch_size, seq_len, out_dim, device=device)
            decoder_inputs[:, 1:, :] = listener_expr[:, :-1, :]
            print("Here")
            # Move data to device
            speaker_expr = speaker_expr.to(device).float()
            listener_expr = listener_expr.to(device).float()
            speaker_mfcc = speaker_mfcc.to(device).float()

            optimizer.zero_grad()
            print("haofan")
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
                for batch in tqdm(val_loader, desc=f"Epoch {epoch}/{epochs} [Validation]", unit="batch"):
                    if batch is None:
                        continue
                    speaker_expr, listener_expr, speaker_mfcc, _, _, _ = batch
                    if speaker_expr.ndim == 4:
                        B, N, T, F_expr = speaker_expr.shape
                        speaker_expr = speaker_expr.view(-1, T, F_expr)
                        listener_expr = listener_expr.view(-1, T, listener_expr.shape[-1])
                        speaker_mfcc = speaker_mfcc.view(-1, T, speaker_mfcc.shape[-1])
                    
                    batch_size, seq_len, out_dim = listener_expr.shape
                    decoder_inputs = torch.zeros(batch_size, seq_len, out_dim, device=device)
                    decoder_inputs[:, 1:, :] = listener_expr[:, :-1, :]
                    
                    speaker_expr = speaker_expr.to(device).float()
                    listener_expr = listener_expr.to(device).float()
                    speaker_mfcc = speaker_mfcc.to(device).float()
                    decoder_inputs = decoder_inputs.to(device).float()
                    
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
