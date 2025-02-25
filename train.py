import os
import time
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import random



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
    # Hyperparameters for scheduled sampling and noise injection
    decay_rate = 0.025      # how fast teacher forcing ratio decays per epoch
    noise_std = 0.05       # standard deviation for Gaussian noise
    
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
            B, N, T, F_expr = speaker_expr.shape
            speaker_expr = speaker_expr.view(-1, T, F_expr)
            listener_expr = listener_expr.view(-1, T, listener_expr.shape[-1])
            speaker_mfcc = speaker_mfcc.view(-1, T, speaker_mfcc.shape[-1])
            # Prepare decoder inputs using teacher forcing: first time step is zeros.
            batch_size, seq_len, out_dim = listener_expr.shape

            


            # --- Scheduled Sampling: Autoregressive Decoding in Training ---
            # To access the encoder and decoder parts, we assume the model is GRUEncoderDecoder.
            encoder = model.module if isinstance(model, torch.nn.DataParallel) else model
            # Run encoders on speaker features and audio.
            speaker_expr = speaker_expr.to(device).float()
            speaker_mfcc = speaker_mfcc.to(device).float()
            _, speaker_hidden = encoder.speaker_encoder(speaker_expr)
            _, audio_hidden = encoder.audio_encoder(speaker_mfcc)

            
            # Use the last layer's hidden state.
            speaker_hidden = speaker_hidden[-1]  # shape: [B, hidden_dim]
            audio_hidden = audio_hidden[-1]      # shape: [B, hidden_dim]
            fused = torch.cat((speaker_hidden, audio_hidden), dim=-1)  # shape: [B, 2*hidden_dim]
            decoder_init = torch.tanh(encoder.fusion_fc(fused))        # shape: [B, hidden_dim]
            decoder_init = decoder_init.unsqueeze(0).repeat(encoder.decoder.num_layers, 1, 1)
            hidden_state = decoder_init

            # Initialize the first input token (start token) as zeros.
            input_t = torch.zeros(batch_size, 1, out_dim, device=device)
            predictions = []
            
            # Compute teacher forcing ratio for this epoch.
            teacher_forcing_ratio = max(0.25, 1.0 - epoch * decay_rate)
            for t in range(seq_len):
                # Run decoder for one time step.
                out_step, hidden_state = encoder.decoder(input_t, hidden_state)  # out_step: [B, 1, hidden_dim]
                pred_t = encoder.out_fc(out_step)  # shape: [B, 1, out_dim]
                predictions.append(pred_t)
                
                # Scheduled sampling: with probability teacher_forcing_ratio, use ground truth.
                if random.random() < teacher_forcing_ratio:
                    # Use ground truth (with noise injection)
                    gt_t = listener_expr[:, t:t+1, :].to(device).float()
                    next_input = gt_t + torch.randn_like(gt_t) * noise_std
                else:
                    # Use model's own prediction.
                    next_input = pred_t.detach()
                input_t = next_input
            outputs = torch.cat(predictions, dim=1)  # shape: [B, seq_len, out_dim]
            # ------------------------------------------------------------------

            optimizer.zero_grad()
            
            #outputs = model(speaker_expr, speaker_mfcc, decoder_inputs)
            
            loss = criterion(outputs, listener_expr.to(device).float())
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix(loss=loss.item(), tf_ratio=teacher_forcing_ratio)
        
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
