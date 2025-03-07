import argparse
import os
import torch
import torch.nn as nn
from data_process import get_dataloader
from tqdm import tqdm

def test_model(model, test_loader, device):
    """
    Evaluate the model on the test dataset.
    
    Args:
        model (nn.Module): The trained model.
        test_loader (DataLoader): DataLoader for the test set.
        device (str): Device to run inference on.
    
    Returns:
        float: Average test loss computed over all batches.
    """
    model.eval()
    criterion = nn.MSELoss()
    total_loss = 0.0
    batch_count = 0
    #progress_bar = tqdm(test_loader, desc=f"Epoch {epoch}/{epochs}", unit="batch")

    with torch.no_grad():
        for batch in test_loader:
            if batch is None:
                continue
            # Unpack the batch. For our GRU-EncoderDecoder, we use:
            # speaker_expr, listener_expr, speaker_mfcc, speaker_mel, listener_mfcc, listener_mel
            
            speaker_expr, listener_expr, speaker_mfcc, _, _, _ = batch
            B, N, T, F_expr = speaker_expr.shape
            speaker_expr = speaker_expr.view(-1, T, F_expr)
            listener_expr = listener_expr.view(-1, T, listener_expr.shape[-1])
            speaker_mfcc = speaker_mfcc.view(-1, T, speaker_mfcc.shape[-1])
            # Prepare decoder inputs using teacher forcing: first time step is zeros.
            batch_size, seq_len, out_dim = listener_expr.shape
            decoder_inputs = torch.zeros(batch_size, seq_len, out_dim, device=device)
            decoder_inputs[:, 1:, :] = listener_expr[:, :-1, :]

            # Move data to device
            speaker_expr = speaker_expr.to(device).float()
            listener_expr = listener_expr.to(device).float()
            speaker_mfcc = speaker_mfcc.to(device).float()
            
            outputs = model(speaker_expr, speaker_mfcc, decoder_inputs)
            loss = criterion(outputs, listener_expr)
            total_loss += loss.item()
            batch_count += 1
            
    avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
    return avg_loss



def main():
    parser = argparse.ArgumentParser(description="Test the trained multi-modal model on test data.")
    
    # Dataset and scaler paths
    parser.add_argument("--mapping_csv", type=str, default="Robot_dataset/test.csv",
                        help="Path to the test CSV mapping file.")
    parser.add_argument("--scaler_path", type=str, default="Robot_Face_Reaction/data_process/Face_Scaler.pkl",
                        help="Path to the face features scaler.")
    parser.add_argument("--audio_scaler_path", type=str, default="Robot_Face_Reaction/data_process/Audio_Scaler.pkl",
                        help="Path to the audio features scaler.")
    
    # DataLoader parameters
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for testing.")
    parser.add_argument("--sequence_length", type=int, default=100,
                        help="Length of each sequence window.")
    parser.add_argument("--stride", type=int, default=10,
                        help="Stride for sequence creation.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for DataLoader.")
    parser.add_argument("--num_select", type=int, default=1,
                        help="Number of select sequences.")
    # Model and checkpoint
    parser.add_argument("--model", type=str, default="gru_encoderdecoder",
                        help="Model type to use for testing. Should correspond to a model in the models/ folder.")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/checkpoint_epoch_40.pt",
                        help="Path to the model checkpoint file to load.")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run testing on.")
    
    args = parser.parse_args()
    
    print("Test configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Create test DataLoader using dataset.py
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
    
    # Instantiate the model using the factory function in model.py
    from model import get_model
    model = get_model(args.model, **vars(args))
    model.to(args.device)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    # Load checkpoint
    if not os.path.exists(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}")
        return
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from {args.checkpoint} (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Evaluate on the test set
    test_loss = test_model(model, test_loader, args.device)
    print(f"Test Loss: {test_loss:.6f}")

if __name__ == '__main__':
    main()
