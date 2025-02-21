import argparse
import os
import torch
import torch.optim as optim
from data_process import get_dataloader

def main():
    parser = argparse.ArgumentParser(
        description="Train a multi-modal NN model for listener face feature generation."
    )
    
    # Dataset and scaler paths
    parser.add_argument("--mapping_csv", type=str, required=True,
                        help="Path to the training CSV mapping file.")
    parser.add_argument("--val_mapping_csv", type=str, default=None,
                        help="Path to the validation CSV mapping file (optional).")
    parser.add_argument("--scaler_path", type=str, default="scaler.pkl",
                        help="Path to the face features scaler.")
    parser.add_argument("--audio_scaler_path", type=str, default="audio_scaler.pkl",
                        help="Path to the audio features scaler.")
    
    # DataLoader parameters
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for training.")
    parser.add_argument("--sequence_length", type=int, default=100,
                        help="Length of each sequence window.")
    parser.add_argument("--stride", type=int, default=10,
                        help="Stride for sequence creation.")
    parser.add_argument("--num_workers", type=int, default=0,
                        help="Number of workers for DataLoader.")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate for the optimizer.")
    
    # Checkpointing and logging
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/",
                        help="Directory to save model checkpoints.")
    parser.add_argument("--logs_dir", type=str, default="logs/",
                        help="Directory to save training logs and history.")
    parser.add_argument("--checkpoint_interval", type=int, default=5,
                        help="Interval (in epochs) to save checkpoints.")
    
    # Model selection (model factory will load models from the models/ folder)
    parser.add_argument("--model", type=str, default="gru_encoderdecoder",
                        help="Model type to use (e.g., gru_encoderdecoder, attention, Transformer).")
    
    # Device configuration
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run training on.")
    
    args = parser.parse_args()
    
    # Display configuration
    print("Configuration:")
    for arg, value in vars(args).items():
        print(f"  {arg}: {value}")
    
    # Create required directories for checkpoints and logs
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    
    # Create the training dataloader using the dataset module
    train_loader = get_dataloader(
        mapping_csv=args.mapping_csv,
        batch_size=args.batch_size,
        stride=args.stride,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length,
        scaler_path=args.scaler_path,
        audio_scaler_path=args.audio_scaler_path
    )
    
    # Optionally create a validation dataloader if a validation CSV is provided
    val_loader = None
    if args.val_mapping_csv is not None:
        val_loader = get_dataloader(
            mapping_csv=args.val_mapping_csv,
            batch_size=args.batch_size,
            stride=args.stride,
            num_workers=args.num_workers,
            sequence_length=args.sequence_length,
            scaler_path=args.scaler_path,
            audio_scaler_path=args.audio_scaler_path
        )
    
    # Import and instantiate the selected model from the models folder via model.py
    from model import get_model
    model = get_model(args.model, **vars(args))
    model.to(args.device)
    
    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Import the training routine from train.py and start training
    from train import train_model
    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device=args.device,
        epochs=args.epochs,
        checkpoint_dir=args.checkpoint_dir,
        logs_dir=args.logs_dir,
        checkpoint_interval=args.checkpoint_interval
    )

if __name__ == '__main__':
    main()
