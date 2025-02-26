import argparse
import os
import torch
import torch.optim as optim
from data_process import get_dataloader

def main():
    parser = argparse.ArgumentParser(description="Train a multi-modal NN model for listener face feature generation.")

    # Dataset and scaler paths
    parser.add_argument("--mapping_csv", type=str, default='Robot_dataset/train.csv',
                        help="Path to the training CSV mapping file.")
    parser.add_argument("--val_mapping_csv", type=str, default='Robot_dataset/val.csv',
                        help="Path to the validation CSV mapping file (optional).")
    parser.add_argument("--scaler_path", type=str, default="Robot_Face_Reaction/data_process/Face_Scaler.pkl",
                        help="Path to the face features scaler.")
    parser.add_argument("--audio_scaler_path", type=str, default="Robot_Face_Reaction/data_process/Audio_Scaler.pkl",
                        help="Path to the audio features scaler.")
    parser.add_argument("--num_select", type=int, default=1,
                        help="Number of select sequences.")
    # DataLoader parameters
    parser.add_argument("--batch_size", type=int, default=256,
                        help="Batch size for training.")
    parser.add_argument("--sequence_length", type=int, default=100,
                        help="Length of each sequence window.")
    parser.add_argument("--stride", type=int, default=10,
                        help="Stride for sequence creation.")
    parser.add_argument("--num_workers", type=int, default=16,
                        help="Number of workers for DataLoader.")
    
    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.005,
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
    
    print("Configuration!!!!!!!!:")
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.logs_dir, exist_ok=True)
    
    train_loader = get_dataloader(
        mapping_csv=args.mapping_csv,
        batch_size=args.batch_size,
        stride=args.stride,
        num_workers=args.num_workers,
        sequence_length=args.sequence_length,
        scaler_path=args.scaler_path,
        audio_scaler_path=args.audio_scaler_path,
        num_select=args.num_select
    )
    
    val_loader = None
    if args.val_mapping_csv is not None:
        val_loader = get_dataloader(
            mapping_csv=args.val_mapping_csv,
            batch_size=args.batch_size,
            stride=args.stride,
            num_workers=args.num_workers,
            sequence_length=args.sequence_length,
            scaler_path=args.scaler_path,
            audio_scaler_path=args.audio_scaler_path,
            num_select=args.num_select
        )
    
    from model import get_model
    model = get_model(args.model, **vars(args))
    model.to(args.device)
    
    # Multi-GPU support: wrap the model if more than one GPU is available.
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
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
        checkpoint_interval=args.checkpoint_interval,
        num_select = args.num_select
    )

if __name__ == '__main__':
    main()
