import argparse
import os
import torch
import torch.optim as optim
from data_process import get_dataloader

def main():
    parser = argparse.ArgumentParser(description="Train a multi-modal NN model for listener face feature generation.")
    
    # ... [other argument definitions] ...
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run training on.")
    
    args = parser.parse_args()
    
    print("Configuration:")
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
        audio_scaler_path=args.audio_scaler_path
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
            audio_scaler_path=args.audio_scaler_path
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
        checkpoint_interval=args.checkpoint_interval
    )

if __name__ == '__main__':
    main()
