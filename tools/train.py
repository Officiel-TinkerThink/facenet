import argparse
import torch
import os
import yaml
from typing import Dict, Any
from torchvision import transforms
from dataset.dataset import CelebADataset, PKSampler
from model.facenet import FaceNetNN2
from utils.triplet import TripletMiner
import torch.optim as optim
from tqdm import tqdm
import sys


class Trainer:
    def __init__(self, config_path: str) -> None:
        """
        Initializes the Trainer class with the given configuration file and wandb project name.

        Args:
            config_path (str): Path to the YAML configuration file.
        Raise:
            ValueError: If the configuration file does not exist at the specified path.
        """
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config: Dict[str, Any] = yaml.safe_load(f)
            # wandb logging config for experiment tracking
            self.wandb_logging: Dict[str, Any] = config.get("logging")
            # train config are contain of dynamic hyperparams namely epochs,
            # batch, imgsz, others
            self.train_config: Dict[str, Any] = config.get("train")

            self.transform: Dict[str, Any] = config.get('transform')
            # aug config are contain of static augmentations params namely,
        else:
            raise ValueError("Config file does not exist")

    def train(self, name) -> None:
        """
        Executes the training pipeline with wandb logging and model export.

        This method initializes a wandb run, loads and trains a YOLO model,
        exports it to the specified format, and logs the results.

        """
        # setup device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # transforms
        transform = transforms.Compose([
            transforms.Resize((self.transform['resize'], self.transform['resize'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.transform['mean'], std=self.transform['std']),
        ])

        # create dataset
        dataset = CelebADataset(root_dir=self.train_config['data_dir'], transform=transform,
                                max_people=self.train_config['max_people'])

        # create dataloader (PKsampler)
        sampler = PKSampler(dataset, P=self.train_config['batch_p'], K=self.train_config['batch_k'])

        # create model
        model = FaceNetNN2(embedding_dim=self.train_config['embedding_dim'])
        model.to(device)

        # create miner optimizer
        miner =  TripletMiner(margin=self.train_config['margin'])
        optimizer = optim.Adam(model.parameters(), lr=self.train_config['lr'])

        # learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )

        # Training statistics
        stats = {
            'train_loss': [],
            'triplets_per_batch': [],
            'learning_rate': []
        }
        
        print("\n" + "="*60)
        print("Starting Training")
        print("="*60)

        num_epochs = self.train_config['epochs']
        # training loop
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            epoch_triplets = 0
            batch_count = 0

            pbar = tqdm(sampler, desc=f"Epoch {epoch+1} / {num_epochs}")
            
            for batch_indices, batch_labels in pbar:
                # get batch
                batch_images = []
                for idx in batch_indices:
                    img, _ = dataset[idx]
                    batch_images.append(img)
                
                batch_images = torch.stack(batch_images).to(device)
                batch_labels = torch.tensor(batch_labels).to(device)

                # forward pass
                embeddings = model(batch_images)
                print('embeddings device: ',embeddings.device)
                print('batch_labels device: ',batch_labels.device)

                # Mine triplets
                triplets = miner.mine_triplets(embeddings, batch_labels)

                if len(triplets) == 0:
                    continue

                # compute loss
                loss = miner.compute_loss(embeddings, triplets)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Statistics
                epoch_loss += loss.item()
                epoch_triplets += len(triplets)
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'triplets': len(triplets),
                    'lr': optimizer.param_groups[0]['lr']
                })

            # Epoch statistics
            avg_loss = epoch_loss / max(batch_count, 1)
            avg_triplets = epoch_triplets / max(batch_count, 1)
            
            stats['train_loss'].append(avg_loss)
            stats['triplets_per_batch'].append(avg_triplets)
            stats['learning_rate'].append(optimizer.param_groups[0]['lr'])
            
            # Update scheduler
            scheduler.step(avg_loss)
            
            # Print epoch summary
            print(f"\nEpoch {epoch+1} Summary:")
            print(f"  Loss: {avg_loss:.4f}")
            print(f"  Triplets/batch: {avg_triplets:.1f}")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if (epoch + 1) % self.train_config['save_every'] == 0 or epoch == self.train_config['epochs'] - 1:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'config': self.train_config,
                    'stats': stats
                }
                if not os.path.exists(self.train_config['checkpoints']):
                    os.makedirs(self.train_config['checkpoints'])
                torch.save(checkpoint, f"{self.train_config['checkpoints']}/checkpoint_epoch_{epoch+1}.pth")
                print(f"  Saved checkpoint: checkpoint_epoch_{epoch+1}.pth")
            
            print("-" * 50)
        
        # Save final model
        torch.save(model.state_dict(), f"{self.train_config['checkpoints']}/facenet_final.pth")
        print("\nSaved final model: facenet_final.pth")

    

def argparser(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str, help="run name")
    parser.add_argument(
        "--config_path",
        type=str,
        default="configs/settings.yaml",
        help="Path to config path",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = argparser(sys.argv[1:])
    trainer = Trainer(args.config_path)
    if args.name is None:
        raise ValueError("Train run name does not specified")
    trainer.train(args.name)