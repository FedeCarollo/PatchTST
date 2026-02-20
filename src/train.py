import os
import json
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional, Dict, Any, List
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from checkpoint import Checkpoint


class TrainConfig:
    def __init__(
        self,
        epochs: int = 10,
        learning_rate: float = 1e-3,
        batch_size: int = 32,
        device: str = 'cuda',
        checkpoint_dir: str = './checkpoints',
        save_every: int = 5,
        early_stopping_patience: Optional[int] = None
    ) -> None:
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.checkpoint_dir = checkpoint_dir
        self.save_every = save_every
        self.early_stopping_patience = early_stopping_patience

    def to_dict(self) -> Dict[str, Any]:
        return {
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'device': self.device,
            'checkpoint_dir': self.checkpoint_dir,
            'save_every': self.save_every,
            'early_stopping_patience': self.early_stopping_patience
        }

    def save(self, path: str) -> None:
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=4)

    @staticmethod
    def load(path: str) -> 'TrainConfig':
        with open(path, 'r') as f:
            config_dict = json.load(f)
        return TrainConfig(**config_dict)


class Trainer:
    def __init__(
        self, 
        model: nn.Module, 
        train_loader: DataLoader, 
        test_loader: DataLoader, 
        optimizer: Optimizer, 
        config: TrainConfig, 
        scheduler: Optional[LRScheduler] = None
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        
        self.device = config.device
        self.model.to(self.device)
        
        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # Save configuration
        config.save(os.path.join(config.checkpoint_dir, 'config.json'))
        
        # Training history
        self.history = {
            'train_loss': [],
            'test_loss': [],
            'epochs': []
        }
        
        self.best_test_loss = float('inf')
        self.epochs_no_improve = 0
        self.start_epoch = 0

    def train_epoch(self) -> float:
        """Train for one epoch"""
        self.model.train()
        train_loss = 0.0
        train_pbar = tqdm(self.train_loader, desc='Training')
        
        for batch in train_pbar:
            input = batch['input'].to(self.device)
            target = batch['target'].to(self.device)

            pred = self.model(input)
            loss = nn.functional.mse_loss(pred, target)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            train_loss += loss.item()
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return train_loss / len(self.train_loader)

    def test_epoch(self) -> float:
        """Evaluate on test set"""
        self.model.eval()
        test_loss = 0.0
        test_pbar = tqdm(self.test_loader, desc='Testing')
        
        with torch.no_grad():
            for batch in test_pbar:
                input = batch['input'].to(self.device)
                target = batch['target'].to(self.device)

                pred = self.model(input)
                loss = nn.functional.mse_loss(pred, target)
                test_loss += loss.item()
                test_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return test_loss / len(self.test_loader)

    def train(self) -> Dict[str, List[float]]:
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Configuration: {self.config.to_dict()}")
        print(f"Checkpoints will be saved to: {self.config.checkpoint_dir}\n")
        
        for epoch in range(self.start_epoch, self.config.epochs):
            print(f"\nEpoch {epoch+1}/{self.config.epochs}")
            print("-" * 50)
            
            # Train and test
            avg_train_loss = self.train_epoch()
            avg_test_loss = self.test_epoch()
            
            # Step scheduler if provided
            if self.scheduler is not None:
                # Some schedulers (e.g., ReduceLROnPlateau) require a metric
                if hasattr(self.scheduler, 'step') and 'metrics' in self.scheduler.step.__code__.co_varnames:
                    self.scheduler.step(avg_test_loss)
                else:
                    self.scheduler.step()
            
            # Update history
            self.history['train_loss'].append(avg_train_loss)
            self.history['test_loss'].append(avg_test_loss)
            self.history['epochs'].append(epoch + 1)
            
            print(f'\nEpoch {epoch+1} Summary:')
            print(f'  Train Loss: {avg_train_loss:.4f}')
            print(f'  Test Loss:  {avg_test_loss:.4f}')
            
            # Save checkpoint periodically
            if (epoch + 1) % self.config.save_every == 0:
                Checkpoint.save(
                    checkpoint_path=self.config.checkpoint_dir,
                    checkpoint_name=f'checkpoint_epoch_{epoch+1}.pth',
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    scheduler=self.scheduler,
                    train_loss=avg_train_loss,
                    test_loss=avg_test_loss
                )
                print(f'  Checkpoint saved: checkpoint_epoch_{epoch+1}.pth')
            
            # Save best model
            if avg_test_loss < self.best_test_loss:
                self.best_test_loss = avg_test_loss
                self.epochs_no_improve = 0
                Checkpoint.save(
                    checkpoint_path=self.config.checkpoint_dir,
                    checkpoint_name='best_model.pth',
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch + 1,
                    scheduler=self.scheduler,
                    train_loss=avg_train_loss,
                    test_loss=avg_test_loss
                )
                print(f'  New best model saved! Test Loss: {avg_test_loss:.4f}')
            else:
                self.epochs_no_improve += 1
            
            # Early stopping
            if self.config.early_stopping_patience is not None:
                if self.epochs_no_improve >= self.config.early_stopping_patience:
                    print(f'\nEarly stopping triggered after {epoch+1} epochs')
                    print(f'No improvement for {self.epochs_no_improve} epochs')
                    break
            
            # Always save latest checkpoint
            Checkpoint.save(
                checkpoint_path=self.config.checkpoint_dir,
                checkpoint_name='latest.pth',
                model=self.model,
                optimizer=self.optimizer,
                epoch=epoch + 1,
                scheduler=self.scheduler,
                train_loss=avg_train_loss,
                test_loss=avg_test_loss
            )
        
        print("\n" + "=" * 50)
        print("Training completed!")
        print(f"Best test loss: {self.best_test_loss:.4f}")
        print(f"Checkpoints saved in: {self.config.checkpoint_dir}")
        
        # Save training history
        history_path = os.path.join(self.config.checkpoint_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to: history.json")
        
        return self.history

    def resume_from_checkpoint(self, checkpoint_name: str = 'latest.pth') -> None:
        """Resume training from a checkpoint"""
        print(f"Resuming from checkpoint: {checkpoint_name}")
        result = Checkpoint.load(
            checkpoint_path=self.config.checkpoint_dir,
            checkpoint_name=checkpoint_name,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler
        )
        
        self.start_epoch = result['epoch']
        print(f"Resuming from epoch {self.start_epoch}")
        
        if result['train_loss'] is not None:
            print(f"Previous train loss: {result['train_loss']:.4f}")
        if result['test_loss'] is not None:
            print(f"Previous test loss: {result['test_loss']:.4f}")
            self.best_test_loss = result['test_loss']
