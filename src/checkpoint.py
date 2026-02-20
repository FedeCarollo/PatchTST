import os
import torch

class Checkpoint:
    @staticmethod
    def save(checkpoint_path, checkpoint_name='latest.pth', model=None , optimizer=None, epoch=None, scheduler=None, train_loss=None, test_loss=None):
        path = os.path.join(checkpoint_path, checkpoint_name)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict() if model else None,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'train_loss': train_loss,
            'test_loss': test_loss
        }, path)

    @staticmethod
    def load(checkpoint_path, checkpoint_name='latest.pth', model=None, optimizer=None, scheduler=None):
        path = os.path.join(checkpoint_path, checkpoint_name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Checkpoint file not found: {path}")
        
        checkpoint = torch.load(path)
        
        if model is not None:
            model.load_state_dict(checkpoint['model_state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if scheduler is not None and checkpoint['scheduler_state_dict'] is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        return {
            'epoch': checkpoint['epoch'],
            'train_loss': checkpoint.get('train_loss'),
            'test_loss': checkpoint.get('test_loss')
        }

    

