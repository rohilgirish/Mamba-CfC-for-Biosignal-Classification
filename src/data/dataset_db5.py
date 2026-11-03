import h5py
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
import numpy as np

class NinaproDB5Dataset(Dataset):
    def __init__(self, h5_path, max_samples=None):
        with h5py.File(h5_path, 'r') as f:
            if max_samples is not None:
                self.X = f['X'][:max_samples]
                self.Y = f['Y'][:max_samples]
            else:
                self.X = f['X'][:]
                self.Y = f['Y'][:]
            
        self.X = torch.from_numpy(self.X).float()
        # Label is at index 3 of metadata: [subject, exercise, repetition, label, label_purity, start_idx]
        self.labels = torch.from_numpy(self.Y[:, 3]).long()
        
    def __len__(self):
        return len(self.X)
        
    def __getitem__(self, idx):
        return self.X[idx], self.labels[idx]

def get_balanced_sampler_db5(dataset):
    """
    Creates a WeightedRandomSampler that balances the classes (e.g. oversampling rare classes,
    undersampling dominant class like rest).
    """
    labels = dataset.labels.numpy()
    num_classes = int(labels.max()) + 1
    class_counts = np.bincount(labels, minlength=num_classes)
    class_weights = np.zeros(num_classes, dtype=np.float32)
    nonzero_mask = class_counts > 0
    class_weights[nonzero_mask] = 1.0 / class_counts[nonzero_mask]
    
    sample_weights = class_weights[labels]
    
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

