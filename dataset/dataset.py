import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import numpy as np
from PIL import Image
import os


class CelebADataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None,
                max_people=None, min_images_per_person=4):
        """
        CelebA dataset with identity labels
        """
        self.root_dir = root_dir
        self.img_dir = os.path.join(root_dir, 'img_align_celeba')
        if os.path.exists(self.img_dir):
            print('image directory exist')
        else:
            raise ValueError("Image directory does not exist")
        self.transform = transform
        self.min_images = min_images_per_person
        self.samples = []
        self.person_to_indices = {}
        
        # Load identity and split
        self._load_data(split)
        
        # Filter people if max_people specified
        if max_people:
            self._filter_people(max_people)
        
        # Create person to indices mapping (for PK sampling)
        self._create_person_mapping()
        
        print(f"CelebA {split}: {len(self.samples)} images, {self.n_people} people")
        print(f"Average {len(self.samples)/self.n_people:.1f} images per person")

    @property   
    def n_people(self):
      return len(self.person_to_indices)

    def _load_data(self, split):
        """Load identity and split files"""
        # Load identity file
        identity_file = os.path.join(self.root_dir, 'identity_CelebA.txt')
        identity_df = pd.read_csv(identity_file, sep=' ', 
                                 header=None, names=['image', 'person_id'])
        
        # Load split file
        split_file = os.path.join(self.root_dir, 'list_eval_partition.csv')
        split_df = pd.read_csv(split_file, sep=',', 
                              header=0,names=['image', 'split'])
        
        # Merge
        merged = pd.merge(identity_df, split_df, on='image')
        print('before split:', len(merged))

        # Filter split (0=train, 1=val, 2=test)
        split_map = {'train': 0, 'val': 1, 'test': 2}
        merged = merged[merged['split'] == split_map[split]]
        print('after split:', len(merged))
        
        # Store
        for _, row in merged.iterrows():
            img_path = os.path.join(self.img_dir, row['image'])
            if os.path.exists(img_path):
                self.samples.append((img_path, row['person_id']))
            else:
                print(f"Image not found: {img_path}")

    def _filter_people(self, max_people):
        """Keep only first max_people for faster training"""
        # Count images per person
        person_counts = {}
        for _, person_id in self.samples:
            person_counts[person_id] = person_counts.get(person_id, 0) + 1
        
        # Get top people with enough images
        valid_people = [p for p, cnt in person_counts.items() 
                       if cnt >= self.min_images]
        valid_people = valid_people[:max_people]
        
        # Filter samples
        self.samples = [(img, pid) for img, pid in self.samples 
                       if pid in valid_people]
        

    def _create_person_mapping(self):
        """Create person_id -> [indices] mapping"""
        for idx, (_, person_id) in enumerate(self.samples):
            if person_id not in self.person_to_indices:
                self.person_to_indices[person_id] = []
            self.person_to_indices[person_id].append(idx)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, person_id = self.samples[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, person_id
    
    def get_person_ids(self):
        return list(self.person_to_indices.keys())

class PKSampler:
    def __init__(self, dataset: CelebADataset, P=32, K=4):
        """
        P: number of people per batch
        K: images per person
        """
        self.dataset = dataset
        self.P = P
        self.K = K
        
        # Get people with at least K images
        self.valid_people = []
        self.people_indices = {}
        
        for person_id, indices in dataset.person_to_indices.items():
            if len(indices) >= K:
                self.valid_people.append(person_id)
                self.people_indices[person_id] = indices
        
        print(f"PK Sampler: {len(self.valid_people)} people with ≥{K} images")
        print(f"Batch size: {P}×{K} = {P*K} images")
    
    def __iter__(self):
        """Generate batches"""
        np.random.shuffle(self.valid_people)
        
        for i in range(0, len(self.valid_people), self.P):
            batch_people = self.valid_people[i:i + self.P]
            if len(batch_people) < self.P:
                continue
            
            batch_indices = []
            batch_labels = []
            
            for label_idx, person_id in enumerate(batch_people):
                indices = self.people_indices[person_id]
                selected = np.random.choice(indices, self.K, replace=False)
                batch_indices.extend(selected)
                batch_labels.extend([label_idx] * self.K)
            
            yield batch_indices, batch_labels
    
    def __len__(self):
        return len(self.valid_people) // self.P
