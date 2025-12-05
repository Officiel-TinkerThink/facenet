import torch
import torch.nn.functional as F

class TripletMiner:
    def __init__(self, margin=0.2, device='cpu'):
        self.margin = margin
    
    def pairwise_distance(self, embeddings):
        """Compute pairwise squared Euclidean distances"""
        # embeddings: (batch_size, embedding_dim)
        dot_product = torch.matmul(embeddings, embeddings.t())
        square_norm = torch.diag(dot_product)
        distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
        distances = F.relu(distances)  # Ensure non-negative
        return distances
    
    def mine_triplets(self, embeddings: torch.Tensor, labels):
        """
        Online semi-hard triplet mining
        Returns: list of (anchor_idx, positive_idx, negative_idx)
        """
        batch_size = embeddings.size(0)
        dist_matrix = self.pairwise_distance(embeddings)
        
        triplets = []
        
        for i in range(batch_size):
            # Positive indices (same label, not self)
            pos_mask = (labels == labels[i]) & (torch.arange(batch_size) != i).to(embeddings.device)
            pos_indices = torch.where(pos_mask)[0]
            
            if len(pos_indices) == 0:
                continue
            
            # Negative indices (different label)
            neg_mask = labels != labels[i]
            neg_indices = torch.where(neg_mask)[0]
            
            if len(neg_indices) == 0:
                continue
            
            # For each positive
            for pos_idx in pos_indices:
                pos_dist = dist_matrix[i, pos_idx]
                
                # Find semi-hard negatives
                neg_dists = dist_matrix[i, neg_indices]
                semi_hard_mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + self.margin)
                semi_hard = neg_indices[semi_hard_mask]
                
                if len(semi_hard) > 0:
                    # Random semi-hard negative
                    neg_idx = semi_hard[torch.randint(0, len(semi_hard), (1,))]
                    triplets.append((i, pos_idx, neg_idx))
        
        return triplets
    
    def compute_loss(self, embeddings, triplets):
        """Compute triplet loss"""
        if not triplets:
            return torch.tensor(0.0, device=embeddings.device)
        
        dist_matrix = self.pairwise_distance(embeddings)
        losses = []
        
        for a, p, n in triplets:
            pos_dist = dist_matrix[a, p]
            neg_dist = dist_matrix[a, n]
            loss = F.relu(pos_dist - neg_dist + self.margin)
            losses.append(loss)
        
        return torch.stack(losses).mean()