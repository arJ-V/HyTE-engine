"""
HyTE: Hybrid Transformer for Events
Implementation based on the HyTE paper architecture.

This module implements:
- Logic Encoder (Stream 2): Encodes structured market data
- Semantic Encoder (Stream 1): Processes text using DistilRoBERTa
- Bi-Directional Co-Attention: Fuses the two streams
- Gated Unification: Dynamically combines streams
- Disentangled Embedding: Returns [z_sem; z_ent; z_logic]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from typing import Tuple


class FourierFeatureMLP(nn.Module):
    """
    Random Fourier Feature MLP layer for encoding continuous values.
    
    Implements Eq 7 & 8: Transforms scalar inputs using Fourier features
    for better representation of continuous values (threshold, time).
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 256,
        num_fourier_features: int = 64,
        sigma: float = 1.0
    ):
        super().__init__()
        self.num_fourier_features = num_fourier_features
        self.sigma = sigma
        
        # Random Fourier feature projection matrix (Eq 7)
        # Sample from normal distribution and freeze during training
        self.register_buffer(
            'B',
            torch.randn(num_fourier_features, input_dim) * sigma
        )
        
        # MLP to process Fourier features (Eq 8)
        self.mlp = nn.Sequential(
            nn.Linear(num_fourier_features * 2, hidden_dim),  # *2 for sin and cos
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (Batch, 1) - Scalar values (threshold or time)
        
        Returns:
            (Batch, hidden_dim) - Encoded continuous values
        """
        # Project to Fourier space (Eq 7)
        # x: (Batch, 1), B: (num_fourier, 1)
        projected = torch.matmul(x, self.B.t())  # (Batch, num_fourier)
        
        # Apply sin and cos (Eq 8)
        fourier_features = torch.cat([
            torch.sin(2 * torch.pi * projected),
            torch.cos(2 * torch.pi * projected)
        ], dim=-1)  # (Batch, num_fourier * 2)
        
        # MLP transformation
        return self.mlp(fourier_features)


class LogicEncoder(nn.Module):
    """
    Logic Encoder (Stream 2): Encodes structured market data.
    
    Processes:
    - Polarity (0 or 1) - Eq 6
    - Type (discrete market types) - Eq 6
    - Threshold (normalized scalar) - Eq 7 & 8
    - Time (normalized time-to-maturity) - Eq 10
    - Entities (variable-length entity embeddings) - Eq 9
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_polarity_types: int = 2,  # 0 or 1
        num_market_types: int = 10,  # Binary, Scalar, etc.
        entity_dim: int = 128,  # d_ent
        num_fourier_features: int = 64
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.entity_dim = entity_dim
        
        # Polarity and Type embeddings (Eq 6)
        self.polarity_embedding = nn.Embedding(num_polarity_types, hidden_dim)
        self.type_embedding = nn.Embedding(num_market_types, hidden_dim)
        
        # Fourier Feature MLPs for continuous values
        self.threshold_encoder = FourierFeatureMLP(
            input_dim=1,
            hidden_dim=hidden_dim,
            num_fourier_features=num_fourier_features
        )
        self.time_encoder = FourierFeatureMLP(
            input_dim=1,
            hidden_dim=hidden_dim,
            num_fourier_features=num_fourier_features
        )
        
        # Entity Aggregation: Attention mechanism (Eq 9)
        # Learned query vector to aggregate variable-length entity list
        self.entity_query = nn.Parameter(torch.randn(1, 1, hidden_dim))
        self.entity_key_proj = nn.Linear(entity_dim, hidden_dim)
        self.entity_value_proj = nn.Linear(entity_dim, hidden_dim)
        self.entity_attention_norm = nn.LayerNorm(hidden_dim)
        
        # Projection to ensure all components have same hidden_dim
        self.entity_proj = nn.Linear(entity_dim, hidden_dim)
    
    def forward(
        self,
        polarity_idx: torch.Tensor,
        type_idx: torch.Tensor,
        threshold_val: torch.Tensor,
        time_val: torch.Tensor,
        entity_embs: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            polarity_idx: (Batch) - 0 or 1
            type_idx: (Batch) - Discrete market types
            threshold_val: (Batch, 1) - Normalized scalar (e.g., strike price)
            time_val: (Batch, 1) - Normalized time-to-maturity
            entity_embs: (Batch, K, d_ent) - Pre-fetched entity embeddings
        
        Returns:
            H_struct: (Batch, 4, hidden_dim) - Stacked logic encodings
        """
        batch_size = polarity_idx.size(0)
        
        # Polarity and Type embeddings (Eq 6)
        h_polarity = self.polarity_embedding(polarity_idx)  # (Batch, hidden_dim)
        h_type = self.type_embedding(type_idx)  # (Batch, hidden_dim)
        
        # Fourier encoding for threshold (Eq 7 & 8)
        h_threshold = self.threshold_encoder(threshold_val)  # (Batch, hidden_dim)
        
        # Fourier encoding for time (Eq 10)
        h_time = self.time_encoder(time_val)  # (Batch, hidden_dim)
        
        # Entity Aggregation with attention (Eq 9)
        # Project entities to hidden_dim
        K = entity_embs.size(1)  # Number of entities
        entity_proj = self.entity_proj(entity_embs)  # (Batch, K, hidden_dim)
        
        # Create query, key, value
        query = self.entity_query.expand(batch_size, -1, -1)  # (Batch, 1, hidden_dim)
        key = self.entity_key_proj(entity_embs)  # (Batch, K, hidden_dim)
        value = self.entity_value_proj(entity_embs)  # (Batch, K, hidden_dim)
        
        # Scaled dot-product attention
        attention_scores = torch.matmul(query, key.transpose(-2, -1))  # (Batch, 1, K)
        attention_scores = attention_scores / (self.hidden_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)  # (Batch, 1, K)
        
        # Aggregate entities
        h_entity = torch.matmul(attention_weights, value)  # (Batch, 1, hidden_dim)
        h_entity = h_entity.squeeze(1)  # (Batch, hidden_dim)
        
        # Combine attention-aggregated with projected mean (residual connection)
        # This ensures robust aggregation even if attention weights are uniform
        h_entity = h_entity + entity_proj.mean(dim=1)
        h_entity = self.entity_attention_norm(h_entity)  # (Batch, hidden_dim)
        
        # Stack 4 components as specified: [polarity, type, threshold, entity]
        # Combine threshold and time into a single temporal/numerical component
        # This respects the 4-vector specification while incorporating both values
        h_temporal = (h_threshold + h_time) / 2  # (Batch, hidden_dim)
        
        H_struct = torch.stack([h_polarity, h_type, h_temporal, h_entity], dim=1)
        # H_struct: (Batch, 4, hidden_dim)
        
        return H_struct


class CoAttention(nn.Module):
    """
    Bi-Directional Co-Attention mechanism.
    
    Implements Eq 11, 12, 13: Computes affinity matrix and attention maps
    for cross-stream attention between text and logic encodings.
    """
    
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learned projection matrix W for affinity computation (Eq 11)
        self.affinity_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        H_text: torch.Tensor,
        H_struct: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            H_text: (Batch, L_text, hidden_dim) - Text encodings
            H_struct: (Batch, L_struct, hidden_dim) - Logic encodings
                where L_struct = 4 (polarity, type, threshold, entity)
        
        Returns:
            A_text_to_struct: (Batch, L_text, L_struct) - Text attending to Logic
            A_struct_to_text: (Batch, L_struct, L_text) - Logic attending to Text
        """
        # Compute affinity matrix C (Eq 11): H_text W H_struct^T
        # W projects H_text to compute compatibility with H_struct
        H_text_proj = self.affinity_proj(H_text)  # (Batch, L_text, hidden_dim)
        
        # Affinity matrix: (Batch, L_text, hidden_dim) @ (Batch, hidden_dim, L_struct)
        C = torch.matmul(H_text_proj, H_struct.transpose(-2, -1))  # (Batch, L_text, L_struct)
        
        # Attention from text to struct (Eq 12): A_text→struct
        # Text tokens attend to logic components
        A_text_to_struct = F.softmax(C, dim=-1)  # (Batch, L_text, L_struct)
        
        # Attention from struct to text (Eq 13): A_struct→text
        # Logic components attend to text tokens
        A_struct_to_text = F.softmax(C.transpose(-2, -1), dim=-1)  # (Batch, L_struct, L_text)
        
        return A_text_to_struct, A_struct_to_text


class HyTEModel(nn.Module):
    """
    HyTE: Hybrid Transformer for Events
    
    Main model that combines:
    - Semantic Encoder (Stream 1): DistilRoBERTa for text
    - Logic Encoder (Stream 2): Structured data encoder
    - Co-Attention: Cross-stream attention
    - Gated Unification: Dynamic stream combination
    - Disentangled Embedding: Returns [z_sem; z_ent; z_logic] (Eq 4)
    """
    
    def __init__(
        self,
        hidden_dim: int = 256,
        num_polarity_types: int = 2,
        num_market_types: int = 10,
        entity_dim: int = 128,
        freeze_text_encoder: bool = False,
        roberta_model_name: str = "distilroberta-base"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.freeze_text_encoder = freeze_text_encoder
        
        # Semantic Encoder (Stream 1): DistilRoBERTa
        self.text_encoder = AutoModel.from_pretrained(roberta_model_name)
        roberta_hidden_dim = self.text_encoder.config.hidden_size  # 768 for distilroberta-base
        
        # Projection from RoBERTa output to hidden_dim
        self.text_projection = nn.Sequential(
            nn.Linear(roberta_hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Freeze text encoder if requested
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        
        # Logic Encoder (Stream 2)
        self.logic_encoder = LogicEncoder(
            hidden_dim=hidden_dim,
            num_polarity_types=num_polarity_types,
            num_market_types=num_market_types,
            entity_dim=entity_dim
        )
        
        # Co-Attention mechanism
        self.co_attention = CoAttention(hidden_dim=hidden_dim)
        
        # Gated Unification (Eq 14, 15)
        # Gating network to combine streams dynamically
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()  # Output is gate weight λ
        )
        
        # Stream fusion layers
        self.text_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        self.struct_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Output projections for disentangled embeddings (Eq 4)
        # z_sem: semantic embedding
        # z_ent: entity embedding (from logic stream)
        # z_logic: logic embedding (from structured data)
        self.sem_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ent_proj = nn.Linear(hidden_dim, hidden_dim)
        self.logic_proj = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(
        self,
        text_input_ids: torch.Tensor,
        text_attention_mask: torch.Tensor,
        polarity_idx: torch.Tensor,
        type_idx: torch.Tensor,
        threshold_val: torch.Tensor,
        time_val: torch.Tensor,
        entity_embs: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            text_input_ids: (Batch, L_text) - Tokenized text input
            text_attention_mask: (Batch, L_text) - Attention mask for text
            polarity_idx: (Batch) - 0 or 1
            type_idx: (Batch) - Discrete market types
            threshold_val: (Batch, 1) - Normalized scalar
            time_val: (Batch, 1) - Normalized time-to-maturity
            entity_embs: (Batch, K, entity_dim) - Entity embeddings
        
        Returns:
            z_sem: (Batch, hidden_dim) - Semantic embedding
            z_ent: (Batch, hidden_dim) - Entity embedding
            z_logic: (Batch, hidden_dim) - Logic embedding
            disentangled: (Batch, hidden_dim * 3) - Combined [z_sem; z_ent; z_logic] (Eq 4)
        """
        batch_size = text_input_ids.size(0)
        
        # ========== Stream 1: Semantic Encoder ==========
        # Encode text with DistilRoBERTa
        text_outputs = self.text_encoder(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask
        )
        
        # Get pooled representation (use CLS token or mean pooling)
        # DistilRoBERTa doesn't have pooler, so we use the first token or mean
        if hasattr(text_outputs, 'pooler_output') and text_outputs.pooler_output is not None:
            text_pooled = text_outputs.pooler_output
        else:
            # Mean pooling over sequence length
            text_pooled = (text_outputs.last_hidden_state * 
                          text_attention_mask.unsqueeze(-1)).sum(dim=1) / \
                          text_attention_mask.sum(dim=1, keepdim=True)
        
        # Project to hidden_dim
        text_repr = self.text_projection(text_pooled)  # (Batch, hidden_dim)
        
        # Get full sequence for co-attention
        text_seq = self.text_projection(text_outputs.last_hidden_state)  # (Batch, L_text, hidden_dim)
        
        # ========== Stream 2: Logic Encoder ==========
        H_struct = self.logic_encoder(
            polarity_idx=polarity_idx,
            type_idx=type_idx,
            threshold_val=threshold_val,
            time_val=time_val,
            entity_embs=entity_embs
        )  # (Batch, 4, hidden_dim)
        
        # ========== Co-Attention ==========
        A_text_to_struct, A_struct_to_text = self.co_attention(
            H_text=text_seq,
            H_struct=H_struct
        )
        # A_text_to_struct: (Batch, L_text, 4)
        # A_struct_to_text: (Batch, 4, L_text)
        
        # Apply attention to get attended representations
        # Text attending to struct: aggregate text tokens weighted by attention to struct
        text_attended_to_struct = torch.matmul(A_text_to_struct, H_struct)  # (Batch, L_text, hidden_dim)
        text_attended = text_attended_to_struct.mean(dim=1)  # (Batch, hidden_dim)
        
        # Struct attending to text: aggregate struct components weighted by attention to text
        struct_attended_to_text = torch.matmul(A_struct_to_text, text_seq)  # (Batch, 4, hidden_dim)
        struct_attended = struct_attended_to_text.mean(dim=1)  # (Batch, hidden_dim)
        
        # ========== Gated Unification (Eq 14, 15) ==========
        # Compute gate weights λ
        gate_input = torch.cat([text_repr, struct_attended], dim=-1)  # (Batch, hidden_dim * 2)
        lambda_gate = self.gate_network(gate_input)  # (Batch, hidden_dim) - Eq 14
        
        # Apply fusion layers
        text_fused = self.text_fusion(text_attended)  # (Batch, hidden_dim)
        struct_fused = self.struct_fusion(struct_attended)  # (Batch, hidden_dim)
        
        # Weighted combination (Eq 15)
        unified = lambda_gate * text_fused + (1 - lambda_gate) * struct_fused  # (Batch, hidden_dim)
        
        # ========== Disentangled Embeddings (Eq 4) ==========
        # Extract components for disentangled representation
        # z_sem: Semantic component (from text stream)
        z_sem = self.sem_proj(unified)  # (Batch, hidden_dim)
        
        # z_ent: Entity component (from entity aggregation in logic stream)
        # Get entity representation from H_struct (the 4th component is entity)
        h_entity_component = H_struct[:, 3, :]  # (Batch, hidden_dim) - entity component
        z_ent = self.ent_proj(h_entity_component)  # (Batch, hidden_dim)
        
        # z_logic: Logic component (from structured data stream)
        # Combine structured components (polarity, type, threshold)
        h_logic_components = H_struct[:, :3, :].mean(dim=1)  # (Batch, hidden_dim)
        z_logic = self.logic_proj(h_logic_components)  # (Batch, hidden_dim)
        
        # Return disentangled embeddings
        # The full disentangled embedding is [z_sem; z_ent; z_logic]
        disentangled = torch.cat([z_sem, z_ent, z_logic], dim=-1)  # (Batch, hidden_dim * 3)
        
        return z_sem, z_ent, z_logic, disentangled


def create_model(
    hidden_dim: int = 256,
    num_market_types: int = 10,
    entity_dim: int = 128,
    freeze_text_encoder: bool = False,
    roberta_model_name: str = "distilroberta-base"
) -> HyTEModel:
    """
    Factory function to create a HyTE model with default parameters.
    
    Args:
        hidden_dim: Hidden dimension for both streams
        num_market_types: Number of discrete market types
        entity_dim: Dimension of pre-computed entity embeddings
        freeze_text_encoder: Whether to freeze DistilRoBERTa weights
        roberta_model_name: HuggingFace model name
    
    Returns:
        HyTEModel instance
    """
    return HyTEModel(
        hidden_dim=hidden_dim,
        num_polarity_types=2,
        num_market_types=num_market_types,
        entity_dim=entity_dim,
        freeze_text_encoder=freeze_text_encoder,
        roberta_model_name=roberta_model_name
    )
