import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPModel

class BaselineModel(nn.Module):
    def __init__(self, freeze_clip=True, use_adapters=True, adapter_dim=256):
        """
        Args:
            freeze_clip (bool): Whether to freeze CLIP's pre-trained parameters
            use_adapters (bool): Whether to use adapter layers
            adapter_dim (int): Hidden dimension for adapter layers
        """
        super().__init__()
        
        # Load the pre-trained CLIP model from Hugging Face
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        
        # --- Freeze CLIP parameters ---
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
            print("✓ CLIP parameters frozen")
        
        # --- Add trainable adapter layers ---
        # CLIP's vision and text embeddings are 512-dimensional for base models
        self.feature_dim = self.clip.config.projection_dim  # 512 for base CLIP
        
        if use_adapters:
            # Video adapter: processes video features after frame averaging
            self.video_adapter = nn.Sequential(
                nn.Linear(self.feature_dim, adapter_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(adapter_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim)
            )
            
            # Text adapter: processes text features
            self.text_adapter = nn.Sequential(
                nn.Linear(self.feature_dim, adapter_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(adapter_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim)
            )
            
            # Temporal adapter: processes frame sequences before averaging
            # This helps capture temporal information across frames
            self.temporal_adapter = nn.Sequential(
                nn.Linear(self.feature_dim, adapter_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(adapter_dim, self.feature_dim)
            )
            
            print(f"✓ Adapter layers initialized (dim={adapter_dim})")
        else:
            self.video_adapter = nn.Identity()
            self.text_adapter = nn.Identity()
            self.temporal_adapter = nn.Identity()
        
        self.use_adapters = use_adapters

    def forward(self, pixel_values, input_ids, attention_mask):
        """
        The forward pass of the model.
        
        Args:
            pixel_values: The processed video frames tensor (batch_size, num_frames, C, H, W)
            input_ids: The tokenized text tensor (batch_size, seq_len)
            attention_mask: The attention mask for the text (batch_size, seq_len)
            
        Returns:
            video_features: Video embeddings (batch_size, feature_dim)
            text_features: Text embeddings (batch_size, feature_dim)
        """
        
        # The CLIP model's vision_model expects a batch dimension, but our video frames
        # already have a dimension for the number of frames. We need to merge them.
        batch_size, num_frames, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_frames, channels, height, width)
        
        # Get the features (embeddings) from each encoder
        # Shape: (batch_size * num_frames, feature_dim)
        frame_features = self.clip.get_image_features(pixel_values=pixel_values)
        
        # Reshape to separate batch and frame dimensions
        # Shape: (batch_size, num_frames, feature_dim)
        frame_features = frame_features.view(batch_size, num_frames, -1)
        
        # Apply temporal adapter to capture temporal information across frames
        if self.use_adapters:
            # Process each frame through temporal adapter
            frame_features = self.temporal_adapter(frame_features)
        
        # Average across frames to get a single video representation
        # Shape: (batch_size, feature_dim)
        video_features = frame_features.mean(dim=1)
        
        # Apply video adapter
        if self.use_adapters:
            video_features = self.video_adapter(video_features)
        
        # Get text features
        # Shape: (batch_size, feature_dim)
        text_features = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        
        # Apply text adapter
        if self.use_adapters:
            text_features = self.text_adapter(text_features)
        
        return video_features, text_features
    
    def get_trainable_params(self):
        """
        Returns the number of trainable parameters in the model.
        Useful for checking if freezing worked correctly.
        """
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        
        return trainable_params, total_params


class UCMR_Block(nn.Module):
    """
    Implements the full Unsymmetrical Cross-Modality Refinement Block (UCMR)
    with multi-head attention mechanisms for P2V-R and V2P-R.
    """
    def __init__(self, feature_dim=512, num_heads=8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.head_dim = feature_dim // num_heads
        
        # --- Layers for P2V-R (Pose-guided Video Self-Attention) ---
        self.to_q_video = nn.Linear(feature_dim, feature_dim)
        self.to_k_video = nn.Linear(feature_dim, feature_dim)
        self.to_v_video = nn.Linear(feature_dim, feature_dim)
        
        # This layer creates the pose "spotlight"
        # It converts pose features into an attention bias
        self.pose_to_bias = nn.Linear(feature_dim, num_heads)
        
        # --- Layers for V2P-R (Video-to-Pose Cross-Attention) ---
        # Q comes from POSE, K and V come from VIDEO
        self.to_q_pose_cross = nn.Linear(feature_dim, feature_dim)
        self.to_k_video_cross = nn.Linear(feature_dim, feature_dim)
        self.to_v_video_cross = nn.Linear(feature_dim, feature_dim)

        # Output linear layers
        self.out_video = nn.Linear(feature_dim, feature_dim)
        self.out_pose = nn.Linear(feature_dim, feature_dim)
        
        self.tau = nn.Parameter(torch.tensor(0.07)) # Learnable temperature
        print("✓ Full UCMR Block initialized")

    def p2v_refine(self, video_features, pose_features):
        # video/pose features shape: (Batch, Num_Frames, 512)
        B, N_v, C = video_features.shape
        
        # Project Q, K, V for video self-attention
        q = self.to_q_video(video_features).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.to_k_video(video_features).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v_video(video_features).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)

        # --- Create the Pose Bias (The "Spotlight") ---
        # This is a simplification of the paper's "Weighted Mask Module"
        avg_pose = pose_features.mean(dim=1) # (B, C)
        pose_bias = self.pose_to_bias(avg_pose) # (B, num_heads)
        pose_bias = pose_bias.view(B, self.num_heads, 1, 1) # Reshape for broadcasting

        # Calculate attention scores
        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        # Apply the pose bias here
        attn_scores = attn_scores + pose_bias
        
        attn = F.softmax(attn_scores, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N_v, C)
        # Add residual connection
        return self.out_video(out) + video_features

    def v2p_refine(self, video_features, pose_features):
        # Standard cross-attention: Pose is Query, Video is Key/Value
        B, N_v, C_v = video_features.shape
        _, N_p, C_p = pose_features.shape
        
        # Q from pose
        q = self.to_q_pose_cross(pose_features).view(B, N_p, self.num_heads, self.head_dim).transpose(1, 2)
        # K, V from video
        k = self.to_k_video_cross(video_features).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.to_v_video_cross(video_features).view(B, N_v, self.num_heads, self.head_dim).transpose(1, 2)

        attn_scores = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn_scores, dim=-1)
        
        out = (attn @ v).transpose(1, 2).reshape(B, N_p, C_v)
        # Add residual connection
        return self.out_pose(out) + pose_features

    def nt_xent_loss(self, z1, z2):
        # Implementation of Equation 3
        z1 = F.normalize(z1, dim=-1)
        z2 = F.normalize(z2, dim=-1)
        
        sim_matrix = (z1 @ z2.T) / self.tau.exp()
        labels = torch.arange(len(z1)).to(z1.device)
        
        loss_z1 = F.cross_entropy(sim_matrix, labels)
        loss_z2 = F.cross_entropy(sim_matrix.T, labels)
        
        return (loss_z1 + loss_z2) / 2

    def forward(self, video_features, pose_features):
        # video_features, pose_features: (batch_size, num_frames, 512)
        
        # 1. P2V-R (Pose-to-Vision)
        refined_video_seq = self.p2v_refine(video_features, pose_features)
        
        # 2. V2P-R (Vision-to-Pose)
        refined_pose_seq = self.v2p_refine(video_features, pose_features)
        
        # 3. Refinement Supervision Loss
        # Applied on the mean-pooled features
        ucmr_loss = self.nt_xent_loss(video_features.mean(dim=1), refined_video_seq.mean(dim=1)) + \
                    self.nt_xent_loss(pose_features.mean(dim=1), refined_pose_seq.mean(dim=1))

        return refined_video_seq, refined_pose_seq, ucmr_loss


class HPeVLModel(nn.Module):
    """
    The final Pose-Enhanced Vision-Language Model (H-PeVL).
    """
    def __init__(self, freeze_clip=True, adapter_dim=256):
        super().__init__()
        
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
            print("✓ CLIP parameters frozen")

        self.feature_dim = self.clip.config.projection_dim # 512
        
        # --- Adapters ---
        self.video_adapter = nn.Sequential(
            nn.Linear(self.feature_dim, adapter_dim),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(adapter_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
        self.text_adapter = nn.Sequential(
            nn.Linear(self.feature_dim, adapter_dim),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(adapter_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
        # This is the new pose adapter you added
        self.pose_adapter = nn.Sequential(
            nn.Linear(self.feature_dim, adapter_dim),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(adapter_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim)
        )
        # We'll use a temporal adapter for the pose sequence too
        self.pose_temporal_adapter = nn.Sequential(
            nn.Linear(self.feature_dim, adapter_dim),
            nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(adapter_dim, self.feature_dim)
        )
        
        # --- Pose Encoder (projects 64-dim to 512-dim) ---
        self.pose_encoder = nn.Sequential(
            nn.Linear(64, adapter_dim), # 64 = 1 frame num + 21 joints * 3 coords
            nn.ReLU(),
            nn.Linear(adapter_dim, self.feature_dim)
        )
        
        # --- NEW: Use the Full UCMR Block ---
        self.ucmr = UCMR_Block(self.feature_dim)
        
        print(f"✓ H-PeVL Model Initialized (dim={adapter_dim})")

    def forward(self, pixel_values, pose_values, input_ids, attention_mask):
        # 1. Get Video Features (as a sequence)
        batch_size, num_frames, c, h, w = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_frames, c, h, w)
        # frame_features shape: (Batch, Num_Frames, 512)
        frame_features = self.clip.get_image_features(pixel_values=pixel_values).view(batch_size, num_frames, -1)
        
        # 2. Text Features (as a single vector)
        # text_features shape: (Batch, 512)
        text_features = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        
        # 3. Pose Features (as a sequence)
        # pose_features shape: (Batch, Num_Frames, 512)
        pose_features = self.pose_encoder(pose_values)

        # 4. Pass frame sequences through UCMR Block for refinement
        # refined_video_seq shape: (Batch, Num_Frames, 512)
        # refined_pose_seq shape: (Batch, Num_Frames, 512)
        refined_video_seq, refined_pose_seq, ucmr_loss = self.ucmr(frame_features, pose_features)
        
        # 5. Get final single-vector representations by averaging the *refined* sequences
        refined_video_features = refined_video_seq.mean(dim=1)
        refined_pose_features = refined_pose_seq.mean(dim=1)

        # 6. Apply final Adapters
        refined_video_features = self.video_adapter(refined_video_features)
        refined_pose_features = self.pose_adapter(refined_pose_features)
        text_features = self.text_adapter(text_features)
        
        # Return the final features and the UCMR loss
        return refined_video_features, refined_pose_features, text_features, ucmr_loss

    def get_trainable_params(self):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")