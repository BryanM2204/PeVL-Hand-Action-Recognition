import torch
import torch.nn as nn
from transformers import CLIPModel

class BaselineModel(nn.Module):
    def __init__(self, freeze_clip=True, use_adapters=True, adapter_dim=256):

        super().__init__()
        
        self.clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
        
        if freeze_clip:
            for param in self.clip.parameters():
                param.requires_grad = False
            print("✓ CLIP parameters frozen")
    
        self.feature_dim = self.clip.config.projection_dim
        
        if use_adapters:
            self.video_adapter = nn.Sequential(
                nn.Linear(self.feature_dim, adapter_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(adapter_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim)
            )
            
            self.text_adapter = nn.Sequential(
                nn.Linear(self.feature_dim, adapter_dim),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(adapter_dim, self.feature_dim),
                nn.LayerNorm(self.feature_dim)
            )
            
        
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
        batch_size, num_frames, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.view(batch_size * num_frames, channels, height, width)
        
      
        frame_features = self.clip.get_image_features(pixel_values=pixel_values)
        
       
        frame_features = frame_features.view(batch_size, num_frames, -1)
      
        if self.use_adapters:
            frame_features = self.temporal_adapter(frame_features)
        
        
        video_features = frame_features.mean(dim=1)
        
      
        if self.use_adapters:
            video_features = self.video_adapter(video_features)
       
        text_features = self.clip.get_text_features(input_ids=input_ids, attention_mask=attention_mask)
        
        
        if self.use_adapters:
            text_features = self.text_adapter(text_features)
        
        return video_features, text_features
    
    def get_trainable_params(self):
       
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(f"Percentage trainable: {100 * trainable_params / total_params:.2f}%")
        
        return trainable_params, total_params