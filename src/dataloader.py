import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class FPHADataset(Dataset):
    def __init__(self, data_root, processor, subject_id='Subject_1', num_frames=16):
        self.data_root = data_root
        self.processor = processor
        self.num_frames = num_frames
        
        self.samples = []
        
        info_file_path = os.path.join(self.data_root, f'Subjects_info/{subject_id}_info.txt')
        video_base_path = os.path.join(self.data_root, f'Video_files/{subject_id}')

        with open(info_file_path, 'r') as f:
            lines = f.readlines()
            print(f"Loaded {len(lines)-3} samples from {info_file_path}")

        for line in lines[3:]:
            parts = line.strip().split()
            if len(parts) < 2:
                continue
            
            action_name = parts[0]
            instance_number = parts[1]
            action_label = action_name
            action_folder_path = os.path.join(video_base_path, action_name, instance_number)
            
            if os.path.isdir(os.path.join(action_folder_path, 'color')):
                self.samples.append((action_folder_path, action_label))

    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):
        action_folder_path, action_label = self.samples[idx]
        
        color_folder_path = os.path.join(action_folder_path, 'color')
        
        frame_files = sorted(os.listdir(color_folder_path))
        total_frames = len(frame_files)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for i in indices:
            frame_path = os.path.join(color_folder_path, frame_files[i])
            frame = cv2.imread(frame_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        pixel_values = self.processor(images=frames, return_tensors="pt")["pixel_values"]
        text_prompt = f"a video of a hand performing the action of {action_label}"
        
        return {
            "pixel_values": pixel_values,
            "text": text_prompt
        }