import os
import cv2 # The OpenCV library for video processing
import torch
from torch.utils.data import Dataset
import numpy as np

class FPHADataset(Dataset):
    # In your FPHADataset class in src/dataloader.py

    def __init__(self, data_root, processor, subjects=['Subject_1', 'Subject_2', 'Subject_3', 'Subject_4', 'Subject_5', 'Subject_6'], num_frames=16, use_pose=False):
        """
        Args:
            data_root (str): Path to the root FPHA dataset folder.
            processor (CLIPProcessor): The Hugging Face processor for CLIP.
            subject_id (str): The subject ID to load data for (e.g., 'Subject_1').
            num_frames (int): The number of frames to sample from each video.
        """
        self.data_root = data_root
        self.processor = processor
        self.num_frames = num_frames
        self.use_pose = use_pose
        self.samples = []

        # --- This is the new parsing logic ---
        for subject_id in subjects:
          # Construct the path to the info file and the video directory for the subject
          info_file_path = os.path.join(self.data_root, f'Subjects_info/{subject_id}_info.txt')
          video_base_path = os.path.join(self.data_root, f'Video_files/{subject_id}')

          # Open and read the info file
          with open(info_file_path, 'r') as f:
              lines = f.readlines()
              print(f"Loaded {len(lines)-3} samples from {info_file_path}")

          # Skip the header lines and process each sample line
          for line in lines[3:]: # Start from the 4th line
              parts = line.strip().split()
              if len(parts) < 2:
                  continue # Skip empty lines
              
              action_name = parts[0]
              instance_number = parts[1]
              num_frames_in_video = int(parts[2]) # Get the frame count
                
              # Fix for IndexError: SKIP if the video has 0 frames
              if num_frames_in_video == 0:
                  continue
              
              # This is the action label
              action_label = action_name
              
              # This constructs the path to the folder containing the image frames
              action_folder_path = os.path.join(video_base_path, action_name, instance_number)
              
              # Check if the directory actually exists before adding it
              if os.path.isdir(os.path.join(action_folder_path, 'color')):
                  # Append the folder path and its label to our list of samples
                  self.samples.append((action_folder_path, action_label))

    def __len__(self):
        # This returns the total number of samples in the dataset.
        return len(self.samples)


    def __getitem__(self, idx):
        action_folder_path, action_label = self.samples[idx]
        
        color_folder_path = os.path.join(action_folder_path, 'color')
        
        # 1. Video Processing (Revised)
        frame_files = sorted(os.listdir(color_folder_path))
        total_frames = len(frame_files)
        indices = np.linspace(0, total_frames - 1, self.num_frames, dtype=int)
        
        frames = []
        for i in indices:
            frame_path = os.path.join(color_folder_path, frame_files[i])
            frame = cv2.imread(frame_path)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        
        # Process the frames with the processor
        # Note: We are NOT returning a tensor yet, just the list of images
        pixel_values = self.processor(images=frames, return_tensors="pt")["pixel_values"]

        # Return the RAW text prompt, not the tokenized version
        text_prompt = f"a video of a hand performing the action of {action_label}"
        item = {
          "pixel_values": pixel_values,
          "text": text_prompt
        }

        if self.use_pose:
          #New pose processing
          pose_folder_path = action_folder_path.replace("Video_files", "Hand_pose_annotation_v1", 1)
          pose_file_path = os.path.join(pose_folder_path, 'skeleton.txt')
          all_pose_data = np.loadtxt(pose_file_path)
          sampled_pose_data = all_pose_data[indices]
          item["pose_values"] = torch.from_numpy(sampled_pose_data).float()

        
        
        return item