import os
import numpy as np
import torch
from torch.utils.data import Dataset
import cv2
from glob import glob
import pandas as pd
from src.utils.data_utils import load_and_preprocess_image, augment_image

class EyeImageDataset(Dataset):
    """
    눈 이미지 데이터셋 클래스 (CNN 모델용)
    """
    def __init__(self, data_dir, split='train', target_size=(128, 128), transform=None):
        """
        Args:
            data_dir: 데이터 디렉토리 (open/closed 하위 디렉토리가 있어야 함)
            split: 데이터셋 분할 ('train', 'val', 'test' 중 하나)
            target_size: 이미지 크기
            transform: 추가 변환 함수
        """
        self.data_dir = data_dir
        self.split = split
        self.target_size = target_size
        self.transform = transform
        
        # 클래스 디렉토리
        self.open_dir = os.path.join(data_dir, split, 'open')
        self.closed_dir = os.path.join(data_dir, split, 'closed')
        
        # 파일 목록 가져오기
        self.open_files = glob(os.path.join(self.open_dir, "*.jpg")) + \
                          glob(os.path.join(self.open_dir, "*.png"))
        self.closed_files = glob(os.path.join(self.closed_dir, "*.jpg")) + \
                            glob(os.path.join(self.closed_dir, "*.png"))
        
        # 파일 경로 및 레이블 리스트 생성
        self.files = []
        self.labels = []
        
        for file_path in self.open_files:
            self.files.append(file_path)
            self.labels.append(0)  # 열린 눈 = 0
            
        for file_path in self.closed_files:
            self.files.append(file_path)
            self.labels.append(1)  # 감은 눈 = 1
        
        # 데이터 셔플 (train 데이터에만 적용)
        if split == 'train':
            indices = np.random.permutation(len(self.files))
            self.files = [self.files[i] for i in indices]
            self.labels = [self.labels[i] for i in indices]
        
        print(f"Loaded {len(self.files)} images for {split} split")
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        # 이미지 로드 및 전처리
        img_path = self.files[idx]
        label = self.labels[idx]
        
        # 이미지 로드 및 전처리
        img = load_and_preprocess_image(img_path, self.target_size)
        
        # 훈련 데이터 증강
        if self.split == 'train' and self.transform:
            img = self.transform(img)
        
        # 텐서로 변환 (H, W, C) -> (C, H, W)
        img = torch.from_numpy(img.transpose(2, 0, 1))
        
        return img, torch.tensor(label, dtype=torch.long)

class EyeSequenceDataset(Dataset):
    """
    눈 이미지 시퀀스 데이터셋 클래스 (RNN 모델용)
    """
    def __init__(self, data_dir, sequence_length=10, step=1, split='train', target_size=(128, 128), transform=None):
        """
        Args:
            data_dir: 비디오 프레임 또는 시퀀스 데이터 디렉토리
            sequence_length: 시퀀스 길이
            step: 시퀀스 추출 스텝 크기
            split: 데이터셋 분할 ('train', 'val', 'test' 중 하나)
            target_size: 이미지 크기
            transform: 추가 변환 함수
        """
        self.data_dir = data_dir
        self.sequence_length = sequence_length
        self.step = step
        self.split = split
        self.target_size = target_size
        self.transform = transform
        
        # 시퀀스 메타데이터 (CSV 파일)를 로드
        # 예: video_id, start_frame, end_frame, label 열이 있는 CSV
        metadata_path = os.path.join(data_dir, f"{split}_sequences.csv")
        self.metadata = pd.read_csv(metadata_path)
        
        # 시퀀스 프레임 디렉토리
        self.frames_dir = os.path.join(data_dir, 'frames')
        
        print(f"Loaded {len(self.metadata)} sequences for {split} split")
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, idx):
        # 시퀀스 메타데이터
        seq_data = self.metadata.iloc[idx]
        video_id = seq_data['video_id']
        start_frame = seq_data['start_frame']
        label = seq_data['label']  # 0=열린 눈, 1=깜빡임
        
        # 시퀀스 프레임 로드
        frames = []
        for i in range(self.sequence_length):
            frame_idx = start_frame + i * self.step
            frame_path = os.path.join(self.frames_dir, video_id, f"frame_{frame_idx:06d}.jpg")
            
            # 이미지 로드 및 전처리
            frame = load_and_preprocess_image(frame_path, self.target_size)
            
            # 훈련 데이터 증강 (모든 프레임에 동일한 증강 적용)
            if self.split == 'train' and self.transform and i == 0:
                # 전체 시퀀스에 동일한 변환 적용하기 위해 첫 프레임에만 변환 결정 적용
                transformed_frame = self.transform(frame)
                frames.append(transformed_frame)
            else:
                frames.append(frame)
        
        # 텐서로 변환 (S, H, W, C) -> (S, C, H, W)
        frames = np.array(frames)
        frames = torch.from_numpy(frames.transpose(0, 3, 1, 2))
        
        return frames, torch.tensor(label, dtype=torch.long)

class EARDataset(Dataset):
    """
    Eye Aspect Ratio (EAR) 데이터셋 클래스
    """
    def __init__(self, data_path, sequence_length=10, split='train'):
        """
        Args:
            data_path: EAR 값이 저장된 CSV 파일 경로
            sequence_length: 시퀀스 길이
            split: 데이터셋 분할 ('train', 'val', 'test' 중 하나)
        """
        self.sequence_length = sequence_length
        
        # EAR 값 데이터 로드 (CSV 형식: timestamp, ear_value, blink_label)
        df = pd.read_csv(data_path)
        
        # 데이터 분할
        if split == 'train':
            df = df.iloc[:int(len(df) * 0.7)]
        elif split == 'val':
            df = df.iloc[int(len(df) * 0.7):int(len(df) * 0.9)]
        else:  # test
            df = df.iloc[int(len(df) * 0.9):]
        
        # 시퀀스 및 레이블 생성
        self.sequences = []
        self.labels = []
        
        for i in range(0, len(df) - self.sequence_length, 1):
            # EAR 시퀀스 추출
            seq = df.iloc[i:i+self.sequence_length]['ear_value'].values
            
            # 레이블 - 시퀀스 다음 프레임의 깜빡임 여부
            # (보통 깜빡임은 EAR 값이 특정 임계값 이하로 떨어질 때 감지됨)
            label = df.iloc[i+self.sequence_length]['blink_label']
            
            self.sequences.append(seq)
            self.labels.append(label)
        
        print(f"Loaded {len(self.sequences)} sequences for {split} split")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        # EAR 시퀀스를 텐서로 변환
        seq = self.sequences[idx]
        label = self.labels[idx]
        
        # 텐서 형태: (sequence_length, 1)
        seq_tensor = torch.FloatTensor(seq).view(-1, 1)
        
        return seq_tensor, torch.tensor(label, dtype=torch.long) 