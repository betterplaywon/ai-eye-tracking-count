import torch
import torch.nn as nn
import torch.nn.functional as F

class EyeFeatureExtractor(nn.Module):
    """
    눈 이미지에서 특징을 추출하는 CNN 모델
    """
    def __init__(self, input_channels=3):
        super(EyeFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # x 형태: (배치 크기, 채널, 높이, 너비)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.dropout(x)
        
        # 특징 맵을 평평하게 만들지 않고 형태 유지
        return x

class BlinkDetectionCNN(nn.Module):
    """
    단일 이미지 프레임에서 눈 깜빡임을 검출하는 CNN 모델
    """
    def __init__(self, input_channels=3, input_size=(128, 128)):
        super(BlinkDetectionCNN, self).__init__()
        self.feature_extractor = EyeFeatureExtractor(input_channels)
        
        # 특징 크기 계산 (input_size가 (h, w)인 경우 -> (h/8, w/8)로 축소됨)
        h, w = input_size
        feature_size = 128 * (h // 8) * (w // 8)  # 128은 마지막 컨볼루션 레이어의 출력 채널 수
        
        self.fc1 = nn.Linear(feature_size, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 2)  # 2 클래스: 깜빡임 / 비깜빡임
        
    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)  # 1번 차원 이후로 평평하게 만듦
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.5, training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class BlinkDetectionRNN(nn.Module):
    """
    시퀀스 데이터로 눈 깜빡임을 검출하는 CNN+LSTM 모델
    """
    def __init__(self, input_channels=3, input_size=(128, 128), hidden_size=256, num_layers=2, sequence_length=10):
        super(BlinkDetectionRNN, self).__init__()
        self.feature_extractor = EyeFeatureExtractor(input_channels)
        
        # 특징 크기 계산
        h, w = input_size
        feature_size = 128 * (h // 8) * (w // 8)
        
        # LSTM 계층
        self.lstm = nn.LSTM(
            input_size=feature_size,  # CNN에서 추출한 특징의 크기
            hidden_size=hidden_size,  # LSTM 은닉 상태의 크기
            num_layers=num_layers,    # LSTM 계층 수
            batch_first=True,         # 입력 텐서의 첫 번째 차원이 배치 크기
            dropout=0.3 if num_layers > 1 else 0  # 다중 계층 시 dropout 적용
        )
        
        # 최종 분류 계층
        self.fc = nn.Linear(hidden_size, 2)  # 2 클래스: 깜빡임 / 비깜빡임
        
    def forward(self, x):
        # x 형태: (배치 크기, 시퀀스 길이, 채널, 높이, 너비)
        batch_size, seq_len, c, h, w = x.size()
        
        # CNN 특징 추출을 위해 텐서 변환
        # (batch_size * seq_len, c, h, w)로 재구성
        x = x.view(batch_size * seq_len, c, h, w)
        x = self.feature_extractor(x)
        
        # 특징을 평평하게 만들기
        feature_size = x.size(1) * x.size(2) * x.size(3)
        x = x.view(batch_size * seq_len, feature_size)
        
        # (batch_size, seq_len, feature_size)로 재구성
        x = x.view(batch_size, seq_len, -1)
        
        # LSTM 계층 통과
        # output 형태: (batch_size, seq_len, hidden_size)
        output, (hidden, cell) = self.lstm(x)
        
        # 마지막 타임 스텝의 출력을 사용하여 분류
        # (batch_size, hidden_size)
        x = output[:, -1, :]
        
        # 최종 분류
        x = self.fc(x)
        return x

class EyeAspectRatioDetector(nn.Module):
    """
    눈 종횡비(EAR) 값을 입력으로 받아 눈 깜빡임을 감지하는 모델
    """
    def __init__(self, sequence_length=10, hidden_size=64, num_layers=1):
        super(EyeAspectRatioDetector, self).__init__()
        
        # LSTM 계층
        self.lstm = nn.LSTM(
            input_size=1,  # EAR은 단일 값
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 완전 연결 계층
        self.fc = nn.Linear(hidden_size, 2)  # 2 클래스: 깜빡임 / 비깜빡임
        
    def forward(self, x):
        # x 형태: (배치 크기, 시퀀스 길이, 1)
        output, (hidden, cell) = self.lstm(x)
        
        # 마지막 타임 스텝의 출력 사용
        x = output[:, -1, :]
        
        # 분류 계층
        x = self.fc(x)
        return x 