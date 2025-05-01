import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split

def create_dataset_from_video(video_path, output_dir, sample_rate=1, max_frames=None):
    """
    비디오에서 프레임을 추출하여 데이터셋을 생성합니다.
    
    Args:
        video_path: 비디오 파일 경로
        output_dir: 추출된 프레임을 저장할 디렉토리
        sample_rate: 몇 프레임마다 샘플링할지 결정 (기본값: 1 = 모든 프레임)
        max_frames: 최대 추출할 프레임 수 (기본값: None = 제한 없음)
    
    Returns:
        추출된 프레임 수
    """
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 비디오 캡처 객체 생성
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_count % sample_rate == 0:
            # 프레임 저장
            frame_filename = os.path.join(output_dir, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
            
            if max_frames is not None and saved_count >= max_frames:
                break
                
        frame_count += 1
        
    cap.release()
    return saved_count

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    """
    이미지를 로드하고 전처리합니다.
    
    Args:
        image_path: 이미지 파일 경로
        target_size: 조정할 이미지 크기 (width, height)
    
    Returns:
        전처리된 이미지
    """
    # 이미지 로드
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    # 크기 조정
    img = cv2.resize(img, target_size)
    
    # BGR -> RGB 변환
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # 정규화 (0-1 범위로)
    img = img.astype(np.float32) / 255.0
    
    return img

def augment_image(image):
    """
    이미지 증강 기법을 적용합니다.
    
    Args:
        image: 입력 이미지
    
    Returns:
        증강된 이미지
    """
    # 랜덤 밝기 변화
    brightness = np.random.uniform(0.8, 1.2)
    image = np.clip(image * brightness, 0, 1)
    
    # 랜덤 회전 (약간만)
    rows, cols = image.shape[:2]
    angle = np.random.uniform(-10, 10)
    M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
    image = cv2.warpAffine(image, M, (cols, rows))
    
    # 좌우 반전 (50% 확률)
    if np.random.random() > 0.5:
        image = cv2.flip(image, 1)
    
    return image

def prepare_sequences(data, sequence_length, step=1):
    """
    시계열 데이터를 시퀀스로 변환합니다.
    
    Args:
        data: 특징 벡터 목록 또는 배열
        sequence_length: 각 시퀀스의 길이
        step: 시퀀스 생성 시 건너뛸 스텝 수
        
    Returns:
        (X, y) 튜플 - X는 입력 시퀀스, y는 다음 값
    """
    X, y = [], []
    
    for i in range(0, len(data) - sequence_length, step):
        X.append(data[i:i + sequence_length])
        y.append(data[i + sequence_length])
    
    return np.array(X), np.array(y)

def split_dataset(features, labels, test_size=0.2, val_size=0.2, random_state=42):
    """
    데이터셋을 훈련, 검증, 테스트 세트로 분할합니다.
    
    Args:
        features: 특징 벡터
        labels: 레이블
        test_size: 테스트 세트의 비율
        val_size: 검증 세트의 비율 (훈련 세트에서 분리)
        random_state: 랜덤 시드
        
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test) 튜플
    """
    # 먼저 테스트 세트 분리
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        features, labels, test_size=test_size, random_state=random_state
    )
    
    # 훈련 세트에서 검증 세트 분리
    val_ratio = val_size / (1 - test_size)  # 남은 데이터 중에서의 비율 계산
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_ratio, random_state=random_state
    )
    
    return X_train, X_val, X_test, y_train, y_val, y_test 