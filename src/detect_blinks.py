import os
import argparse
import cv2
import numpy as np
import torch
import time
import collections
from datetime import datetime

# 로컬 모듈 임포트
from src.models.blink_detection_model import BlinkDetectionCNN, BlinkDetectionRNN, EyeAspectRatioDetector
from src.utils.face_utils import calculate_eye_aspect_ratio, extract_eye_features

def load_model(model_path, model_type='cnn', device='cpu'):
    """
    학습된 모델을 로드합니다.
    """
    if model_type == 'cnn':
        model = BlinkDetectionCNN(input_channels=3, input_size=(128, 128))
    elif model_type == 'rnn':
        model = BlinkDetectionRNN(input_channels=3, input_size=(128, 128))
    elif model_type == 'ear':
        model = EyeAspectRatioDetector()
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    # 모델 가중치 로드
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    return model

def preprocess_frame(frame, target_size=(128, 128)):
    """
    비디오 프레임을 전처리하여 모델 입력용으로 변환합니다.
    """
    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # 크기 조정
    frame_resized = cv2.resize(frame_rgb, target_size)
    
    # 정규화 (0-1 범위로)
    frame_normalized = frame_resized.astype(np.float32) / 255.0
    
    # PyTorch 텐서로 변환 (C, H, W)
    frame_tensor = torch.from_numpy(frame_normalized.transpose(2, 0, 1)).unsqueeze(0)
    
    return frame_tensor

def detect_blinks_cnn(model, frame, face_cascade, device='cpu', target_size=(128, 128)):
    """
    단일 프레임에서 CNN 모델을 사용하여 눈 깜빡임을 감지합니다.
    """
    # grayscale 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 얼굴 검출
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,  # 더 작은 값 (원래 1.3)
        minNeighbors=3,   # 더 작은 값 (원래 5)
        minSize=(50, 50)  # 최소 얼굴 크기 지정
    )
    
    results = []
    for (x, y, w, h) in faces:
        face_frame = frame[y:y+h, x:x+w]
        
        # 얼굴 영역 표시
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # 얼굴 영역에서 눈 검출 
        face_gray = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
        
        # 얼굴 영역에서 눈 검출
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(
            face_gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(20, 20)
        )
        
        # 눈 검출 결과 표시
        cv2.putText(frame, f"Eyes in face {x},{y}: {len(eyes)}", (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # 눈이 2개 검출된 경우에만 처리
        if len(eyes) >= 1:
            # 각 눈에 대해 처리
            for (ex, ey, ew, eh) in eyes:
                # 눈 영역 추출
                eye_roi = face_gray[ey:ey+eh, ex:ex+ew]
                
                # 눈 영역 표시
                cv2.rectangle(face_frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
                
                # 눈 크기 조정 및 전처리
                eye_roi_resized = cv2.resize(eye_roi, target_size)
                eye_roi_rgb = cv2.cvtColor(eye_roi_resized, cv2.COLOR_GRAY2RGB)
                eye_tensor = preprocess_frame(eye_roi_rgb, target_size)
                
                # 모델 예측
                with torch.no_grad():
                    eye_tensor = eye_tensor.to(device)
                    outputs = model(eye_tensor)
                    _, predicted = torch.max(outputs, 1)
                    
                # 결과 저장
                is_closed = predicted.item() == 1
                results.append((ex, ey, ew, eh, is_closed))
                
                # 눈 상태 시각화
                status = "CLOSED" if is_closed else "OPEN"
                cv2.putText(face_frame, status, (ex, ey-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255) if is_closed else (0, 255, 0), 1)
    
    return frame, results

def detect_blinks_ear(frame, face_cascade, ear_threshold=0.2, eye_cascade=None):
    """
    EAR(Eye Aspect Ratio)을 사용하여 눈 깜빡임을 감지합니다.
    """
    # 원본 프레임 복사
    debug_frame = frame.copy()
    
    # grayscale 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # eye_cascade이 None인 경우 생성
    if eye_cascade is None:
        eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
        eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
        if eye_cascade.empty():
            print("Warning: Eye cascade file could not be loaded!")
            eye_cascade = cv2.CascadeClassifier()
    
    # 얼굴 검출 - 감도를 낮추고 최소 크기를 키워 오탐지 줄이기
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,     # 값을 높여 감도 낮춤 (1.05 -> 1.1)
        minNeighbors=4,      # 값을 높여 확실한 얼굴만 검출 (2 -> 4)
        minSize=(100, 100)   # 최소 얼굴 크기를 키워 노이즈 줄임 (30, 30 -> 100, 100)
    )
    
    # 가장 큰 얼굴만 선택 (일반적으로 카메라에 가장 가까운 사람)
    if len(faces) > 0:
        # 넓이 기준으로 얼굴을 정렬
        faces = sorted(faces, key=lambda face: face[2] * face[3], reverse=True)
        # 가장 큰 얼굴만 선택
        faces = [faces[0]]
    
    # 얼굴 검출 결과 표시 (간소화)
    if len(faces) > 0:
        cv2.putText(frame, "Face detected", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    else:
        cv2.putText(frame, "No face detected", (10, 180), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    results = []
    
    # dlib 사용 시도 - 한 번만 출력하도록 변경
    has_dlib = False
    try:
        import dlib
        detector = dlib.get_frontal_face_detector()
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if os.path.exists(predictor_path):
            predictor = dlib.shape_predictor(predictor_path)
            has_dlib = True
        else:
            # 처음 실행 시에만 메시지 출력
            if not hasattr(detect_blinks_ear, 'dlib_msg_shown'):
                print(f"Landmark file does not exist: {predictor_path}")
                detect_blinks_ear.dlib_msg_shown = True
    except ImportError:
        # 처음 실행 시에만 메시지 출력
        if not hasattr(detect_blinks_ear, 'dlib_msg_shown'):
            print("dlib is not available. Using simple OpenCV-based detection.")
            detect_blinks_ear.dlib_msg_shown = True
    
    if has_dlib:
        # dlib으로 얼굴 감지 및 랜드마크 검출
        rects = detector(gray, 0)
        
        for rect in rects:
            # 얼굴 영역 표시
            cv2.rectangle(frame, (rect.left(), rect.top()), 
                     (rect.right(), rect.bottom()), (255, 0, 0), 2)
            
            # 랜드마크 검출
            shape = predictor(gray, rect)
            
            # 왼쪽 눈 랜드마크 (36-41)
            left_eye = []
            for i in range(36, 42):
                x, y = shape.part(i).x, shape.part(i).y
                left_eye.append(np.array([x, y]))
                # 랜드마크 점 표시
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
            
            # 오른쪽 눈 랜드마크 (42-47)
            right_eye = []
            for i in range(42, 48):
                x, y = shape.part(i).x, shape.part(i).y
                right_eye.append(np.array([x, y]))
                # 랜드마크 점 표시
                cv2.circle(frame, (x, y), 2, (0, 255, 255), -1)
            
            # EAR 계산
            left_ear = calculate_eye_aspect_ratio(np.array(left_eye))
            right_ear = calculate_eye_aspect_ratio(np.array(right_eye))
            
            # 양쪽 눈의 평균 EAR
            ear = (left_ear + right_ear) / 2.0
            
            # 눈 깜빡임 감지
            is_closed = ear < ear_threshold
            
            # 결과 저장
            results.append((rect.left(), rect.top(), rect.width(), rect.height(), is_closed, ear))
            
            # 시각화
            color = (0, 0, 255) if is_closed else (0, 255, 0)
            cv2.putText(frame, f"EAR: {ear:.2f}", (rect.left(), rect.top() - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    else:
        # OpenCV cascade로 얼굴 및 눈 감지
        for (x, y, w, h) in faces:
            # 얼굴 영역 표시
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # 얼굴 영역에서 눈 검출 - 감도를 높이기 위해 파라미터 조정
            face_roi = gray[y:y+h, x:x+w]
            
            # Contrast와 밝기 향상
            face_roi = cv2.equalizeHist(face_roi)
            
            # 일반적으로 눈은 얼굴 상단 반쪽에 위치함
            eyes_region_height = int(h * 0.5)  # 눈 영역을 얼굴의 50%로 제한 (0.6 -> 0.5)
            eyes_region = face_roi[0:eyes_region_height, :]
            
            # 현재 프레임의 전처리된 이미지를 사용하여 눈 감지 - 파라미터 조정
            eyes = eye_cascade.detectMultiScale(
                eyes_region,
                scaleFactor=1.05,     # 더 정확한 감지를 위해 조정 (1.03 -> 1.05)
                minNeighbors=3,       # 더 확실한 눈만 검출 (2 -> 3)
                minSize=(20, 20)      # 최소 눈 크기 증가 (15, 15 -> 20, 20)
            )
            
            # 최대 2개의 눈만 처리 (좌/우)
            if len(eyes) > 2:
                # 크기 기준으로 정렬하고 가장 큰 2개만 선택
                eyes = sorted(eyes, key=lambda eye: eye[2] * eye[3], reverse=True)[:2]
            
            # 눈 검출 결과 표시 (간소화)
            cv2.putText(frame, f"Eyes: {len(eyes)}", (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # 눈 상태 판단을 위한 변수
            is_closed = False
            avg_eye_openness = 0.3  # 기본값
            
            # 각 눈의 개별 상태를 추적
            eye_states = []
            
            # 눈이 감지되지 않으면 눈이 감겨있다고 가정할 수 있음
            if len(eyes) == 0:
                is_closed = True
                avg_eye_openness = 0.1  # 감겨있을 때의 값
                eye_states.append(True)  # 감겨있음
            else:
                total_white_ratio = 0.0
                
                # 눈이 감지되면 각 눈에 대해 분석
                for (ex, ey, ew, eh) in eyes:
                    # 실제 좌표 계산 (얼굴 내 상대 좌표 -> 절대 좌표)
                    abs_ex = x + ex
                    abs_ey = y + ey

                    # 눈 영역 표시
                    cv2.rectangle(frame, (abs_ex, abs_ey), (abs_ex+ew, abs_ey+eh), (0, 255, 0), 2)
                    
                    # 눈 영역 추출 및 분석
                    eye_roi = eyes_region[ey:ey+eh, ex:ex+ew]
                    
                    # 눈 이미지 처리 개선
                    eye_roi = cv2.GaussianBlur(eye_roi, (5, 5), 0)
                    eye_roi = cv2.equalizeHist(eye_roi)
                    
                    # 이진화를 위한 적절한 임계값 설정 (Otsu 방법 사용)
                    _, thresh = cv2.threshold(eye_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    
                    # 흰색 픽셀 비율 계산
                    white_ratio = np.sum(thresh) / (255.0 * thresh.size)
                    total_white_ratio += white_ratio
                    
                    # 눈 상태 결정 - 흰색 픽셀 비율에 따라 동적으로 판단
                    eye_closed = white_ratio < 0.08  # 더 낮은 임계값 사용
                    eye_states.append(eye_closed)
                    
                    # 눈 상태 표시
                    eye_status = "CLOSED" if eye_closed else "OPEN"
                    status_color = (0, 0, 255) if eye_closed else (0, 255, 0)
                    cv2.putText(frame, eye_status, (abs_ex, abs_ey-5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)
                
                # 감지된 눈이 있는 경우 평균 값 계산
                if len(eyes) > 0:
                    avg_eye_openness = total_white_ratio / len(eyes)
                    
                    # 모든 눈 또는 대부분의 눈이 감겨있으면 눈이 감긴 것으로 판단
                    closed_count = sum(1 for state in eye_states if state)
                    is_closed = closed_count >= len(eyes) * 0.5  # 절반 이상 감기면 감긴 것으로 판단
            
            # 결과 저장
            results.append((x, y, w, h, is_closed, avg_eye_openness))
            
            # 전체 눈 상태 표시
            eye_state_text = "EYES CLOSED" if is_closed else "EYES OPEN"
            eye_state_color = (0, 0, 255) if is_closed else (0, 255, 0)  
            cv2.putText(frame, eye_state_text, (x+w//2-40, y+h+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, eye_state_color, 2)
    
    return frame, results

def main():
    parser = argparse.ArgumentParser(description='Real-time Eye Blink Detection')
    parser.add_argument('--model_path', type=str, default=None,
                        help='Path to the trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='ear', choices=['cnn', 'rnn', 'ear'],
                        help='Model type to use (cnn, rnn, or ear)')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to run model on (cuda or cpu)')
    parser.add_argument('--ear_threshold', type=float, default=0.18,
                        help='EAR threshold for blink detection (for ear model only)')
    parser.add_argument('--video_path', type=str, default=None,
                        help='Path to video file (if not using webcam)')
    parser.add_argument('--output_path', type=str, default=None,
                        help='Path to save output video')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with additional logging')
    args = parser.parse_args()
    
    # 디버그 모드 설정
    debug_mode = args.debug
    
    # GPU 사용 가능 여부 확인
    device = torch.device(args.device)
    
    # 얼굴 감지 파라미터 개선
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    if debug_mode:
        print(f"Cascade file path: {cascade_path}")
        print(f"File exists: {os.path.exists(cascade_path)}")
    
    face_cascade = cv2.CascadeClassifier(cascade_path)
    if face_cascade.empty():
        print("Warning: Face cascade file could not be loaded!")
        face_cascade = cv2.CascadeClassifier()
    
    eye_cascade_path = cv2.data.haarcascades + 'haarcascade_eye.xml'
    if debug_mode:
        print(f"Eye cascade file path: {eye_cascade_path}")
        print(f"File exists: {os.path.exists(eye_cascade_path)}")
    
    eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
    if eye_cascade.empty():
        print("Warning: Eye cascade file could not be loaded!")
        eye_cascade = cv2.CascadeClassifier()
    
    # 모델 로드 (EAR 방식이 아닌 경우)
    model = None
    if args.model_type != 'ear' and args.model_path is not None:
        model = load_model(args.model_path, args.model_type, device)
        print(f"Loaded {args.model_type} model from {args.model_path}")
    
    # 비디오 캡처 초기화
    if args.video_path:
        cap = cv2.VideoCapture(args.video_path)
    else:
        cap = cv2.VideoCapture(0)  # 웹캠
    
    # 출력 비디오 설정
    writer = None
    if args.output_path:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        writer = cv2.VideoWriter(args.output_path, fourcc, fps, (width, height))
    
    # 깜빡임 카운터 및 시간 추적
    blink_count = 0
    last_blink_time = time.time()
    blink_times = []
    
    # EAR 값 히스토리 (시계열 처리용)
    ear_history = collections.deque(maxlen=30)
    
    # 상태 추적
    prev_state = False  # False = 열림, True = 닫힘
    blink_detected = False  # 깜빡임 상태 추적
    blink_frames = 0  # 깜빡임으로 감지된 연속 프레임 수
    open_frames = 0   # 눈이 열린 프레임 수 추적
    
    # 깜빡임 감지 매개변수 튜닝
    MIN_BLINK_FRAMES = 1  # 최소 깜빡임으로 인정할 프레임 수 (2 -> 1로 줄임)
    MIN_OPEN_FRAMES = 2   # 눈이 열린 상태로 인정할 최소 프레임 수 (3 -> 2로 줄임)
    BLINK_COOLDOWN = 0.3  # 깜빡임 사이의 최소 시간 (초) (0.15 -> 0.3으로 증가)
    
    # 사용자 조정 가능한 매개변수
    ear_threshold = args.ear_threshold  # 초기 EAR 임계값
    
    print("Starting blink detection. Press 'q' to quit.")
    print("Control keys:")
    print("  '+' or '=': Increase EAR threshold (more sensitive)")
    print("  '-': Decrease EAR threshold (less sensitive)")
    print("  'r': Reset counter")
    
    blink_rate = 0.0  # 기본값 설정
    
    # 디버그 메시지 제한을 위한 카운터
    debug_frame_counter = 0
    DEBUG_MSG_INTERVAL = 60  # 60 프레임마다 한 번씩만 디버그 메시지 출력 (30 -> 60)
    
    # 화면에 표시할 상태 정보
    status_info = {
        'eye_state': 'OPEN',
        'blink_count': 0,
        'blink_rate': 0.0,
        'ear_threshold': ear_threshold
    }
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 프레임 플립 (웹캠 거울 효과)
        if args.video_path is None:
            frame = cv2.flip(frame, 1)
        
        # 눈 깜빡임 감지
        if args.model_type == 'ear':
            # EAR 기반 방식
            processed_frame, results = detect_blinks_ear(frame, face_cascade, ear_threshold, eye_cascade)
        elif args.model_type == 'cnn':
            # CNN 기반 방식
            processed_frame, results = detect_blinks_cnn(model, frame, face_cascade, device)
        else:
            # RNN 모델은 구현 필요
            print("RNN model implementation for real-time detection is not provided")
            break
        
        # 눈 상태 확인
        is_closed = any(result[4] for result in results) if results else False
        
        # EAR 값 기록 (있는 경우)
        if args.model_type == 'ear' and results:
            ear_value = results[0][5]  # 첫 번째 감지된 얼굴의 EAR 값
            ear_history.append(ear_value)
        
        # 디버그 메시지 조절 - 일정 간격으로만 출력
        should_print_debug = debug_mode and (debug_frame_counter % DEBUG_MSG_INTERVAL == 0)
        debug_frame_counter += 1
        
        # 깜빡임 감지 로직 개선 - 상태 변화에 따른 감지
        if is_closed:
            # 눈이 감겨있는 상태
            blink_frames += 1
            
            if open_frames > 0:  # 열림에서 닫힘으로 상태 변화가 발생한 첫 프레임
                if should_print_debug:
                    print(f"Eye state changed: OPEN -> CLOSED")
            
            open_frames = 0  # 열린 프레임 카운터 리셋
            
            # 깜빡임 조건: 상태가 변화했고(열림->닫힘), 연속적으로 일정 프레임 이상 눈이 감겼을 때
            if blink_frames >= MIN_BLINK_FRAMES and not blink_detected and not prev_state:
                current_time = time.time()
                # 이전 깜빡임과 충분한 시간이 지났는지 확인 (연속 감지 방지)
                if current_time - last_blink_time > BLINK_COOLDOWN:
                    blink_count += 1
                    blink_times.append(current_time)
                    last_blink_time = current_time
                    blink_detected = True
                    
                    # 깜빡임 감지 시 항상 출력
                    print(f"Blink detected! Count: {blink_count}")
                    
                    # 깜빡임 감지 후 일정 시간 동안 추가 감지를 방지하기 위해 cooldown 설정
                    open_frames = 0
                    blink_frames = 0
        else:
            # 눈이 열려있는 상태
            open_frames += 1
            
            if blink_frames > 0:  # 닫힘에서 열림으로 상태 변화가 발생한 첫 프레임
                if should_print_debug:
                    print(f"Eye state changed: CLOSED -> OPEN")
            
            # 일정 프레임 이상 눈이 열린 상태면 깜빡임 상태 리셋
            if open_frames >= MIN_OPEN_FRAMES:
                blink_frames = 0
                blink_detected = False  # 새로운 깜빡임 감지를 위해 상태 리셋
        
        # 상태 변화 감지
        if prev_state != is_closed:
            if should_print_debug:
                print(f"Eye state transition: {prev_state} -> {is_closed}")
        
        # 이전 상태 저장
        prev_state = is_closed
        
        # 현재 상태 표시 - 더 명확하게 표시
        status_text = "Eye status: " + ("CLOSED" if is_closed else "OPEN")
        status_info['eye_state'] = "CLOSED" if is_closed else "OPEN"
        status_color = (0, 0, 255) if is_closed else (0, 255, 0)
        cv2.putText(processed_frame, status_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        
        # 깜빡임 횟수 표시
        status_info['blink_count'] = blink_count
        cv2.putText(processed_frame, f"Blink count: {blink_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 분당 깜빡임 횟수 계산 및 표시
        current_time = time.time()
        
        # 최근 60초 동안의 깜빡임만 고려
        recent_blinks = [t for t in blink_times if current_time - t <= 60]
        
        # 충분한 데이터가 있을 때만 계산
        if blink_times:
            elapsed_time = min(60, current_time - blink_times[0])
            if elapsed_time > 0:  # 0으로 나누기 방지
                blink_rate = len(recent_blinks) * 60 / elapsed_time
                status_info['blink_rate'] = blink_rate
        
        cv2.putText(processed_frame, f"Blinks per minute: {blink_rate:.1f}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        # 현재 임계값 표시
        status_info['ear_threshold'] = ear_threshold
        cv2.putText(processed_frame, f"EAR threshold: {ear_threshold:.3f}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # EAR 값 표시 (있을 경우)
        if args.model_type == 'ear' and results:
            ear_value = results[0][5]
            cv2.putText(processed_frame, f"Current EAR: {ear_value:.3f}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 화면에 사용 방법 표시 (하단에 작게 표시)
        cv2.putText(processed_frame, "Press '+'/'-': Adjust threshold | 'r': Reset | 'q': Quit", 
                   (10, processed_frame.shape[0]-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        # 화면에 표시
        cv2.imshow("Blink Detection", processed_frame)
        
        # 출력 비디오에 쓰기
        if writer:
            writer.write(processed_frame)
        
        # 키 입력 처리
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('+') or key == ord('='):
            # EAR 임계값 증가 (더 민감하게)
            ear_threshold += 0.005  # 더 세밀한 조정 (0.01 -> 0.005)
            print(f"EAR threshold increased: {ear_threshold:.3f}")
        elif key == ord('-'):
            # EAR 임계값 감소 (덜 민감하게)
            ear_threshold = max(0.02, ear_threshold - 0.005)  # 더 세밀한 조정
            print(f"EAR threshold decreased: {ear_threshold:.3f}")
        elif key == ord('r'):
            # 카운터 리셋
            blink_count = 0
            blink_times = []
            print("Blink counter reset")
        
        # 임계값 업데이트
        args.ear_threshold = ear_threshold
    
    # 자원 해제
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    print(f"Total blink detections: {blink_count}")
    print(f"Average blinks per minute: {blink_rate:.1f}")

if __name__ == "__main__":
    main() 