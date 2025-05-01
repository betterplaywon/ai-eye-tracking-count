import cv2
import numpy as np

def detect_face(image, face_cascade):
    """
    이미지에서 얼굴을 감지합니다.
    
    Args:
        image: 입력 이미지 (cv2 형식)
        face_cascade: OpenCV 얼굴 검출기
    
    Returns:
        감지된 얼굴 영역 목록
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    return faces

def get_eyes_from_face(image, face, eye_cascade):
    """
    감지된 얼굴에서 눈 영역을 검출합니다.
    
    Args:
        image: 입력 이미지
        face: 얼굴 영역 (x, y, w, h)
        eye_cascade: OpenCV 눈 검출기
    
    Returns:
        감지된 눈 영역 목록 (얼굴 영역 내 상대 좌표)
    """
    x, y, w, h = face
    face_img = image[y:y+h, x:x+w]
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    eyes = eye_cascade.detectMultiScale(gray_face, 1.1, 4)
    return eyes, gray_face

def calculate_eye_aspect_ratio(eye_landmarks):
    """
    눈 랜드마크를 사용하여 EAR(Eye Aspect Ratio)를 계산합니다.
    EAR은 눈이 얼마나 열려있는지를 나타내는 비율입니다.
    
    Args:
        eye_landmarks: 눈의 랜드마크 좌표 배열
        
    Returns:
        EAR 값
    """
    # 세로 랜드마크 거리 계산
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    
    # 가로 랜드마크 거리 계산
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    
    # EAR 계산
    ear = (A + B) / (2.0 * C)
    return ear

def extract_eye_features(eye_roi):
    """
    눈 이미지에서 특징을 추출합니다.
    
    Args:
        eye_roi: 눈 영역 이미지
        
    Returns:
        추출된 특징 벡터
    """
    # 이미지 전처리
    eye_roi = cv2.resize(eye_roi, (24, 24))
    eye_roi = cv2.equalizeHist(eye_roi)
    
    # HOG 특징 추출
    winSize = (24, 24)
    blockSize = (8, 8)
    blockStride = (4, 4)
    cellSize = (4, 4)
    nbins = 9
    derivAperture = 1
    winSigma = -1.
    histogramNormType = 0
    L2HysThreshold = 0.2
    gammaCorrection = True
    nlevels = 64
    
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture,
                          winSigma, histogramNormType, L2HysThreshold, gammaCorrection, nlevels)
    
    features = hog.compute(eye_roi)
    return features.flatten() 