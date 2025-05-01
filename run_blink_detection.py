#!/usr/bin/env python3
import argparse
import os
import sys
from src.detect_blinks import main as detect_main

def main():
    parser = argparse.ArgumentParser(description='눈 깜빡임 감지 시스템')
    subparsers = parser.add_subparsers(dest='command', help='실행할 명령')
    
    # 실시간 감지 명령어
    detect_parser = subparsers.add_parser('detect', help='실시간 눈 깜빡임 감지')
    detect_parser.add_argument('--model_path', type=str, default=None,
                      help='학습된 모델 파일 경로')
    detect_parser.add_argument('--model_type', type=str, default='ear', 
                      choices=['cnn', 'rnn', 'ear'],
                      help='사용할 모델 유형 (cnn, rnn, 또는 ear)')
    detect_parser.add_argument('--device', type=str, default='cpu',
                      help='실행할 장치 (cuda 또는 cpu)')
    detect_parser.add_argument('--ear_threshold', type=float, default=0.2,
                      help='깜빡임 감지를 위한 EAR 임계값 (ear 모델 전용)')
    detect_parser.add_argument('--video_path', type=str, default=None,
                      help='비디오 파일 경로 (웹캠을 사용하지 않는 경우)')
    detect_parser.add_argument('--output_path', type=str, default=None,
                      help='출력 비디오 저장 경로')
    
    # 모델 학습 명령어 (미구현)
    train_parser = subparsers.add_parser('train', help='눈 깜빡임 감지 모델 학습')
    train_parser.add_argument('--model_type', type=str, default='cnn',
                     choices=['cnn', 'rnn', 'ear'],
                     help='학습할 모델 유형 (cnn, rnn, 또는 ear)')
    train_parser.add_argument('--data_path', type=str, required=True,
                     help='데이터셋 경로')
    train_parser.add_argument('--batch_size', type=int, default=32,
                     help='배치 크기 (기본값: 32)')
    train_parser.add_argument('--epochs', type=int, default=50,
                     help='에폭 수 (기본값: 50)')
    train_parser.add_argument('--save_dir', type=str, default='./output',
                     help='모델과 결과를 저장할 디렉토리')
    
    args = parser.parse_args()
    
    if args.command == 'detect':
        # 이 간단한 방식을 사용하여 detect_blinks.py 스크립트를 직접 실행합니다
        if len(sys.argv) > 1:
            # 'detect' 인수를 제거하고 나머지 인수를 그대로 전달
            os.system(f"python -m src.detect_blinks {' '.join(sys.argv[2:])}")
        else:
            # 인수 없이 실행
            os.system("python -m src.detect_blinks")
    elif args.command == 'train':
        # 모델 학습 실행 (아직 구현되지 않음)
        print("모델 학습은 아직 구현되지 않았습니다. 향후 업데이트를 기대해주세요.")
        print("대신 다음 명령어로 직접 학습 스크립트를 실행할 수 있습니다:")
        print(f"python -m src.models.train --model_type {args.model_type} --data_path {args.data_path} --batch_size {args.batch_size} --epochs {args.epochs} --save_dir {args.save_dir}")
    else:
        # 명령어가 지정되지 않은 경우
        parser.print_help()

if __name__ == "__main__":
    main() 