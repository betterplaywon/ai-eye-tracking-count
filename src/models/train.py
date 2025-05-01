import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 로컬 모듈 임포트
from src.models.blink_detection_model import BlinkDetectionCNN, BlinkDetectionRNN, EyeAspectRatioDetector

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=50, device='cuda', save_path=None, patience=10):
    """
    모델을 학습시키는 함수
    
    Args:
        model: 학습할 PyTorch 모델
        train_loader: 학습 데이터 로더
        val_loader: 검증 데이터 로더
        criterion: 손실 함수
        optimizer: 최적화 알고리즘
        num_epochs: 총 에폭 수
        device: 학습 장치 (CPU/GPU)
        save_path: 모델 저장 경로
        patience: 조기 종료 인내심
        
    Returns:
        history: 손실과 정확도 기록
    """
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    
    for epoch in range(num_epochs):
        # 학습 모드
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        # 진행 상황 표시를 위한 tqdm 사용
        train_loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        
        for inputs, targets in train_loop:
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 그래디언트 초기화
            optimizer.zero_grad()
            
            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 역전파
            loss.backward()
            optimizer.step()
            
            # 손실과 정확도 계산
            train_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()
            
            # 진행 상황 업데이트
            train_loop.set_postfix(loss=loss.item(), acc=train_correct/train_total)
            
        train_loss = train_loss / len(train_loader.dataset)
        train_acc = train_correct / train_total
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        val_loop = tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Valid]')
        
        with torch.no_grad():
            for inputs, targets in val_loop:
                inputs, targets = inputs.to(device), targets.to(device)
                
                # 순전파
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # 손실과 정확도 계산
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()
                
                # 진행 상황 업데이트
                val_loop.set_postfix(loss=loss.item(), acc=val_correct/val_total)
        
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = val_correct / val_total
        
        # 결과 기록
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch+1}/{num_epochs}: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # 모델 저장 (검증 손실이 개선된 경우)
        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            early_stop_counter = 0
            
            # 저장 경로가 없으면 생성
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # 모델 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc
            }, save_path)
            print(f"Model saved to {save_path}")
        else:
            early_stop_counter += 1
            
        # 조기 종료
        if early_stop_counter >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs")
            break
            
    return history

def evaluate_model(model, test_loader, criterion, device='cuda'):
    """
    모델을 평가하는 함수
    
    Args:
        model: 평가할 PyTorch 모델
        test_loader: 테스트 데이터 로더
        criterion: 손실 함수
        device: 평가 장치 (CPU/GPU)
        
    Returns:
        metrics: 평가 지표 (정확도, 정밀도, 재현율, F1 점수)
    """
    model.eval()
    test_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc='Evaluating'):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # 순전파
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # 손실 계산
            test_loss += loss.item() * inputs.size(0)
            
            # 예측값 저장
            _, predicted = torch.max(outputs, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    # 총 손실 계산
    test_loss = test_loss / len(test_loader.dataset)
    
    # 성능 지표 계산
    accuracy = accuracy_score(all_targets, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_predictions, average='binary')
    
    metrics = {
        'loss': test_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    print("테스트 결과:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    return metrics

def plot_history(history, save_path=None):
    """
    학습 과정을 시각화하는 함수
    
    Args:
        history: 학습 과정에서 기록한 지표
        save_path: 그래프 저장 경로 (선택사항)
    """
    plt.figure(figsize=(12, 4))
    
    # 손실 그래프
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 정확도 그래프
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Training Accuracy')
    plt.plot(history['val_acc'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f"학습 그래프가 {save_path}에 저장되었습니다.")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Eye Blink Detection Model Training')
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'rnn', 'ear'],
                        help='Model type to train (cnn, rnn, or ear)')
    parser.add_argument('--data_path', type=str, required=True, 
                        help='Path to the dataset directory or files')
    parser.add_argument('--batch_size', type=int, default=32, 
                        help='Batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=50, 
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=0.001, 
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--save_dir', type=str, default='./output', 
                        help='Directory to save model and results')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to train on (default: cuda if available)')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Sequence length for RNN models (default: 10)')
    args = parser.parse_args()
    
    # 저장 디렉토리 생성
    os.makedirs(args.save_dir, exist_ok=True)
    
    # GPU 사용 가능 여부 확인
    device = torch.device(args.device)
    
    print(f"Training on: {device}")
    print(f"Model type: {args.model_type}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    
    # TODO: 여기에 데이터 로드 코드 추가
    # 이 부분은 실제 데이터셋 형식에 따라 구현해야 함
    
    # 예시 코드:
    '''
    train_dataset = YourDataset(args.data_path, split='train')
    val_dataset = YourDataset(args.data_path, split='val')
    test_dataset = YourDataset(args.data_path, split='test')
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    '''
    
    # 모델 생성
    if args.model_type == 'cnn':
        model = BlinkDetectionCNN(input_channels=3, input_size=(128, 128))
    elif args.model_type == 'rnn':
        model = BlinkDetectionRNN(input_channels=3, input_size=(128, 128), 
                                sequence_length=args.sequence_length)
    elif args.model_type == 'ear':
        model = EyeAspectRatioDetector(sequence_length=args.sequence_length)
    
    model.to(device)
    
    # 손실 함수 및 최적화 알고리즘 설정
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # 모델 저장 경로
    model_save_path = os.path.join(args.save_dir, f"blink_detection_{args.model_type}.pth")
    
    # 여기서 train_loader와 val_loader가 정의되어 있다고 가정
    # history = train_model(model, train_loader, val_loader, criterion, optimizer, 
    #                    num_epochs=args.epochs, device=device, save_path=model_save_path)
    
    # 학습 곡선 시각화 및 저장
    # plot_history(history, save_path=os.path.join(args.save_dir, f"training_history_{args.model_type}.png"))
    
    # 테스트 평가
    # metrics = evaluate_model(model, test_loader, criterion, device=device)
    
    print("학습 완료!")

if __name__ == "__main__":
    main() 