from ultralytics import YOLO
import os
import torch.hub
import gc
import torch
import urllib.request
import shutil

# 초기 설정
Date = "250808"
Time = "11h"  # 시간 업데이트
User = "hwoh"
Product = "wln"
yolo_nm = "11l"  # YOLO v11 Large
model_nm = f"{Product}_dod_{yolo_nm}_{Date}{Time}"

os.chdir(f"/home/{User}/detection/{Product}")
torch.hub.set_dir(f"/home/{User}/detection/{Product}")

# 🔥 공통 YOLO 모델 폴더 시스템 + 자동 다운로드
YOLO_MODELS_DIR = "/home/hwoh/yolo_models"
target_model = f"yolo{yolo_nm}.pt"

print(f"🔍 YOLO 모델 시스템 초기화 ({Product.upper()})")
print(f"   요청 모델: {target_model}")
print(f"   공통 폴더: {YOLO_MODELS_DIR}")

# 공통 폴더 생성
os.makedirs(YOLO_MODELS_DIR, exist_ok=True)

# 🎯 실제 존재하는 YOLO 모델 다운로드 URL 매핑 (2024년 8월 기준)
model_urls = {
    # ✅ YOLO v8 모델들 (확실히 존재하는 URL)
    "yolo8n.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n.pt",
    "yolo8s.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8s.pt",
    "yolo8m.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8m.pt",
    "yolo8l.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8l.pt",
    "yolo8x.pt": "https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8x.pt",
    
    # ⚠️ YOLO v11 모델들은 직접 URL이 없음 - Ultralytics 라이브러리만 사용
    # "yolo11n.pt": "URL_NOT_AVAILABLE",
    # "yolo11s.pt": "URL_NOT_AVAILABLE", 
    # "yolo11m.pt": "URL_NOT_AVAILABLE",
    # "yolo11l.pt": "URL_NOT_AVAILABLE",
    # "yolo11x.pt": "URL_NOT_AVAILABLE",
}

# 🔥 YOLO v11 전용 라이브러리 다운로드 목록
yolo11_models = ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]

def download_model_direct(model_name, save_path):
    """직접 HTTP 다운로드 함수 (YOLO v8만 지원)"""
    if model_name not in model_urls:
        print(f"⚠️  {model_name}은 직접 다운로드 URL이 없습니다.")
        if model_name in yolo11_models:
            print(f"   YOLO v11 모델은 Ultralytics 라이브러리를 통해서만 다운로드 가능합니다.")
        return False
    
    url = model_urls[model_name]
    print(f"🌐 직접 다운로드 시도: {url}")
    
    try:
        def progress_hook(block_num, block_size, total_size):
            if total_size > 0:
                percent = min(100, (block_num * block_size) / total_size * 100)
                print(f"\r   다운로드 진행률: {percent:.1f}%", end="", flush=True)
        
        urllib.request.urlretrieve(url, save_path, reporthook=progress_hook)
        print("")  # 줄바꿈
        
        if os.path.exists(save_path):
            file_size = os.path.getsize(save_path) / (1024 * 1024)
            print(f"✅ 직접 다운로드 성공! ({file_size:.1f}MB)")
            return True
        else:
            print(f"❌ 다운로드 실패: 파일이 생성되지 않음")
            return False
            
    except Exception as e:
        print(f"❌ 직접 다운로드 실패: {e}")
        return False

def download_model_ultralytics(model_name, save_path):
    """Ultralytics 라이브러리를 통한 다운로드 (YOLO v8 + v11 지원)"""
    try:
        print(f"🤖 Ultralytics Hub에서 다운로드: {model_name}")
        
        # yolo8l -> yolov8l 변환 (Ultralytics 라이브러리 호환)
        hub_model_name = model_name
        if model_name.startswith("yolo8") and not model_name.startswith("yolov8"):
            hub_model_name = model_name.replace("yolo8", "yolov8")
            print(f"   Hub 호환 이름으로 변환: {model_name} -> {hub_model_name}")
        elif model_name.startswith("yolo11"):
            # YOLO v11은 그대로 사용 (최신 ultralytics에서 지원)
            print(f"   YOLO v11 모델: {hub_model_name}")
        
        # 임시 디렉토리 사용하여 안전하게 다운로드
        import tempfile
        original_cwd = os.getcwd()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                os.chdir(temp_dir)
                
                # 모델 다운로드
                temp_model = YOLO(hub_model_name)
                print(f"   모델 객체 생성 성공")
                
                # 다운로드된 파일 찾기
                downloaded_files = [f for f in os.listdir(temp_dir) if f.endswith('.pt')]
                
                if downloaded_files:
                    temp_model_path = os.path.join(temp_dir, downloaded_files[0])
                    shutil.copy2(temp_model_path, save_path)
                    
                    if os.path.exists(save_path):
                        file_size = os.path.getsize(save_path) / (1024 * 1024)
                        print(f"✅ Ultralytics 다운로드 성공! ({file_size:.1f}MB)")
                        return True
                else:
                    # 파일이 없으면 직접 저장 시도
                    print(f"   다운로드 파일을 찾을 수 없어 직접 저장 시도...")
                    temp_model.save(save_path)
                    
                    if os.path.exists(save_path):
                        file_size = os.path.getsize(save_path) / (1024 * 1024)
                        print(f"✅ Ultralytics save() 성공! ({file_size:.1f}MB)")
                        return True
                    else:
                        print(f"❌ save() 실패: 파일 생성 안됨")
                        return False
                        
            finally:
                os.chdir(original_cwd)
                
    except Exception as e:
        print(f"❌ Ultralytics 다운로드 실패: {e}")
        if "yolo11" in model_name.lower():
            print(f"   💡 YOLO v11은 최신 Ultralytics 버전이 필요할 수 있습니다:")
            print(f"      pip install ultralytics --upgrade")
        os.chdir(original_cwd)
        return False

# 실제 파일명 확정
if yolo_nm in model_mapping:
    actual_filename = model_mapping[yolo_nm]
    actual_model_path = f"{YOLO_MODELS_DIR}/{actual_filename}"
    print(f"🔄 모델명 매핑: {target_model} -> {actual_filename}")
else:
    actual_filename = target_model
    actual_model_path = f"{YOLO_MODELS_DIR}/{target_model}"

print(f"   실제 파일: {actual_model_path}")

# 다단계 모델 로딩 시스템
model = None

# yolo_nm이 v8 계열인지 v11 계열인지 확인
if yolo_nm.startswith("8"):
    model_series = "YOLO v8"
elif yolo_nm.startswith("11"):
    model_series = "YOLO v11"
else:
    model_series = "Unknown"
    
print(f"   모델 시리즈: {model_series}")

# 1단계: 공통 폴더에서 로딩 시도
if os.path.exists(actual_model_path):
    try:
        file_size = os.path.getsize(actual_model_path) / (1024 * 1024)
        print(f"📂 공통 폴더에서 모델 발견 ({file_size:.1f}MB)")
        
        model = YOLO(actual_model_path)
        print(f"✅ 공통 모델 로딩 성공!")
        
    except Exception as load_error:
        print(f"❌ 공통 모델 로딩 실패: {load_error}")
        print(f"⚠️  파일이 손상되었을 수 있습니다. 재다운로드 시도...")
        
        # 손상된 파일 삭제
        try:
            os.remove(actual_model_path)
            print(f"🗑️  손상된 파일 삭제: {actual_model_path}")
        except:
            pass
        model = None

# 2단계: 모델이 없거나 손상된 경우 다운로드
if model is None:
    print(f"📥 {actual_filename} 다운로드 시작...")
    
    # YOLO v11인지 v8인지 확인
    is_yolo11 = actual_filename in yolo11_models
    
    if is_yolo11:
        print(f"🔄 YOLO v11 모델 - Ultralytics 라이브러리 우선 사용")
        # YOLO v11은 Ultralytics 라이브러리만 사용
        success = download_model_ultralytics(actual_filename, actual_model_path)
        
        if not success:
            print(f"❌ YOLO v11 다운로드 실패")
            print(f"💡 해결 방법:")
            print(f"1. Ultralytics 업그레이드: pip install ultralytics --upgrade")
            print(f"2. 또는 YOLO v8 모델 사용 (yolo_nm을 8l, 8m, 8s 등으로 변경)")
            exit(1)
    else:
        print(f"🔄 YOLO v8 모델 - 직접 다운로드 우선 시도")
        # YOLO v8은 직접 다운로드 → Ultralytics 순서
        success = download_model_direct(actual_filename, actual_model_path)
        
        if not success:
            print(f"🔄 대안 방법으로 재시도...")
            success = download_model_ultralytics(actual_filename, actual_model_path)
        
        if not success:
            print(f"❌ 모든 다운로드 방법 실패")
            print(f"💡 수동 해결 방법:")
            print(f"1. 인터넷 연결 확인")
            print(f"2. 수동 다운로드 (YOLO v8):")
            print(f"   cd {YOLO_MODELS_DIR}")
            if actual_filename in model_urls:
                print(f"   wget {model_urls[actual_filename]}")
            print(f"3. check_pkg.sh 실행:")
            print(f"   bash /home/hwoh/check_pkg.sh")
            print(f"4. 지원 모델:")
            print(f"   YOLO v8 (안정): 8n, 8s, 8m, 8l, 8x")
            print(f"   YOLO v11 (최신): 11n, 11s, 11m, 11l, 11x")
            exit(1)
    
    # 다운로드 성공 후 로딩
    try:
        model = YOLO(actual_model_path)
        print(f"✅ 다운로드된 모델 로딩 성공!")
    except Exception as load_error:
        print(f"❌ 다운로드된 모델 로딩 실패: {load_error}")
        exit(1)

# 최종 모델 확인
if model is None:
    print(f"❌ 모델 로딩에 완전히 실패했습니다.")
    exit(1)

# 모델 정보 출력
print(f"")
print(f"🤖 모델 정보 ({Product.upper()}):")
try:
    total_params = sum(p.numel() for p in model.model.parameters())
    model_size = "Large" if "l" in yolo_nm else ("Medium" if "m" in yolo_nm else ("Small" if "s" in yolo_nm else ("Nano" if "n" in yolo_nm else "Extra Large")))
    
    print(f"   모델: {model_series} {model_size} - {actual_filename}")
    print(f"   파라미터: {total_params/1e6:.1f}M")
    print(f"   로딩 경로: {actual_model_path}")
    
    # 공통 폴더 내 다른 모델들 확인
    yolo_files = [f for f in os.listdir(YOLO_MODELS_DIR) if f.startswith('yolo') and f.endswith('.pt')]
    if len(yolo_files) > 1:
        print(f"   공통 폴더 내 모델: {', '.join(sorted(yolo_files))}")
except:
    print(f"   모델: {actual_filename} (정보 로딩 실패)")

# GPU 메모리 정리 (YOLO v11 Large는 메모리 집약적)
if torch.cuda.is_available():
    torch.cuda.set_device(0)
    for i in range(3):  # 3번 반복 정리
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
    
    # GPU 상태 확인
    device = torch.cuda.current_device()
    gpu_name = torch.cuda.get_device_name(0)
    total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
    allocated = torch.cuda.memory_allocated(0) / 1024**3
    cached = torch.cuda.memory_reserved(0) / 1024**3
    
    print(f"")
    print(f"🖥️  GPU 정보 (YOLO v11 Large 전용):")
    print(f"   GPU: {gpu_name}")
    print(f"   Device: {device}")
    print(f"   Total Memory: {total_memory:.2f}GB")
    print(f"   Allocated: {allocated:.2f}GB")
    print(f"   Cached: {cached:.2f}GB")
    print(f"   Available: {total_memory - cached:.2f}GB")
    
    # YOLO v11 Large 메모리 요구사항 체크
    available_memory = total_memory - cached
    if available_memory < 6.0:  # YOLO v11 Large는 최소 6GB 권장
        print(f"   ⚠️  YOLO v11 Large는 최소 6GB 메모리 권장 (현재: {available_memory:.2f}GB)")
        print(f"      배치 크기를 줄이거나 더 작은 모델(11m, 11s) 사용 권장")
else:
    print("⚠️  CUDA not available, using CPU (YOLO v11 Large는 GPU 강력 권장)")

print(f"")
print(f"🚀 YOLO 학습 시작 ({Product.upper()})...")
print(f"   모델: {actual_filename}")
print(f"   시리즈: {model_series}")
print(f"   데이터: {Product}_defect_detection_data_v2.yaml")

# 모델 학습 (YOLO v11 Large 최적화 파라미터)
model.train(
    data=f"/hdd/datasets/dod_data/{Product}/{Product}_defect_detection_data_v2.yaml",
    epochs=500,
    batch=16,  # 🔥 Large 모델이므로 배치 크기 감소 (32->16)
    patience=150,
    dropout=0.3,
    iou=0.48, 
    lr0=0.0003,  # 🔥 Large 모델 학습률 조정
    lrf=0.00001,
    optimizer="AdamW",
    exist_ok=True,
    project=f"/home/{User}/detection/{Product}/runs/detect",
    name=f"{model_nm}",
    verbose=True,
)

# 텐서보드 안내 메시지 출력
print(f"")
print(f"📊 [INFO] 텐서보드 실행: tensorboard --logdir /home/{User}/detection/{Product}/runs/detect")

# 베스트 모델 로드
best_weight_path = f"/home/{User}/detection/{Product}/runs/detect/{model_nm}/weights/best.pt"

print(f"")
print(f"🎉 학습 완료 ({Product.upper()})!")
print(f"   Best 가중치: {best_weight_path}")
print(f"   사용 모델: {actual_model_path}")

if not os.path.exists(best_weight_path):
    best_weight_path = f"/home/{User}/detection/{Product}/runs/detect/{model_nm}/weights/last.pt"
    if not os.path.exists(best_weight_path):
        raise FileNotFoundError(f"가중치 파일이 존재하지 않습니다: {best_weight_path}")
    print(f"   ⚠️  best.pt가 없어서 last.pt 사용: {best_weight_path}")

model_Product = YOLO(best_weight_path)

print(f"")
print(f"🎯 모델 예측 시작 ({Product.upper()})...")

# 모델 예측 수행 (신뢰도 포함)
results = model_Product.predict(    
    f"/hdd/datasets/dod_data/{Product}/val2/images",
    save=True,
    save_crop=True,
    save_txt=True,
    exist_ok=True,
    project=f"/home/{User}/detection/{Product}/runs/detect/{model_nm}",
    name=f"pred_{model_nm}_val",
)
print(f"✅ 신뢰도 포함 예측 완료")

# 신뢰도 없이 라벨만
results = model_Product.predict(
    f"/hdd/datasets/dod_data/{Product}/val2/images",
    save=True,
    save_crop=True,
    save_txt=True,
    show_conf=False,
    exist_ok=True,
    project=f"/home/{User}/detection/{Product}/runs/detect/{model_nm}",
    name=f"pred_{model_nm}_val_without_conf",
)
print(f"✅ 신뢰도 제외 예측 완료")

print(f"")
print(f"🎯 모든 작업 완료! ({Product.upper()})")
print(f"   공통 모델 폴더: {YOLO_MODELS_DIR}")
print(f"   사용된 모델: {actual_filename}")
print(f"   학습 결과: {best_weight_path}")
print(f"   예측 결과: /home/{User}/detection/{Product}/runs/detect/{model_nm}/")