"""
YOLO 데이터셋 무결성 검사 통합 스크립트
- train.txt 기반 이미지/라벨 파일 존재 여부 확인
- 이미지 파일 읽기 가능 여부 검증
- 라벨 파일 형식 검증 (선택적)
"""

import os
import cv2
import glob
import argparse
from pathlib import Path
from typing import List, Tuple, Dict


def check_image_readability(image_paths: List[str], verbose: bool = False) -> Tuple[List[str], List[str]]:
    """
    이미지 파일 읽기 가능 여부 확인
    
    Args:
        image_paths: 확인할 이미지 경로 리스트
        verbose: 상세 출력 여부
        
    Returns:
        (읽기 가능한 이미지 리스트, 읽기 불가능한 이미지 리스트)
    """
    readable = []
    unreadable = []
    
    print(f"\n📷 이미지 읽기 가능 여부 확인 중... (총 {len(image_paths)}개)")
    
    for i, path in enumerate(image_paths, 1):
        if verbose and i % 100 == 0:
            print(f"  진행중: {i}/{len(image_paths)}")
            
        if not os.path.exists(path):
            unreadable.append(path)
            continue
            
        img = cv2.imread(path)
        if img is None:
            unreadable.append(path)
            if verbose:
                print(f"  ❌ 이미지 읽기 실패: {path}")
        else:
            readable.append(path)
    
    return readable, unreadable


def check_files_from_txt(
    train_txt: str, 
    check_labels: bool = True,
    check_readability: bool = False,
    verbose: bool = False
) -> Dict:
    """
    train.txt 기반 파일 존재 여부 및 무결성 확인
    
    Args:
        train_txt: train.txt 파일 경로
        check_labels: 라벨 파일 존재 여부 확인
        check_readability: 이미지 읽기 가능 여부 확인
        verbose: 상세 출력 여부
        
    Returns:
        검사 결과를 담은 딕셔너리
    """
    if not os.path.exists(train_txt):
        print(f"❌ 파일을 찾을 수 없습니다: {train_txt}")
        return None
    
    print(f"\n📋 파일 검사 시작: {train_txt}")
    print("=" * 60)
    
    # train.txt 읽기
    with open(train_txt, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    print(f"✅ train.txt에 등록된 이미지: {len(image_paths)}개")
    
    # 이미지 존재 여부 확인
    existing_images = []
    missing_images = []
    
    for img_path in image_paths:
        if os.path.exists(img_path):
            existing_images.append(img_path)
        else:
            missing_images.append(img_path)
    
    print(f"✅ 존재하는 이미지: {len(existing_images)}개")
    print(f"❌ 누락된 이미지: {len(missing_images)}개")
    
    results = {
        'total_images': len(image_paths),
        'existing_images': existing_images,
        'missing_images': missing_images,
    }
    
    # 라벨 파일 확인
    if check_labels:
        print(f"\n🏷️  라벨 파일 확인 중...")
        existing_labels = []
        missing_labels = []
        
        for img_path in existing_images:
            # 이미지 확장자 자동 감지
            label_path = None
            for ext in ['.bmp', '.jpg', '.jpeg', '.png', '.tif', '.tiff', '.webp']:
                if img_path.lower().endswith(ext):
                    label_path = img_path.replace('/images/', '/labels/').replace(ext, '.txt')
                    break
            
            if label_path and os.path.exists(label_path):
                existing_labels.append(label_path)
            elif label_path:
                missing_labels.append(label_path)
        
        print(f"✅ 존재하는 라벨: {len(existing_labels)}개")
        print(f"❌ 누락된 라벨: {len(missing_labels)}개")
        
        results['existing_labels'] = existing_labels
        results['missing_labels'] = missing_labels
    
    # 이미지 읽기 가능 여부 확인
    if check_readability and existing_images:
        readable, unreadable = check_image_readability(existing_images, verbose)
        print(f"\n✅ 읽기 가능한 이미지: {len(readable)}개")
        print(f"❌ 읽기 불가능한 이미지: {len(unreadable)}개")
        
        results['readable_images'] = readable
        results['unreadable_images'] = unreadable
    
    # 요약 출력
    print("\n" + "=" * 60)
    print("📊 검사 요약")
    print("=" * 60)
    print(f"총 이미지:           {results['total_images']}개")
    print(f"존재하는 이미지:     {len(results['existing_images'])}개")
    print(f"누락된 이미지:       {len(results['missing_images'])}개")
    
    if check_labels:
        print(f"존재하는 라벨:       {len(results['existing_labels'])}개")
        print(f"누락된 라벨:         {len(results['missing_labels'])}개")
    
    if check_readability:
        print(f"읽기 가능한 이미지:  {len(results['readable_images'])}개")
        print(f"읽기 불가능한 이미지: {len(results['unreadable_images'])}개")
    
    # 문제가 있는 파일들 출력
    if missing_images:
        print(f"\n⚠️  누락된 이미지 (처음 5개):")
        for path in missing_images[:5]:
            print(f"  - {path}")
        if len(missing_images) > 5:
            print(f"  ... 외 {len(missing_images) - 5}개")
    
    if check_labels and missing_labels:
        print(f"\n⚠️  누락된 라벨 (처음 5개):")
        for path in missing_labels[:5]:
            print(f"  - {path}")
        if len(missing_labels) > 5:
            print(f"  ... 외 {len(missing_labels) - 5}개")
    
    if check_readability and unreadable:
        print(f"\n⚠️  읽기 불가능한 이미지 (처음 5개):")
        for path in unreadable[:5]:
            print(f"  - {path}")
        if len(unreadable) > 5:
            print(f"  ... 외 {len(unreadable) - 5}개")
    
    return results


def check_directory_images(
    image_dir: str, 
    pattern: str = "*",
    check_readability: bool = True,
    verbose: bool = False
) -> Dict:
    """
    특정 디렉토리의 모든 이미지 파일 검사
    
    Args:
        image_dir: 이미지 디렉토리 경로
        pattern: glob 패턴 (예: "*.jpg", "*")
        check_readability: 이미지 읽기 가능 여부 확인
        verbose: 상세 출력 여부
        
    Returns:
        검사 결과를 담은 딕셔너리
    """
    if not os.path.exists(image_dir):
        print(f"❌ 디렉토리를 찾을 수 없습니다: {image_dir}")
        return None
    
    print(f"\n📁 디렉토리 검사: {image_dir}")
    print("=" * 60)
    
    # 이미지 파일 수집
    img_paths = glob.glob(os.path.join(image_dir, pattern))
    print(f"✅ 발견된 파일: {len(img_paths)}개")
    
    results = {
        'image_dir': image_dir,
        'total_files': len(img_paths),
        'image_paths': img_paths,
    }
    
    if check_readability and img_paths:
        readable, unreadable = check_image_readability(img_paths, verbose)
        print(f"\n✅ 읽기 가능한 이미지: {len(readable)}개")
        print(f"❌ 읽기 불가능한 이미지: {len(unreadable)}개")
        
        results['readable_images'] = readable
        results['unreadable_images'] = unreadable
        
        if unreadable:
            print(f"\n⚠️  읽기 불가능한 이미지:")
            for path in unreadable:
                print(f"  - {path}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 데이터셋 무결성 검사 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # train.txt 기반 검사 (기본)
  python check_data_integrity.py --train-txt /path/to/train.txt
  
  # train.txt 기반 검사 + 라벨 확인
  python check_data_integrity.py --train-txt /path/to/train.txt --check-labels
  
  # train.txt 기반 검사 + 라벨 + 읽기 가능 여부 확인
  python check_data_integrity.py --train-txt /path/to/train.txt --check-labels --check-readability
  
  # 디렉토리 내 모든 이미지 검사
  python check_data_integrity.py --image-dir /path/to/images
  
  # 디렉토리 내 특정 패턴 이미지 검사
  python check_data_integrity.py --image-dir /path/to/images --pattern "*.jpg"
        """
    )
    
    # 모드 선택
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        '--train-txt',
        type=str,
        help='train.txt 파일 경로'
    )
    mode_group.add_argument(
        '--image-dir',
        type=str,
        help='이미지 디렉토리 경로'
    )
    
    # 옵션
    parser.add_argument(
        '--check-labels',
        action='store_true',
        help='라벨 파일 존재 여부 확인 (train.txt 모드에서만)'
    )
    parser.add_argument(
        '--check-readability',
        action='store_true',
        help='이미지 읽기 가능 여부 확인'
    )
    parser.add_argument(
        '--pattern',
        type=str,
        default='*',
        help='이미지 파일 패턴 (image-dir 모드에서만, 기본: *)'
    )
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='상세 출력'
    )
    
    args = parser.parse_args()
    
    # 모드별 실행
    if args.train_txt:
        results = check_files_from_txt(
            train_txt=args.train_txt,
            check_labels=args.check_labels,
            check_readability=args.check_readability,
            verbose=args.verbose
        )
    else:  # args.image_dir
        results = check_directory_images(
            image_dir=args.image_dir,
            pattern=args.pattern,
            check_readability=args.check_readability,
            verbose=args.verbose
        )
    
    # 결과 저장 옵션 (추후 확장 가능)
    if results:
        print("\n✅ 검사 완료!")


if __name__ == "__main__":
    main()
