"""
YOLO 데이터셋 경로 텍스트 파일 생성기
train.txt, valid.txt, test.txt 파일을 자동으로 생성합니다.
"""

import os
import argparse
from glob import glob
from pathlib import Path
from typing import List, Set


# 지원하는 이미지 확장자
DEFAULT_IMAGE_EXTENSIONS = [
    "bmp", "jpg", "jpeg", "png", "tif", "tiff", "webp",
    "BMP", "JPG", "JPEG", "PNG", "TIF", "TIFF", "WEBP"
]


def collect_images(base_dir: str, extensions: List[str] = None, recursive: bool = False) -> List[str]:
    """
    디렉토리에서 이미지 파일 경로를 수집합니다.
    
    Args:
        base_dir: 이미지가 있는 디렉토리 경로
        extensions: 이미지 확장자 리스트 (None이면 기본값 사용)
        recursive: 하위 폴더까지 재귀적으로 검색할지 여부
        
    Returns:
        이미지 파일의 절대 경로 리스트
    """
    if not os.path.exists(base_dir):
        print(f"⚠️  디렉토리가 존재하지 않습니다: {base_dir}")
        return []
    
    if extensions is None:
        extensions = DEFAULT_IMAGE_EXTENSIONS
    
    image_list = []
    pattern = "**/*" if recursive else "*"
    
    for ext in extensions:
        search_pattern = os.path.join(base_dir, f"{pattern}.{ext}")
        found = glob(search_pattern, recursive=recursive)
        image_list.extend(found)
    
    # 절대 경로로 변환 및 중복 제거
    image_list = [os.path.abspath(path) for path in image_list]
    
    return image_list


def remove_duplicates(image_list: List[str]) -> List[str]:
    """중복된 이미지 경로 제거 (순서 유지)"""
    seen = set()
    result = []
    for path in image_list:
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def save_image_list(image_list: List[str], output_path: str, create_dirs: bool = True) -> bool:
    """
    이미지 경로 리스트를 텍스트 파일로 저장합니다.
    
    Args:
        image_list: 이미지 경로 리스트
        output_path: 출력 파일 경로
        create_dirs: 출력 디렉토리가 없으면 생성할지 여부
        
    Returns:
        저장 성공 여부
    """
    try:
        # 출력 디렉토리 생성
        if create_dirs:
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
        
        # 파일 저장
        with open(output_path, "w") as f:
            if image_list:
                f.write("\n".join(image_list) + "\n")
            else:
                f.write("")  # 빈 파일 생성
        
        return True
    except Exception as e:
        print(f"❌ 파일 저장 실패: {e}")
        return False


def create_dataset_txt(
    product: str,
    data_version: str,
    base_path: str = "/hdd/datasets",
    data_type: str = "dod_data",
    splits: List[str] = ["train", "val", "test"],
    extra_dirs: List[str] = None,
    output_dir: str = None,
    recursive: bool = False,
    verbose: bool = False
):
    """
    YOLO 데이터셋의 train.txt, valid.txt, test.txt 파일을 생성합니다.
    
    Args:
        product: 제품명 (예: jjb, csn, wln, obj)
        data_version: 데이터 버전 (예: v8, v10, 251015_psm)
        base_path: 데이터셋 기본 경로
        data_type: 데이터 타입 (dod_data, cls_data, obj_data 등)
        splits: 생성할 split 리스트 (train, val, test)
        extra_dirs: 추가 이미지 디렉토리 리스트
        output_dir: 출력 파일을 저장할 디렉토리 (None이면 자동 설정)
        recursive: 하위 폴더까지 검색할지 여부
        verbose: 상세 출력 여부
    """
    print(f"\n{'='*70}")
    print(f"📋 YOLO 데이터셋 텍스트 파일 생성")
    print(f"{'='*70}")
    print(f"제품: {product}")
    print(f"버전: {data_version}")
    print(f"데이터 타입: {data_type}")
    print(f"Split: {', '.join(splits)}")
    print(f"{'='*70}\n")
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = os.path.join(base_path, data_type, product, data_version)
    
    results = {}
    
    for split in splits:
        print(f"📂 {split.upper()} 데이터 수집 중...")
        
        # 기본 이미지 디렉토리
        main_dir = os.path.join(base_path, data_type, product, data_version, split, "images")
        image_list = collect_images(main_dir, recursive=recursive)
        
        if verbose:
            print(f"  - 기본 경로: {main_dir} ({len(image_list)}개)")
        
        # 추가 디렉토리 처리 (train만 해당)
        if split == "train" and extra_dirs:
            print(f"  ➕ 추가 디렉토리 수집 중...")
            for extra_dir in extra_dirs:
                # 절대 경로가 아니면 기본 경로 기준으로 변환
                if not os.path.isabs(extra_dir):
                    extra_dir = os.path.join(base_path, data_type, product, extra_dir)
                
                extra_images = collect_images(extra_dir, recursive=recursive)
                image_list.extend(extra_images)
                
                if verbose:
                    print(f"  - 추가 경로: {extra_dir} ({len(extra_images)}개)")
        
        # 중복 제거
        original_count = len(image_list)
        image_list = remove_duplicates(image_list)
        duplicates = original_count - len(image_list)
        
        if duplicates > 0:
            print(f"  🔄 중복 제거: {duplicates}개")
        
        # 파일 저장
        output_name = "valid.txt" if split == "val" else f"{split}.txt"
        output_path = os.path.join(output_dir, output_name)
        
        if save_image_list(image_list, output_path):
            print(f"  ✅ 저장 완료: {output_path}")
            print(f"  📊 이미지 수: {len(image_list)}개\n")
            results[split] = {
                "count": len(image_list),
                "path": output_path,
                "success": True
            }
        else:
            print(f"  ❌ 저장 실패\n")
            results[split] = {
                "count": len(image_list),
                "path": output_path,
                "success": False
            }
    
    # 결과 요약
    print(f"{'='*70}")
    print(f"📊 생성 결과 요약")
    print(f"{'='*70}")
    
    total_images = 0
    for split, result in results.items():
        status = "✅" if result["success"] else "❌"
        split_name = "valid" if split == "val" else split
        print(f"{status} {split_name}.txt: {result['count']:,}개 이미지")
        if result["success"]:
            total_images += result["count"]
    
    print(f"\n📁 출력 위치: {output_dir}")
    print(f"🎯 전체 이미지: {total_images:,}개")
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO 데이터셋 경로 텍스트 파일 생성기",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 기본 사용 (train, val, test 모두 생성)
  python make_txt_yolo.py --product jjb --version v10
  
  # train만 생성
  python make_txt_yolo.py --product csn --version v8 --splits train
  
  # 추가 디렉토리 포함
  python make_txt_yolo.py --product jjb --version v10 --extra-dirs 250529_add_data/images 250916_add_data/images
  
  # 출력 경로 지정
  python make_txt_yolo.py --product obj --version 251015_psm --data-type obj_data --output-dir /hdd/datasets/obj_data/251015_psm
  
  # 하위 폴더까지 재귀 검색
  python make_txt_yolo.py --product wln --version v5 --recursive
        """
    )
    
    # 필수 인자
    parser.add_argument(
        "--product", "-p",
        type=str,
        required=True,
        help="제품명 (예: jjb, csn, wln, obj)"
    )
    parser.add_argument(
        "--version", "-v",
        type=str,
        required=True,
        help="데이터 버전 (예: v8, v10, 251015_psm)"
    )
    
    # 선택 인자
    parser.add_argument(
        "--base-path",
        type=str,
        default="/hdd/datasets",
        help="데이터셋 기본 경로 (기본값: /hdd/datasets)"
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="dod_data",
        help="데이터 타입 (기본값: dod_data, 예: cls_data, obj_data)"
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        choices=["train", "val", "test"],
        help="생성할 split (기본값: train val test)"
    )
    parser.add_argument(
        "--extra-dirs",
        nargs="+",
        help="추가 이미지 디렉토리 (train에만 적용, 상대경로 또는 절대경로)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="출력 파일을 저장할 디렉토리 (기본값: 자동 설정)"
    )
    parser.add_argument(
        "--recursive", "-r",
        action="store_true",
        help="하위 폴더까지 재귀적으로 검색"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 출력"
    )
    
    args = parser.parse_args()
    
    # 실행
    create_dataset_txt(
        product=args.product,
        data_version=args.version,
        base_path=args.base_path,
        data_type=args.data_type,
        splits=args.splits,
        extra_dirs=args.extra_dirs,
        output_dir=args.output_dir,
        recursive=args.recursive,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()