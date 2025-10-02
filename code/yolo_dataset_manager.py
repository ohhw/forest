#!/usr/bin/env python
"""
YOLO 데이터셋 관리 통합 스크립트
기능:
1. 데이터셋 무결성 검사
   - 이미지-라벨 매칭 검사
   - 라벨 파일 형식 검증
   - 중복 파일 검사
2. 데이터셋 전처리
   - 매칭되지 않는 파일 정리
   - 라벨 파일 클래스 ID 형식 수정
3. train.txt 생성 및 검증
"""

import os
from glob import glob
from pathlib import Path
import shutil
from datetime import datetime
from collections import defaultdict
from typing import List, Dict, Tuple, Set, Optional
import argparse
from yolo_augmenter import YOLODataAugmenter

class YOLODatasetManager:
    def __init__(self, img_dir: str, label_dir: str, output_dir: str, extra_img_dirs: List[Tuple[str, str]] = None):
        """
        YOLODatasetManager 초기화
        
        Args:
            img_dir: 기본 이미지 디렉토리 경로
            label_dir: 라벨 디렉토리 경로
            output_dir: 출력 파일(train.txt 등)이 저장될 디렉토리 경로
            extra_img_dirs: 추가 이미지 디렉토리 목록. (경로, 설명) 튜플의 리스트
        """
        # 경로 설정
        self.img_dir = os.path.abspath(img_dir)
        self.label_dir = os.path.abspath(label_dir)
        self.output_dir = os.path.abspath(output_dir)
        
        # 백업 설정
        self.backup_root = os.path.join(os.path.dirname(self.img_dir), "backups")
        self.backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.backup_dir = os.path.join(self.backup_root, self.backup_time)
        
        # 데이터 소스 설정
        self.data_sources = [(self.img_dir, "기본 데이터")]
        if extra_img_dirs:
            self.data_sources.extend(
                [(os.path.abspath(path), desc) for path, desc in extra_img_dirs]
            )
        
        # 지원하는 이미지 확장자
        self.img_exts = ["bmp", "jpg", "jpeg", "png", "tif", "tiff",
                        "BMP", "JPG", "JPEG", "PNG", "TIF", "TIFF"]
    
    def create_backup_dirs(self):
        """백업 디렉토리 생성"""
        os.makedirs(os.path.join(self.backup_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.backup_dir, "labels"), exist_ok=True)
        print(f"\n백업 디렉토리 생성됨: {self.backup_dir}")
    
    def backup_file(self, src: str, is_image: bool = True) -> bool:
        """파일 백업"""
        if os.path.exists(src):
            subdir = "images" if is_image else "labels"
            dst = os.path.join(self.backup_dir, subdir, os.path.basename(src))
            shutil.copy2(src, dst)
            return True
        return False
    
    def get_base_name(self, filepath: str) -> str:
        """파일의 기본 이름을 추출 (확장자 제외)"""
        return Path(filepath).stem
    
    def check_dataset_integrity(self) -> Tuple[List[str], List[str], List[str]]:
        """데이터셋 무결성 검사"""
        print("\n=== 1. 데이터셋 무결성 검사 ===")
        
        # 이미지와 라벨 파일 리스트 가져오기
        image_files = []
        for ext in self.img_exts:
            image_files.extend(glob(os.path.join(self.img_dir, f"*.{ext}")))
        
        label_files = glob(os.path.join(self.label_dir, "*.txt"))
        
        # 기본 이름으로 셋 생성
        image_names = set(self.get_base_name(f) for f in image_files)
        label_names = set(self.get_base_name(f) for f in label_files)
        
        # 매칭 검사
        unmatched_images = list(image_names - label_names)
        unmatched_labels = list(label_names - image_names)
        
        # 중복 파일 검사
        duplicate_files = []
        seen_files = set()
        for img_file in image_files:
            base_name = self.get_base_name(img_file)
            if base_name in seen_files:
                duplicate_files.append(img_file)
            seen_files.add(base_name)
        
        # 결과 출력
        print(f"\n1) 매칭되지 않는 파일:")
        print(f"   - 라벨 없는 이미지: {len(unmatched_images)}개")
        print(f"   - 이미지 없는 라벨: {len(unmatched_labels)}개")
        print(f"2) 중복 파일: {len(duplicate_files)}개")
        
        return unmatched_images, unmatched_labels, duplicate_files
    
    def check_label_format(self) -> List[Tuple[str, str]]:
        """라벨 파일 형식 검사"""
        print("\n=== 2. 라벨 파일 형식 검사 ===")
        
        invalid_labels = []
        label_files = glob(os.path.join(self.label_dir, "*.txt"))
        
        for file_path in label_files:
            try:
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                
                for line_num, line in enumerate(lines, 1):
                    parts = line.strip().split()
                    if not parts:
                        invalid_labels.append((file_path, f"빈 줄 (라인 {line_num})"))
                        continue
                    
                    try:
                        # 클래스 ID 검사
                        class_id = float(parts[0])
                        if class_id != int(class_id):
                            invalid_labels.append((file_path, f"소수점 클래스 ID: {class_id} (라인 {line_num})"))
                        
                        # 좌표값 검사
                        coords = [float(x) for x in parts[1:]]
                        if len(coords) != 4:
                            invalid_labels.append((file_path, f"잘못된 좌표 수: {len(coords)} (라인 {line_num})"))
                        elif not all(0 <= x <= 1 for x in coords):
                            invalid_labels.append((file_path, f"좌표값 범위 오류 (라인 {line_num})"))
                    
                    except ValueError:
                        invalid_labels.append((file_path, f"숫자 변환 오류 (라인 {line_num})"))
            
            except Exception as e:
                invalid_labels.append((file_path, f"파일 읽기 오류: {str(e)}"))
        
        print(f"\n발견된 형식 오류: {len(invalid_labels)}개")
        return invalid_labels
    
    def clean_unmatched_files(self):
        """매칭되지 않는 파일 정리"""
        print("\n=== 3. 매칭되지 않는 파일 정리 ===")
        
        # 무결성 검사 실행
        unmatched_images, unmatched_labels, _ = self.check_dataset_integrity()
        
        # 1. 라벨이 없는 이미지 처리
        removed_images = 0
        print(f"\n라벨이 없는 이미지 처리 중... (총 {len(unmatched_images)}개)")
        for name in unmatched_images:
            for ext in self.img_exts:
                img_path = os.path.join(self.img_dir, f"{name}.{ext}")
                if os.path.exists(img_path):
                    if self.backup_file(img_path, is_image=True):
                        os.remove(img_path)
                        removed_images += 1
                        print(f"  - 제거됨: {os.path.basename(img_path)}")
        
        # 2. 이미지가 없는 라벨 처리
        removed_labels = 0
        print(f"\n이미지가 없는 라벨 처리 중... (총 {len(unmatched_labels)}개)")
        for name in unmatched_labels:
            label_path = os.path.join(self.label_dir, f"{name}.txt")
            if os.path.exists(label_path):
                if self.backup_file(label_path, is_image=False):
                    os.remove(label_path)
                    removed_labels += 1
                    if removed_labels % 100 == 0:
                        print(f"  - {removed_labels}/{len(unmatched_labels)} 처리됨...")
        
        print("\n정리 완료:")
        print(f"- 제거된 이미지: {removed_images}개")
        print(f"- 제거된 라벨: {removed_labels}개")
    
    def fix_label_format(self):
        """라벨 파일 형식 수정"""
        print("\n=== 4. 라벨 파일 형식 수정 ===")
        
        # 형식 검사 실행
        invalid_labels = self.check_label_format()
        
        modified = 0
        for file_path, error in invalid_labels:
            if "소수점 클래스 ID" in error:  # 소수점 클래스 ID만 수정
                try:
                    with open(file_path, 'r') as f:
                        lines = f.readlines()
                    
                    fixed_lines = []
                    changes_made = False
                    for line in lines:
                        parts = line.strip().split()
                        if parts:
                            # 클래스 ID를 float에서 int로 변환
                            class_id = int(float(parts[0]))
                            fixed_line = f"{class_id} {' '.join(parts[1:])}\n"
                            if fixed_line != line:
                                changes_made = True
                            fixed_lines.append(fixed_line)
                    
                    if changes_made:
                        self.backup_file(file_path, is_image=False)
                        with open(file_path, 'w') as f:
                            f.writelines(fixed_lines)
                        modified += 1
                
                except Exception as e:
                    print(f"Error fixing {file_path}: {e}")
        
        print(f"\n형식 수정 완료:")
        print(f"- 수정된 파일: {modified}개")
    
    def create_train_txt(self):
        """train.txt 파일 생성 및 검증"""
        print("\n=== 5. train.txt 생성 및 검증 ===")
        
        # 이미지 경로 리스트 생성
        train_img_list = []
        for img_dir, source_name in self.data_sources:
            source_images = []
            for ext in self.img_exts:
                source_images.extend(glob(os.path.join(img_dir, f"*.{ext}")))
            print(f"{source_name}: {len(source_images)}개 이미지 발견")
            train_img_list.extend(source_images)
        
        # 중복 제거
        train_img_list = list(set(train_img_list))
        
        # 모든 이미지 파일 존재 여부 확인
        missing_files = []
        valid_files = []
        for img_path in train_img_list:
            if os.path.exists(img_path):
                valid_files.append(img_path)
            else:
                missing_files.append(img_path)
        
        # 결과 출력
        print(f"\n검증 결과:")
        print(f"- 총 이미지 경로: {len(train_img_list)}개")
        print(f"- 유효한 경로: {len(valid_files)}개")
        print(f"- 존재하지 않는 경로: {len(missing_files)}개")
        
        # train.txt 파일 생성
        if valid_files:
            os.makedirs(self.output_dir, exist_ok=True)
            output_path = os.path.join(self.output_dir, "train.txt")
            
            with open(output_path, "w") as f:
                f.write("\n".join(valid_files) + "\n")
            
            print(f"\ntrain.txt 생성 완료:")
            print(f"- 저장 위치: {output_path}")
            print(f"- 포함된 이미지 수: {len(valid_files)}개")
    
    def process_all(self, check_only: bool = False):
        """전체 처리 과정 실행"""
        print("=== YOLO 데이터셋 관리 시작 ===")
        
        # 백업 디렉토리 생성
        self.create_backup_dirs()
        
        # 1. 데이터셋 무결성 검사
        unmatched_images, unmatched_labels, duplicate_files = self.check_dataset_integrity()
        
        # 2. 라벨 파일 형식 검사
        invalid_labels = self.check_label_format()
        
        if check_only:
            print("\n검사 결과 요약:")
            print(f"1. 매칭되지 않는 파일:")
            print(f"   - 라벨 없는 이미지: {len(unmatched_images)}개")
            print(f"   - 이미지 없는 라벨: {len(unmatched_labels)}개")
            print(f"2. 중복 파일: {len(duplicate_files)}개")
            print(f"3. 형식이 잘못된 라벨: {len(invalid_labels)}개")
            return
        
        # 3. 매칭되지 않는 파일 정리
        self.clean_unmatched_files()
        
        # 4. 라벨 파일 형식 수정
        self.fix_label_format()
        
        # 5. train.txt 생성 및 검증
        self.create_train_txt()
        
        print("\n=== 모든 처리 완료 ===")
        print(f"백업 위치: {self.backup_dir}")

class YOLODatasetManagerCLI:
    def __init__(self):
        self.manager = None
    
    def clear_screen(self):
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def print_header(self):
        self.clear_screen()
        print("=" * 50)
        print("       YOLO 데이터셋 관리 도구 v1.0")
        print("=" * 50)
        print()
    
    def get_directory_path(self, prompt: str, required: bool = True) -> str:
        while True:
            path = input(prompt).strip()
            if not path and not required:
                return ""
            if not path:
                print("경로를 입력해주세요.")
                continue
            
            path = os.path.expanduser(path)  # ~ 확장
            path = os.path.abspath(path)     # 절대 경로 변환
            
            if not os.path.exists(path):
                print(f"경로가 존재하지 않습니다: {path}")
                if input("계속 사용하시겠습니까? (y/N) ").lower() != 'y':
                    continue
            return path
    
    def setup_directories(self):
        self.print_header()
        print("[1/3] 기본 디렉토리 설정")
        print("-" * 30)
        
        img_dir = self.get_directory_path("이미지 디렉토리 경로: ")
        label_dir = self.get_directory_path("라벨 디렉토리 경로: ")
        output_dir = self.get_directory_path("출력 디렉토리 경로: ")
        
        extra_dirs = []
        print("\n[2/3] 추가 이미지 디렉토리 설정")
        print("-" * 30)
        print("(추가 디렉토리 입력을 완료하려면 경로를 비워두고 Enter를 누르세요)")
        
        while True:
            path = self.get_directory_path("\n추가 이미지 디렉토리 경로 (선택사항): ", required=False)
            if not path:
                break
            desc = input("디렉토리 설명: ").strip() or f"추가 데이터 {len(extra_dirs) + 1}"
            extra_dirs.append((path, desc))
        
        self.manager = YOLODatasetManager(
            img_dir=img_dir,
            label_dir=label_dir,
            output_dir=output_dir,
            extra_img_dirs=extra_dirs
        )
    
    def show_main_menu(self):
        while True:
            self.print_header()
            print("[3/3] 작업 선택")
            print("-" * 30)
            print("1. 전체 데이터셋 처리")
            print("2. 데이터셋 검사만 수행")
            print("3. 매칭되지 않는 파일 정리")
            print("4. 라벨 파일 형식 수정")
            print("5. train.txt 생성")
            print("6. 디렉토리 설정 변경")
            print("0. 종료")
            print("-" * 30)
            
            choice = input("\n선택하세요 (0-6): ").strip()
            
            if choice == "0":
                return
            elif choice == "1":
                self.manager.process_all(check_only=False)
            elif choice == "2":
                self.manager.process_all(check_only=True)
            elif choice == "3":
                self.manager.clean_unmatched_files()
            elif choice == "4":
                self.manager.fix_label_format()
            elif choice == "5":
                self.manager.create_train_txt()
            elif choice == "6":
                self.setup_directories()
                continue
            
            input("\n계속하려면 Enter를 누르세요...")
    
    def setup_augmentation(self) -> Tuple[Dict[str, str], List[str], int]:
        """데이터 증강 설정"""
        self.print_header()
        print("[데이터 증강 설정]")
        print("-" * 30)
        
        # 디렉토리 설정
        paths = {}
        print("\n1. 디렉토리 설정")
        paths["input_images"] = self.get_directory_path("입력 이미지 디렉토리: ")
        paths["input_labels"] = self.get_directory_path("입력 라벨 디렉토리: ")
        paths["output_images"] = self.get_directory_path("출력 이미지 디렉토리: ")
        paths["output_labels"] = self.get_directory_path("출력 라벨 디렉토리: ")
        
        # 증강 유형 선택
        print("\n2. 증강 유형 선택")
        print("사용 가능한 증강:")
        aug_types = ["기본", "밝기", "회전", "노이즈", "흐림", "색상"]
        for i, aug in enumerate(aug_types, 1):
            print(f"{i}. {aug}")
        
        selected_types = []
        while True:
            choice = input("\n증강 유형 번호를 선택하세요 (여러 개는 쉼표로 구분, 완료는 엔터): ").strip()
            if not choice:
                break
            try:
                for num in map(int, choice.split(",")):
                    if 1 <= num <= len(aug_types):
                        selected_type = aug_types[num-1]
                        if selected_type not in selected_types:
                            selected_types.append(selected_type)
            except ValueError:
                continue
        
        if not selected_types:
            selected_types = ["기본"]
            print("\n기본 증강이 선택되었습니다.")
        else:
            print("\n선택된 증강:", ", ".join(selected_types))
        
        # 복사본 수 설정
        copies = 1
        while True:
            try:
                copies = int(input("\n이미지당 생성할 증강 복사본 수 (기본값: 1): ").strip() or "1")
                if copies > 0:
                    break
                print("1 이상의 숫자를 입력하세요.")
            except ValueError:
                print("올바른 숫자를 입력하세요.")
        
        return paths, selected_types, copies
    
    def run_augmentation(self, paths: Dict[str, str], aug_types: List[str], copies: int):
        """데이터 증강 실행"""
        augmenter = YOLODataAugmenter()
        
        def show_progress(current: int, total: int):
            progress = (current / total) * 100
            print(f"\r진행률: {progress:.1f}% ({current}/{total})", end="")
        
        success, failed = augmenter.augment_dataset(
            image_dir=paths["input_images"],
            label_dir=paths["input_labels"],
            output_image_dir=paths["output_images"],
            output_label_dir=paths["output_labels"],
            augmentation_types=aug_types,
            copies_per_image=copies,
            progress_callback=show_progress
        )
        
        print(f"\n\n증강 완료:")
        print(f"- 성공: {success}개")
        print(f"- 실패: {failed}개")
    
    def show_main_menu(self):
        while True:
            self.print_header()
            print("[3/3] 작업 선택")
            print("-" * 30)
            print("1. 전체 데이터셋 처리")
            print("2. 데이터셋 검사만 수행")
            print("3. 매칭되지 않는 파일 정리")
            print("4. 라벨 파일 형식 수정")
            print("5. train.txt 생성")
            print("6. 데이터 증강")
            print("7. 디렉토리 설정 변경")
            print("0. 종료")
            print("-" * 30)
            
            choice = input("\n선택하세요 (0-7): ").strip()
            
            if choice == "0":
                return
            elif choice == "1":
                self.manager.process_all(check_only=False)
            elif choice == "2":
                self.manager.process_all(check_only=True)
            elif choice == "3":
                self.manager.clean_unmatched_files()
            elif choice == "4":
                self.manager.fix_label_format()
            elif choice == "5":
                self.manager.create_train_txt()
            elif choice == "6":
                paths, aug_types, copies = self.setup_augmentation()
                self.run_augmentation(paths, aug_types, copies)
            elif choice == "7":
                self.setup_directories()
                continue
            
            input("\n계속하려면 Enter를 누르세요...")
    
    def run(self):
        try:
            self.setup_directories()
            self.show_main_menu()
        except KeyboardInterrupt:
            print("\n\n프로그램을 종료합니다.")
        except Exception as e:
            print(f"\n오류가 발생했습니다: {str(e)}")
            if input("\n상세 오류를 보시겠습니까? (y/N) ").lower() == 'y':
                import traceback
                traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="YOLO 데이터셋 관리 도구")
    parser.add_argument("--cli", action="store_true",
                      help="대화형 모드로 실행")
    parser.add_argument("--img-dir",
                      help="기본 이미지 디렉토리 경로")
    parser.add_argument("--label-dir",
                      help="라벨 디렉토리 경로")
    parser.add_argument("--output-dir",
                      help="출력 파일이 저장될 디렉토리 경로")
    parser.add_argument("--extra-img-dirs", nargs="+", action="append",
                      metavar=("PATH", "DESCRIPTION"),
                      help="추가 이미지 디렉토리 경로와 설명. 예: path1 '설명1' path2 '설명2'")
    parser.add_argument("--check-only", action="store_true", 
                      help="검사만 수행하고 수정하지 않습니다")
    
    args = parser.parse_args()
    
    # CLI 모드로 실행
    if args.cli or not (args.img_dir and args.label_dir and args.output_dir):
        cli = YOLODatasetManagerCLI()
        cli.run()
        return
    
    # 커맨드 라인 인자로 실행
    extra_dirs = []
    if args.extra_img_dirs:
        for group in args.extra_img_dirs:
            if len(group) % 2 != 0:
                parser.error("각 추가 이미지 디렉토리에는 경로와 설명이 모두 필요합니다")
            for i in range(0, len(group), 2):
                extra_dirs.append((group[i], group[i+1]))
    
    manager = YOLODatasetManager(
        img_dir=args.img_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        extra_img_dirs=extra_dirs
    )
    manager.process_all(check_only=args.check_only)

if __name__ == "__main__":
    main()
