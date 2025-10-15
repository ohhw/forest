"""
범용 이미지 포맷 변환기 (인터랙티브 경로 탐색 기능)
다양한 이미지 포맷 간 변환을 지원합니다.
지원 포맷: TIFF, BMP, PNG, JPG, JPEG, WEBP, GIF 등
리눅스 터미널처럼 cd, ls, .. 명령어로 경로 탐색 가능
필요한 패키지 자동 설치 기능 포함
"""

import os
import sys
import subprocess

def install_package(package_name):
    """패키지를 자동으로 설치합니다."""
    try:
        print(f"📦 {package_name} 패키지를 설치하는 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package_name])
        print(f"✅ {package_name} 설치 완료!")
        return True
    except subprocess.CalledProcessError:
        print(f"❌ {package_name} 설치 실패!")
        return False

def check_and_install_dependencies():
    """필요한 패키지들이 설치되어 있는지 확인하고 없으면 설치합니다."""
    required_packages = {
        'PIL': 'Pillow'  # import 이름: pip 패키지 이름
    }
    
    missing_packages = []
    
    for import_name, package_name in required_packages.items():
        try:
            if import_name == 'PIL':
                __import__('PIL')
        except ImportError:
            missing_packages.append((import_name, package_name))
    
    if missing_packages:
        print("⚠️  다음 패키지들이 필요합니다:")
        for import_name, package_name in missing_packages:
            print(f"   - {package_name} ({import_name})")
        
        install_confirm = input("\n🤔 필요한 패키지들을 자동으로 설치하시겠습니까? (y/N): ").lower().strip()
        
        if install_confirm in ['y', 'yes']:
            success_count = 0
            for import_name, package_name in missing_packages:
                if install_package(package_name):
                    success_count += 1
            
            if success_count == len(missing_packages):
                print("✅ 모든 패키지가 성공적으로 설치되었습니다!")
                print("🔄 프로그램을 다시 실행해주세요.")
            else:
                print("❌ 일부 패키지 설치에 실패했습니다.")
            sys.exit(0)
        else:
            print("❌ 필요한 패키지 없이는 프로그램을 실행할 수 없습니다.")
            print("💡 수동으로 설치하려면: pip install Pillow")
            sys.exit(1)

# 의존성 확인 및 설치
check_and_install_dependencies()

# 이제 안전하게 PIL import (패키지가 설치되어 있다고 가정)
try:
    from PIL import Image, features
except ImportError:
    print("❌ PIL 패키지 import 실패. 프로그램을 다시 실행해주세요.")
    sys.exit(1)
import shutil

# 지원하는 이미지 포맷을 Pillow에서 동적으로 가져오기
def get_supported_formats():
    """Pillow에서 실제 지원하는 모든 포맷을 동적으로 가져옵니다."""
    
    # 읽기 지원 포맷 (입력)
    input_formats = list(Image.registered_extensions().keys())
    
    # 쓰기 지원 포맷 (출력) 
    output_formats = {}
    for ext, format_name in Image.registered_extensions().items():
        # 확장자에서 점(.) 제거하고 소문자로 변환
        clean_ext = ext.lstrip('.').lower()
        
        # 해당 포맷으로 저장이 가능한지 확인
        if format_name in Image.SAVE:
            output_formats[clean_ext] = format_name
    
    return input_formats, output_formats

# Pillow에서 지원하는 모든 포맷을 동적으로 로드
input_formats, output_formats = get_supported_formats()

SUPPORTED_FORMATS = {
    'input': input_formats,
    'output': output_formats
}

print(f"🎉 Pillow에서 지원하는 포맷을 모두 로드했습니다!")
print(f"📖 입력 지원: {len(input_formats)}개 포맷")
print(f"💾 출력 지원: {len(output_formats)}개 포맷")

# Pillow에서 실제 지원하는 포맷 확인
def check_pillow_support():
    """Pillow에서 실제 지원하는 포맷들을 확인합니다."""
    print("\n🔍 Pillow 라이브러리 지원 포맷 확인 중...")
    
    # 지원되는 입력/출력 포맷 확인
    supported_read = set(Image.registered_extensions().keys())
    supported_write = set()
    
    for ext, format_name in Image.registered_extensions().items():
        try:
            # 해당 포맷으로 저장이 가능한지 확인
            if format_name in Image.SAVE:
                supported_write.add(ext)
        except:
            pass
    
    print(f"📖 읽기 지원: {len(supported_read)}개 포맷")
    print(f"💾 쓰기 지원: {len(supported_write)}개 포맷")
    
    # 우리가 정의한 포맷 중 실제 지원되지 않는 것들 찾기
    unsupported_input = []
    unsupported_output = []
    
    for ext in SUPPORTED_FORMATS['input']:
        if ext.lower() not in [x.lower() for x in supported_read]:
            unsupported_input.append(ext)
    
    for format_key in SUPPORTED_FORMATS['output'].keys():
        ext = f".{format_key}"
        if ext.lower() not in [x.lower() for x in supported_write]:
            unsupported_output.append(format_key)
    
    return unsupported_input, unsupported_output, supported_read, supported_write

# 지원 포맷 확인 (디버그 모드에서만)
DEBUG_MODE = False  # True로 변경하면 지원 포맷 확인

if DEBUG_MODE:
    unsupported_in, unsupported_out, supported_read, supported_write = check_pillow_support()
    if unsupported_in or unsupported_out:
        print("\n⚠️  일부 포맷이 현재 환경에서 지원되지 않을 수 있습니다:")
        if unsupported_in:
            print(f"   입력 미지원: {', '.join(unsupported_in)}")
        if unsupported_out:
            print(f"   출력 미지원: {', '.join(unsupported_out)}")
        print("   → 해당 포맷 변환 시 오류가 발생할 수 있습니다.")
    else:
        print("✅ 모든 정의된 포맷이 현재 환경에서 지원됩니다!")
    
    print(f"\n📊 지원 현황:")
    print(f"   입력 포맷: {len(SUPPORTED_FORMATS['input'])}개 정의 / {len(supported_read)}개 실제 지원")
    print(f"   출력 포맷: {len(SUPPORTED_FORMATS['output'])}개 정의 / {len(supported_write)}개 실제 지원")
    print("=" * 60)

class PathNavigator:
    """터미널처럼 경로를 탐색할 수 있는 클래스"""
    
    def __init__(self):
        # 운영체제에 관계없이 현재 작업 디렉토리를 정규화해서 저장
        self.current_path = os.path.normpath(os.getcwd())
    
    def pwd(self):
        """현재 경로 출력 (정규화된 경로)"""
        return os.path.normpath(self.current_path)
    
    def ls(self, show_details=False):
        """현재 디렉토리 내용 출력"""
        try:
            items = os.listdir(self.current_path)
            if not items:
                print("📁 빈 폴더입니다.")
                return
            
            folders = []
            files = []
            image_files = []
            
            for item in sorted(items):
                item_path = os.path.join(self.current_path, item)
                if os.path.isdir(item_path):
                    folders.append(f"📁 {item}/")
                else:
                    file_ext = os.path.splitext(item)[1].lower()
                    if file_ext in SUPPORTED_FORMATS['input']:
                        image_files.append(f"🖼️  {item}")
                    else:
                        files.append(f"📄 {item}")
            
            # 출력
            if folders:
                print("\n📂 폴더:")
                for folder in folders:
                    print(f"  {folder}")
            
            if image_files:
                print(f"\n🖼️  이미지 파일 ({len(image_files)}개):")
                for img in image_files[:10]:  # 처음 10개만 표시
                    print(f"  {img}")
                if len(image_files) > 10:
                    print(f"  ... 그 외 {len(image_files) - 10}개 더")
            
            if files:
                print(f"\n� 기타 파일 ({len(files)}개):")
                for file in files[:5]:  # 처음 5개만 표시
                    print(f"  {file}")
                if len(files) > 5:
                    print(f"  ... 그 외 {len(files) - 5}개 더")
                    
        except PermissionError:
            print("❌ 권한이 없습니다.")
        except Exception as e:
            print(f"❌ 오류: {e}")
    
    def cd(self, path):
        """디렉토리 변경"""
        if not path or path == "":
            return False
        
        if path == "..":
            # 부모 디렉토리로 이동
            parent = os.path.dirname(self.current_path)
            if parent != self.current_path:  # 루트가 아닌 경우
                self.current_path = os.path.normpath(parent)
                return True
            else:
                print("❌ 이미 루트 디렉토리입니다.")
                return False
        
        elif path == "~":
            # 홈 디렉토리로 이동
            self.current_path = os.path.normpath(os.path.expanduser("~"))
            return True
        
        elif os.path.isabs(path):
            # 절대 경로 (Windows의 C:\ 포함)
            if os.path.isdir(path):
                self.current_path = os.path.normpath(path)
                return True
            else:
                print(f"❌ 존재하지 않는 경로입니다: {path}")
                return False
        
        else:
            # 상대 경로
            new_path = os.path.join(self.current_path, path)
            if os.path.isdir(new_path):
                self.current_path = os.path.normpath(os.path.abspath(new_path))
                return True
            else:
                print(f"❌ 존재하지 않는 경로입니다: {path}")
                return False
    
    def find_images(self, recursive=True):
        """현재 경로에서 이미지 파일 검색 (재귀적 옵션)"""
        image_files = []
        
        if recursive:
            # 재귀적으로 하위 폴더까지 검색
            try:
                for root, dirs, files in os.walk(self.current_path):
                    for file in files:
                        file_ext = os.path.splitext(file)[1].lower()
                        if file_ext in SUPPORTED_FORMATS['input']:
                            image_files.append(os.path.join(root, file))
            except Exception as e:
                print(f"❌ 이미지 검색 오류: {e}")
        else:
            # 현재 폴더만 검색
            try:
                for item in os.listdir(self.current_path):
                    item_path = os.path.join(self.current_path, item)
                    if os.path.isfile(item_path):
                        file_ext = os.path.splitext(item)[1].lower()
                        if file_ext in SUPPORTED_FORMATS['input']:
                            image_files.append(item_path)
            except Exception as e:
                print(f"❌ 이미지 검색 오류: {e}")
        
        return image_files

def navigate_to_input_path():
    """사용자가 입력 경로를 탐색하도록 안내"""
    navigator = PathNavigator()
    
    print("\n🗂️  입력 경로 탐색 모드")
    print("=" * 50)
    print("💡 사용 가능한 명령어:")
    print("   ls        - 현재 폴더 내용 보기")
    print("   cd <폴더> - 폴더로 이동 (예: cd images)")
    print("   cd ..     - 상위 폴더로 이동")
    print("   cd ~      - 홈 폴더로 이동")
    print("   pwd       - 현재 경로 보기")
    print("   ok        - 현재 경로를 입력 경로로 선택 (하위폴더 포함)")
    print("   exit      - 프로그램 종료")
    print("=" * 50)
    print("📌 참고: 'ok' 선택 시 현재 경로/converted 폴더에 변환 결과 저장")
    
    # 현재 위치 표시
    print(f"\n📍 현재 위치: {navigator.pwd()}")
    navigator.ls()
    
    while True:
        # 현재 경로를 명령어 입력 바로 위에 표시
        current_path = navigator.pwd()
        
        # 운영체제별 경로 정규화
        display_path = os.path.normpath(current_path)
        
        # 폴더명 추출 (Windows/Unix 호환)
        folder_name = os.path.basename(display_path)
        if not folder_name:  # 루트 디렉토리인 경우
            if os.name == 'nt':  # Windows
                folder_name = display_path  # C:\ 같은 경우
            else:  # Unix 계열
                folder_name = "root"
        
        print(f"\n📂 현재 경로: {display_path}")
        print(f"🖥️  {folder_name}$ ", end="")
        command = input().strip()
        
        if not command:
            continue
        
        parts = command.split()
        cmd = parts[0].lower()
        
        if cmd == "exit":
            print("👋 프로그램을 종료합니다.")
            sys.exit(0)
        
        elif cmd == "ls":
            navigator.ls()
        
        elif cmd == "pwd":
            print(f"📍 현재 경로: {navigator.pwd()}")
        
        elif cmd == "cd":
            if len(parts) > 1:
                if navigator.cd(parts[1]):
                    # cd 성공 후 바로 ls 실행해서 내용 보여주기
                    navigator.ls()
            else:
                print("❌ 사용법: cd <폴더명>")
        
        elif cmd == "ok":
            # 현재 경로와 하위 폴더에서 이미지 검색
            print("🔍 이미지 파일을 검색하는 중...")
            
            # 현재 폴더만 먼저 검색
            current_images = navigator.find_images(recursive=False)
            # 하위 폴더까지 포함한 전체 검색
            all_images = navigator.find_images(recursive=True)
            
            if all_images:
                print(f"✅ 입력 경로로 선택됨: {navigator.pwd()}")
                print(f"🖼️  현재 폴더 이미지: {len(current_images)}개")
                print(f"🖼️  전체 이미지 (하위폴더 포함): {len(all_images)}개")
                
                if len(current_images) == 0 and len(all_images) > 0:
                    print("💡 현재 폴더에는 이미지가 없지만 하위 폴더에서 이미지를 찾았습니다.")
                    
                return navigator.pwd()
            else:
                print("⚠️  현재 경로와 하위 폴더에서 지원되는 이미지 파일을 찾을 수 없습니다.")
                print("   다른 폴더를 선택해주세요.")
                print(f"💡 지원 포맷: {', '.join(SUPPORTED_FORMATS['input'])}")
        
        else:
            print(f"❌ 알 수 없는 명령어: {command}")
            print("💡 사용 가능한 명령어: ls, cd, pwd, ok, exit")

def get_output_format():
    """출력 포맷을 선택받습니다."""
    print("\n🎯 출력 포맷 선택")
    print("=" * 70)
    
    # 포맷을 카테고리별로 분류해서 표시
    common_formats = ['jpg', 'jpeg', 'png', 'bmp', 'gif', 'tiff', 'tif', 'webp']
    
    # 일반적인 포맷과 특수 포맷 분리
    available_common = [f for f in common_formats if f in SUPPORTED_FORMATS['output']]
    all_formats = list(SUPPORTED_FORMATS['output'].keys())
    special_formats = [f for f in all_formats if f not in common_formats]
    
    print("📋 지원하는 출력 포맷:")
    print(f"\n  🌟 일반 포맷 ({len(available_common)}개):")
    if available_common:
        # 한 줄에 8개씩 표시
        for i in range(0, len(available_common), 8):
            line_formats = available_common[i:i+8]
            print(f"    {', '.join(line_formats)}")
    
    if special_formats:
        print(f"\n  🔧 특수 포맷 ({len(special_formats)}개):")
        # 한 줄에 8개씩 표시
        for i in range(0, len(special_formats), 8):
            line_formats = special_formats[i:i+8]
            print(f"    {', '.join(line_formats)}")
    
    print(f"\n💡 총 {len(SUPPORTED_FORMATS['output'])}개 포맷 지원")
    print("💡 일반 포맷 사용을 권장합니다 (jpg, png, bmp, tiff, webp 등)")
    print("=" * 70)
    
    # 출력 포맷 선택
    while True:
        output_format = input("\n🎯 변환할 출력 포맷을 입력하세요: ").lower().strip()
        
        if output_format in SUPPORTED_FORMATS['output']:
            # 선택한 포맷에 대한 정보 표시
            format_info = {
                'jpg': '🖼️  압축률 높음, 사진에 최적',
                'jpeg': '🖼️  압축률 높음, 사진에 최적', 
                'png': '🎨 무손실, 투명도 지원',
                'bmp': '📁 무손실, 큰 파일 크기',
                'gif': '🎬 애니메이션 지원, 256색 제한',
                'tiff': '🏢 무손실, 전문가용',
                'tif': '🏢 무손실, 전문가용',
                'webp': '🌐 구글 포맷, 웹 최적화',
                'pdf': '📄 문서 포맷',
                'ico': '🎯 아이콘 파일',
                'eps': '🖨️  벡터 그래픽',
                'pcx': '🖥️  PC 페인트브러시',
                'tga': '🎮 게임용 이미지'
            }
            
            if output_format in format_info:
                print(f"✅ 선택됨: {output_format.upper()} {format_info[output_format]}")
            else:
                print(f"✅ 선택됨: {output_format.upper()} (특수 포맷)")
                
            break
        else:
            print(f"❌ 지원하지 않는 포맷입니다: {output_format}")
            print("💡 위 목록에서 선택해주세요.")
    
    # 품질 설정 (압축 포맷의 경우)
    quality = 95
    if output_format in ['jpg', 'jpeg', 'webp']:
        while True:
            try:
                quality = int(input("📸 품질을 입력하세요 (1-100, 기본값 95): ") or "95")
                if 1 <= quality <= 100:
                    break
                print("❌ 품질은 1-100 사이의 값이어야 합니다.")
            except ValueError:
                print("❌ 숫자를 입력해주세요.")
    
    return output_format, quality
def convert_images_recursively(input_dir, output_dir, output_format, quality=95, flatten_structure=False):
    """입력 디렉토리의 이미지들을 변환합니다."""
    
    total_converted = 0
    total_failed = 0
    
    structure_mode = "평면화 (한 폴더에 모음)" if flatten_structure else "폴더 구조 유지"
    print(f"\n🔄 변환 시작: {output_format.upper()} 포맷으로 변환")
    print("=" * 70)
    print(f"📂 입력 경로: {input_dir}")
    print(f"📂 출력 경로: {output_dir}")
    print(f"📁 구조 모드: {structure_mode}")
    print("=" * 70)
    
    # 출력 디렉토리가 없으면 생성
    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 출력 폴더를 생성했습니다: {output_dir}")
    
    # 입력 디렉토리를 재귀적으로 탐색
    for root, dirs, files in os.walk(input_dir):
        # ⚠️ 중요: 출력 폴더가 입력 폴더 안에 있는 경우, 출력 폴더 자체는 처리하지 않음
        # 절대 경로로 정규화해서 비교
        normalized_root = os.path.normpath(os.path.abspath(root))
        normalized_output = os.path.normpath(os.path.abspath(output_dir))
        
        if normalized_root == normalized_output or normalized_root.startswith(normalized_output + os.sep):
            print(f"⏭️  출력 폴더는 건너뜀: {root}")
            continue
            
        # dirs에서도 출력 폴더명 제거 (재귀 탐색에서 제외)
        output_folder_name = os.path.basename(normalized_output)
        if output_folder_name in dirs:
            dirs.remove(output_folder_name)
            print(f"🚫 하위 탐색에서 출력 폴더 제외: {output_folder_name}")
            
        # 폴더 구조 처리
        if flatten_structure:
            # 평면화: 모든 이미지를 출력 디렉토리에 직접 저장
            current_output_dir = output_dir
            rel_display = f"평면화 → {os.path.basename(root) if root != input_dir else '루트'}"
        else:
            # 구조 유지: 기존 로직
            if root == input_dir:
                current_output_dir = output_dir
                rel_display = "루트 폴더"
            else:
                rel_path = root[len(input_dir):].lstrip(os.sep)
                current_output_dir = os.path.join(output_dir, rel_path)
                rel_display = rel_path
        
        # 출력 폴더 생성
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir, exist_ok=True)
        
        # 이미지 파일들 찾기
        image_files_in_folder = []
        for file_name in files:
            file_ext = os.path.splitext(file_name)[1].lower()
            if file_ext in SUPPORTED_FORMATS['input']:
                image_files_in_folder.append(file_name)
        
        if image_files_in_folder:
            print(f"📂 처리 중: {rel_display} ({len(image_files_in_folder)}개 이미지)")
            
            # 경로를 축약해서 표시 (터미널 너비 고려)
            input_display = root
            output_display = current_output_dir
            
            # 경로가 너무 길면 축약
            max_path_length = 80
            if len(input_display) > max_path_length:
                input_display = "..." + input_display[-(max_path_length-3):]
            if len(output_display) > max_path_length:
                output_display = "..." + output_display[-(max_path_length-3):]
            
            print(f"   📥 입력: {input_display}")
            print(f"   📤 출력: {output_display}")
            print()  # 빈 줄 추가로 가독성 향상
            
            for file_name in image_files_in_folder:
                input_file = os.path.join(root, file_name)
                output_file = os.path.join(current_output_dir, f"{os.path.splitext(file_name)[0]}.{output_format}")
                
                try:
                    with Image.open(input_file) as img:
                        # 포맷별 색상 모드 최적화
                        if output_format.lower() in ['jpg', 'jpeg']:
                            # JPEG는 RGBA나 P 모드 지원 안함
                            if img.mode in ['RGBA', 'P', 'LA']:
                                img = img.convert('RGB')
                        elif output_format.lower() in ['bmp']:
                            # BMP는 투명도 지원 안함
                            if img.mode in ['RGBA', 'LA']:
                                # 흰색 배경으로 합성
                                background = Image.new('RGB', img.size, (255, 255, 255))
                                if img.mode == 'RGBA':
                                    background.paste(img, mask=img.split()[-1])
                                else:
                                    background.paste(img)
                                img = background
                        elif output_format.lower() in ['gif']:
                            # GIF는 팔레트 모드 선호
                            if img.mode not in ['P', 'L']:
                                img = img.convert('P', palette=Image.ADAPTIVE)
                        elif output_format.lower() in ['ico']:
                            # ICO는 크기 제한
                            max_size = 256
                            if img.width > max_size or img.height > max_size:
                                img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
                        
                        # 저장 옵션 설정
                        save_kwargs = {}
                        
                        if output_format.lower() in ['jpg', 'jpeg']:
                            save_kwargs['quality'] = quality
                            save_kwargs['optimize'] = True
                        elif output_format.lower() in ['png']:
                            save_kwargs['optimize'] = True
                        elif output_format.lower() in ['webp']:
                            save_kwargs['quality'] = quality if output_format in ['jpg', 'jpeg'] else 90
                            save_kwargs['method'] = 6  # 최고 압축
                        elif output_format.lower() in ['tiff', 'tif']:
                            save_kwargs['compression'] = 'lzw'  # 무손실 압축
                        
                        # 파일 저장
                        img.save(output_file, SUPPORTED_FORMATS['output'][output_format], **save_kwargs)
                    
                    print(f"  ✅ {file_name} → {os.path.basename(output_file)}")
                    total_converted += 1
                    
                except Exception as e:
                    print(f"  ❌ {file_name} 변환 실패: {e}")
                    total_failed += 1
    
    print("=" * 70)
    print(f"🎉 변환 완료!")
    print(f"✅ 성공: {total_converted}개")
    print(f"❌ 실패: {total_failed}개")
    print("=" * 70)

# 메인 실행 부분
if __name__ == "__main__":
    print("=" * 60)
    print("🖼️  범용 이미지 포맷 변환기 (인터랙티브)")
    print("=" * 60)
    
    # 1. 입력 경로 탐색
    input_path = navigate_to_input_path()
    
    # 2. 출력 포맷 선택
    output_format, quality = get_output_format()
    
    # 3. 폴더 구조 옵션 선택
    print("\n📁 폴더 구조 선택")
    print("=" * 50)
    print("1️⃣  폴더 구조 유지 (하위 폴더들도 그대로 유지)")
    print("2️⃣  평면화 (모든 이미지를 한 폴더에 모음)")
    
    while True:
        structure_choice = input("\n선택하세요 (1 또는 2): ").strip()
        if structure_choice in ['1', '2']:
            break
        print("❌ 1 또는 2를 입력해주세요.")
    
    flatten_structure = (structure_choice == '2')
    
    # 4. 출력 경로 설정 (스마트한 중복 방지)
    # converted 중복 방지 로직
    if os.path.basename(input_path).lower() == "converted":
        print("⚠️  현재 경로가 'converted' 폴더입니다.")
        print("   부모 폴더를 기준으로 새로운 출력 폴더를 생성합니다.")
        parent_path = os.path.dirname(input_path)
        output_path = os.path.join(parent_path, "converted_new")
        actual_input_path = input_path
    elif "converted" in input_path.lower():
        print("⚠️  경로에 'converted'가 포함되어 있습니다.")
        print("   중복을 방지하기 위해 'converted_output' 폴더를 생성합니다.")
        output_path = os.path.join(input_path, "converted_output")
        actual_input_path = input_path
    else:
        output_path = os.path.join(input_path, "converted")
        actual_input_path = input_path
    
    print(f"\n📋 변환 설정 확인")
    print(f"📂 입력 경로: {actual_input_path}")
    print(f"📂 출력 경로: {output_path}")
    print(f"🎯 출력 포맷: {output_format.upper()}")
    print(f"📁 폴더 구조: {'평면화 (한 폴더에 모음)' if flatten_structure else '구조 유지'}")
    if output_format in ['jpg', 'jpeg', 'webp']:
        print(f"📸 품질: {quality}")
    
    confirm = input("\n✅ 변환을 시작하시겠습니까? (y/N): ").lower().strip()
    if confirm in ['y', 'yes']:
        # 5. 변환 실행 (평면화 옵션 포함)
        convert_images_recursively(actual_input_path, output_path, output_format, quality, flatten_structure)
    else:
        print("👋 변환을 취소했습니다.")
