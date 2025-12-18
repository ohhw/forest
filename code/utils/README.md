# 🔍 공통 유틸리티 (utils)

프로젝트 전체에서 사용하는 공통 함수 모음

## 📦 모듈 구조

```
utils/
├── __init__.py           # 패키지 초기화
├── time_utils.py         # 시간 관련 함수
├── path_utils.py         # 경로 관련 함수
├── file_utils.py         # 파일 입출력 함수
└── image_utils.py        # 이미지 관련 함수
```

## 🚀 사용법

### 전체 import
```python
from utils import *

# 또는 선택적 import
from utils import format_time, ensure_dir, collect_images
```

### 개별 모듈 import
```python
from utils.time_utils import format_time
from utils.path_utils import ensure_dir
from utils.image_utils import collect_images
```

## 📚 함수 목록

### ⏰ time_utils.py

#### `format_time(seconds: float) -> str`
초를 "X시간 Y분 Z초" 형식으로 변환

```python
from utils import format_time

elapsed = 3665
print(format_time(elapsed))  # "1시간 1분 5초"
```

#### `format_timestamp(format_str: str) -> str`
현재 시간을 지정된 형식으로 반환 (KST 기준)

```python
from utils import format_timestamp

timestamp = format_timestamp()  # "20250101_153045"
readable = format_timestamp("%Y-%m-%d %H:%M:%S")  # "2025-01-01 15:30:45"
```

---

### 📁 path_utils.py

#### `ensure_dir(path: str, create: bool = True) -> str`
디렉토리가 없으면 생성하고 절대 경로 반환

```python
from utils import ensure_dir

output_dir = ensure_dir("/path/to/output")
print(output_dir)  # "/path/to/output" (생성됨)
```

#### `get_absolute_path(path: str) -> str`
상대 경로를 절대 경로로 변환 (~, . 처리)

```python
from utils import get_absolute_path

abs_path = get_absolute_path("~/data")  # "/home/user/data"
abs_path = get_absolute_path("./images")  # "/current/dir/images"
```

#### `check_path_exists(path: str, path_type: str = 'any') -> bool`
경로 존재 여부 확인

```python
from utils import check_path_exists

exists = check_path_exists("/path/to/file", path_type="file")
exists = check_path_exists("/path/to/dir", path_type="dir")
```

---

### 💾 file_utils.py

#### `read_file_lines(file_path: str, strip: bool = True) -> List[str]`
파일을 줄 단위로 읽기

```python
from utils import read_file_lines

lines = read_file_lines("train.txt", skip_empty=True)
for line in lines:
    print(line)
```

#### `write_file_lines(file_path: str, lines: List[str]) -> bool`
리스트를 파일에 쓰기

```python
from utils import write_file_lines

lines = ["/path/to/img1.jpg", "/path/to/img2.jpg"]
success = write_file_lines("output.txt", lines)
```

#### `backup_file(file_path: str, backup_dir: str = None) -> str`
파일 백업 (타임스탬프 추가)

```python
from utils import backup_file

backup_path = backup_file("important.txt", backup_dir="/backups")
print(f"백업됨: {backup_path}")
```

---

### 🖼️ image_utils.py

#### `collect_images(directory: str, recursive: bool = False) -> List[str]`
디렉토리에서 이미지 파일 수집

```python
from utils import collect_images

# 현재 폴더만
images = collect_images("/path/to/images")

# 하위 폴더까지
images = collect_images("/path/to/images", recursive=True)

print(f"총 {len(images)}개 이미지 발견")
```

#### `is_image_file(file_path: str) -> bool`
파일이 이미지인지 확인

```python
from utils import is_image_file

if is_image_file("photo.jpg"):
    print("이미지 파일입니다")
```

#### `get_image_extension_stats(directory: str) -> dict`
확장자별 이미지 통계

```python
from utils import get_image_extension_stats

stats = get_image_extension_stats("/path/to/images")
print(stats)  # {'.jpg': 150, '.png': 50, '.bmp': 20}
```

#### `IMAGE_EXTENSIONS`
지원하는 이미지 확장자 집합

```python
from utils import IMAGE_EXTENSIONS

print(IMAGE_EXTENSIONS)
# {'.jpg', '.png', '.bmp', '.tif', '.tiff', '.webp', '.gif', ...}
```

---

## 🎯 실전 예제

### 예제 1: 이미지 수집 후 train.txt 생성
```python
from utils import collect_images, write_file_lines, ensure_dir

# 이미지 수집
images = collect_images("/hdd/datasets/jjb/v10/train/images", recursive=True)

# 출력 디렉토리 생성
ensure_dir("/hdd/datasets/jjb/v10")

# train.txt 저장
write_file_lines("/hdd/datasets/jjb/v10/train.txt", images)
print(f"{len(images)}개 이미지 경로 저장 완료")
```

### 예제 2: 학습 시간 측정
```python
import time
from utils import format_time

start = time.time()

# 학습 코드
train_model()

end = time.time()
elapsed = format_time(end - start)
print(f"학습 시간: {elapsed}")
```

### 예제 3: 안전한 파일 백업
```python
from utils import backup_file, write_file_lines

# 백업
backup_path = backup_file("config.yaml", backup_dir="/backups")
print(f"백업 완료: {backup_path}")

# 새로운 내용 저장
new_config = ["epochs: 100", "batch: 32"]
write_file_lines("config.yaml", new_config)
```

---

## 🔧 확장 방법

새로운 유틸리티 함수 추가:

1. 적절한 모듈 파일에 함수 작성
2. `__init__.py`의 `__all__`에 추가
3. 이 README 업데이트

```python
# utils/my_utils.py
def my_function():
    pass

# utils/__init__.py
from .my_utils import my_function
__all__ = [..., 'my_function']
```

---

## 📝 주의사항

- 모든 경로는 자동으로 절대 경로로 변환됩니다
- 디렉토리는 필요시 자동 생성됩니다
- 파일 백업은 타임스탬프가 자동으로 추가됩니다
- 이미지 확장자는 대소문자 구분 없이 처리됩니다

---

## 🐛 문제 해결

### ImportError 발생 시
```bash
# code 디렉토리를 Python 경로에 추가
export PYTHONPATH="/home/hwoh/code:$PYTHONPATH"
```

또는 스크립트에서:
```python
import sys
sys.path.insert(0, '/home/hwoh/code')
from utils import *
```
