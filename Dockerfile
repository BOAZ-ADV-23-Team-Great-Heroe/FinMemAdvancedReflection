# project/Dockerfile
# Python 3.9 버전의 슬림(경량) Debian 이미지를 기반으로 사용합니다.
FROM python:3.10.17

# 컨테이너 내에서 애플리케이션 작업 디렉토리를 /app으로 설정합니다.
WORKDIR /app

# 로컬의 requirements.txt 파일을 컨테이너의 /app 디렉토리로 복사합니다.
COPY requirements.txt .

# requirements.txt에 명시된 모든 Python 패키지를 설치합니다.
# --no-cache-dir 옵션은 캐시 파일을 저장하지 않아 이미지 크기를 줄입니다.
RUN pip install --no-cache-dir -r requirements.txt

# 로컬의 모든 파일을 컨테이너의 /app 디렉토리로 복사합니다.
# 이 경우, 'project' 디렉토리의 모든 내용(app 디렉토리 포함)이 /app으로 복사됩니다.
COPY . .

# 컨테이너가 시작될 때 실행될 명령어를 정의합니다.
# Uvicorn을 사용하여 'app' 디렉토리 내의 'main.py' 파일에 정의된 'app' 인스턴스를 실행합니다.
# --host 0.0.0.0은 모든 네트워크 인터페이스에서 접근 가능하도록 하고,
# --port 8080은 Cloud Run의 기본 포트입니다.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]