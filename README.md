# 나의 다이어트 챌린지 앱

Flask 기반의 다이어트 챌린지 웹 애플리케이션입니다. 사용자가 칼로리 목표를 설정하고 음식을 기록하며, AI를 통해 음식 사진을 분석할 수 있습니다.

## 🚀 주요 기능

- **칼로리 챌린지**: 다양한 칼로리 목표 설정 (1000kcal ~ 2500kcal)
- **음식 기록**: 사진 업로드 및 수동 입력
- **AI 음식 분석**: RTX 5070 GPU 가속화를 활용한 음식 인식
- **진행 상황 추적**: 실시간 칼로리 섭취 현황
- **스탬프 시스템**: 목표 달성 시 보상
- **소셜 피드**: 다른 사용자들의 음식 기록 공유

## 🛠 기술 스택

### Frontend
- HTML5, CSS3, JavaScript
- 반응형 웹 디자인
- 모던 UI/UX

### Backend
- **Python 3.8+**
- **Flask 2.3.3** - 웹 프레임워크
- **SQLAlchemy** - ORM
- **SQLite** - 데이터베이스

### AI/ML
- **PyTorch** - 딥러닝 프레임워크
- **OpenCV** - 이미지 처리
- **Transformers** - 사전 훈련된 모델
- **RTX 5070 GPU** - 가속화

## 📦 설치 방법

### 1. 저장소 클론
```bash
git clone <repository-url>
cd diet-challenge-app
```

### 2. 가상환경 생성 및 활성화
```bash
# Anaconda 사용
conda create -n diet-challenge python=3.9
conda activate diet-challenge

# 또는 venv 사용
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. 의존성 설치
```bash
pip install -r requirements.txt
```

### 4. GPU 드라이버 확인 (선택사항)
RTX 5070 GPU를 사용하려면:
```bash
# CUDA 버전 확인
nvidia-smi

# PyTorch CUDA 설치 확인
python -c "import torch; print(torch.cuda.is_available())"
```

## 🚀 실행 방법

### 1. 데이터베이스 마이그레이션 실행 (새로운 comment 필드 추가)
```bash
python migrate_db.py
```

### 2. 데이터베이스 초기화 (처음 실행 시)
```bash
python init_db.py
```

### 3. Flask 서버 실행
```bash
python app.py
```

### 3. 웹 브라우저에서 접속
```
http://localhost:5000
```

## 📁 프로젝트 구조

```
diet-challenge-app/
├── app.py                 # Flask 메인 애플리케이션
├── config.py              # 설정 파일
├── init_db.py             # 데이터베이스 초기화
├── requirements.txt       # Python 의존성
├── README.md             # 프로젝트 문서
├── index.html            # 메인 페이지
├── my-challenges.html    # 챌린지 목록 페이지
├── challenge-detail.html # 챌린지 상세 페이지
├── uploads/              # 업로드된 이미지 저장소
├── models/               # AI 모델 파일 (향후 추가)
└── diet_challenge.db     # SQLite 데이터베이스
```

## 🗄 데이터베이스 스키마

### Users 테이블
- `id`: 사용자 고유 ID
- `username`: 사용자명
- `email`: 이메일
- `created_at`: 가입일

### Challenges 테이블
- `id`: 챌린지 고유 ID
- `user_id`: 사용자 ID (외래키)
- `target_calories`: 목표 칼로리
- `start_date`: 시작일
- `end_date`: 종료일
- `is_active`: 활성 상태

### FoodRecords 테이블
- `id`: 기록 고유 ID
- `user_id`: 사용자 ID (외래키)
- `challenge_id`: 챌린지 ID (외래키)
- `food_name`: 음식명
- `calories`: 칼로리
- `weight`: 중량 (g)
- `image_path`: 이미지 파일 경로
- `recorded_at`: 기록 시간
- `ai_analysis_result`: AI 분석 결과 (JSON)

### Stamps 테이블
- `id`: 스탬프 고유 ID
- `user_id`: 사용자 ID (외래키)
- `challenge_id`: 챌린지 ID (외래키)
- `stamp_type`: 스탬프 유형
- `earned_at`: 획득 시간
- `description`: 설명

## 🔌 API 엔드포인트

### 사용자 관리
- `POST /api/users` - 새 사용자 생성
- `GET /api/users/<id>` - 사용자 정보 조회

### 챌린지 관리
- `POST /api/challenges` - 새 챌린지 생성
- `GET /api/challenges/<id>` - 챌린지 정보 조회

### 음식 기록
- `POST /api/food-records` - 음식 기록 생성
- `GET /api/food-records/<challenge_id>` - 음식 기록 조회

### AI 분석
- `POST /api/analyze-food` - 음식 사진 AI 분석

### 스탬프
- `GET /api/stamps/<user_id>` - 사용자 스탬프 조회

## 🎯 AI 모델 (향후 구현)

### 음식 인식 모델
- **모델**: Vision Transformer (ViT)
- **데이터셋**: Food-101, Korean Food Dataset
- **GPU**: RTX 5070 CUDA 가속화
- **출력**: 음식명, 칼로리, 영양소 정보

### 칼로리 예측 모델
- **모델**: CNN + Regression
- **특징**: 음식 크기, 재료, 조리법 고려
- **정확도**: 85% 이상 목표

## 🔒 보안 개선 사항

### 최근 수정된 보안 이슈
1. **API 키 하드코딩 제거**: Gemini API 키를 환경 변수로 이동
2. **인증되지 않은 API 엔드포인트 보호**: `/api/food-records/all`에 JWT 인증 추가
3. **입력값 검증 강화**: 음식명 길이, 칼로리 범위 검증 추가
4. **파일 업로드 보안**: 파일 크기 및 내용 검증 추가
5. **데이터베이스 롤백**: 예외 발생 시 트랜잭션 롤백 추가

### 권장 보안 설정
- 강력한 SECRET_KEY 사용 (최소 32자)
- HTTPS 환경에서 운영
- 정기적인 의존성 업데이트
- 로그 파일 보안 관리

## 🔧 개발 환경 설정

### 환경 변수 설정
```bash
# env_example.txt를 .env로 복사하고 실제 값으로 수정
cp env_example.txt .env

# 필수 환경 변수
SECRET_KEY=your-super-secret-key-here-change-this
GEMINI_API_KEY=your-gemini-api-key-here

# 선택사항
DATABASE_URL=sqlite:///diet_challenge.db
GPU_ENABLED=true
FLASK_ENV=development
```

### Gemini API 할당량 초과 해결 방법

1. **새로운 API 키 생성**
   - [Google AI Studio](https://makersuite.google.com/app/apikey)에서 새로운 API 키 생성
   - 환경 변수 `GEMINI_API_KEY`에 새 키 설정

2. **할당량 확인**
   - 무료 등급: 하루 50회 요청 제한
   - 유료 플랜으로 업그레이드하여 제한 해제

3. **대체 방법**
   - API 할당량 초과 시 수동으로 음식명 입력
   - 내일 다시 AI 분석 시도

### 개발 서버 실행
```bash
export FLASK_ENV=development
export FLASK_APP=app.py
flask run --host=0.0.0.0 --port=5000
```

## 🧪 테스트

```bash
# 단위 테스트 실행
python -m pytest tests/

# API 테스트
curl http://localhost:5000/api/challenges/1
```

## 📈 성능 최적화

### GPU 가속화
- CUDA 메모리 관리
- 배치 처리 최적화
- 모델 양자화 (INT8)

### 데이터베이스 최적화
- 인덱스 설정
- 쿼리 최적화
- 연결 풀링

## 🤝 기여하기

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다.

## 📞 문의

프로젝트에 대한 문의사항이 있으시면 이슈를 생성해주세요. 