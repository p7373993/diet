from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from datetime import datetime, date, timedelta
import os
import json
from werkzeug.utils import secure_filename
import torch
from config import Config
from flask_bcrypt import Bcrypt
import jwt
from functools import wraps
import google.generativeai as genai
from torchvision import transforms, models
from PIL import Image
import pandas as pd

# Flask 앱 초기화
app = Flask(__name__)
app.config.from_object('config.Config')  # config.py 사용

# CORS 설정 (프론트엔드와 통신)
CORS(app)

# 데이터베이스 초기화
db = SQLAlchemy(app)

# GPU 사용 가능 여부 확인
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# 파일 업로드 설정
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    """파일 확장자 검증"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def validate_file_size(file_size):
    """파일 크기 검증 (16MB 제한)"""
    max_size = 16 * 1024 * 1024  # 16MB
    return file_size <= max_size

def validate_image_content(file_path):
    """이미지 파일 내용 검증"""
    try:
        from PIL import Image
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False

# 데이터베이스 모델 정의
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)
    is_public = db.Column(db.Boolean, default=False)  # 프로필 공개 여부
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    challenges = db.relationship('Challenge', backref='user', lazy=True)
    food_records = db.relationship('FoodRecord', backref='user', lazy=True)

class Challenge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    target_calories = db.Column(db.Integer, nullable=False)
    start_date = db.Column(db.DateTime, default=datetime.utcnow)
    end_date = db.Column(db.DateTime, nullable=False)  # 챌린지 종료일
    is_active = db.Column(db.Boolean, default=True)
    status = db.Column(db.String(50), default='active')  # active, success, fail
    food_records = db.relationship('FoodRecord', backref='challenge', lazy=True)

class FoodRecord(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    challenge_id = db.Column(db.Integer, db.ForeignKey('challenge.id'), nullable=False)
    food_name = db.Column(db.String(200), nullable=False)
    calories = db.Column(db.Integer, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    image_path = db.Column(db.String(500), nullable=True)
    comment = db.Column(db.Text, nullable=True)  # 사용자 메모
    recorded_at = db.Column(db.DateTime, default=datetime.utcnow)
    ai_analysis_result = db.Column(db.Text, nullable=True)  # AI 분석 결과 JSON
    advice = db.Column(db.Text, nullable=True)  # Gemini LLM 조언

class Stamp(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    challenge_id = db.Column(db.Integer, db.ForeignKey('challenge.id'), nullable=False)
    stamp_type = db.Column(db.String(50), nullable=False)  # 'daily_goal', 'weekly_goal', etc.
    earned_at = db.Column(db.DateTime, default=datetime.utcnow)
    description = db.Column(db.String(200), nullable=True)

# 데이터베이스 생성
with app.app_context():
    db.create_all()

# Gemini Vision 기반 음식명 리스트 및 칼로리 매핑 (CSV 기반 동적 생성)
def load_food_calorie_map():
    fallback_map = {
        '김치찌개': 150, '비빔밥': 180, '피자': 270, '햄버거': 250, '샐러드': 60, '초밥': 140, '치킨': 230, '파스타': 160, '스테이크': 250, '샌드위치': 220, '라면': 440, '떡볶이': 220, '샤브샤브': 90, '된장찌개': 70, '불고기': 215, '삼겹살': 340, '볶음밥': 180, '우동': 95, '순두부찌개': 60, '갈비': 350, '고등어구이': 180, '프레첼': 340, '기타 음식': 200
    }
    try:
        excel_path = os.path.join(os.path.dirname(__file__), '음식분류 AI 데이터 영양DB.xlsx')
        if not os.path.exists(excel_path):
            excel_path = os.path.join(os.getcwd(), '음식분류 AI 데이터 영양DB.xlsx')
        df = pd.read_excel(excel_path)
        food_map = {}
        for _, row in df.iterrows():
            try:
                name = str(row['음식명']).strip()
                kcal = float(row['열량(kcal)'])
                gram = float(row['총 내용량(g)'])
                if not name or pd.isnull(kcal) or pd.isnull(gram) or gram == 0:
                    continue
                kcal_per_100g = round((kcal / gram) * 100, 1)
                food_map[name] = kcal_per_100g
            except Exception:
                continue
        food_map['기타 음식'] = 200
        if len(food_map) < 10:
            return fallback_map
        return food_map
    except Exception as e:
        print(f"[FOOD_CALORIE_MAP 로딩 오류] {e}")
        return fallback_map

FOOD_CALORIE_MAP = load_food_calorie_map()
FOOD_LABELS = list(FOOD_CALORIE_MAP.keys())

class FoodAnalysisModel:
    def __init__(self):
        from config import Config
        api_key = Config.GEMINI_API_KEY
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("Gemini 1.5 Flash 모델이 성공적으로 로드되었습니다.")
        except Exception as e:
            print(f"[ERROR] Failed to load Gemini 1.5 Flash model: {e}")
            self.model = None

    def analyze_food_image(self, image_path):
        if self.model is None:
            print("[ERROR] Gemini 1.5 Flash model not loaded or failed to load.")
            return {
                'food_name': 'AI 분석 실패',
                'calories': 0,
                'confidence': 0.0,
                'ingredients': [],
                'nutrition': {},
                'error': 'Gemini 1.5 Flash 모델이 로드되지 않았거나 오류가 발생했습니다.'
            }
        try:
            prompt = "이 사진에 어떤 음식이 있나요? 음식명만 알려주세요. 음식이 아니거나 알 수 없으면 '알 수 없는 음식'이라고 답해주세요."
            response = self.model.generate_content([
                prompt,
                Image.open(image_path)
            ])
            # Gemini Vision 응답 robust 파싱
            food_name = None
            confidence = 0.0
            text = None
            # 다양한 응답 케이스에 대응
            if hasattr(response, 'text') and response.text:
                text = response.text.strip()
            elif hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'text'):
                text = response.candidates[0].text.strip()
            elif hasattr(response, 'parts') and response.parts and hasattr(response.parts[0], 'text'):
                text = response.parts[0].text.strip()
            elif hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'content') and hasattr(response.candidates[0].content, 'parts') and response.candidates[0].content.parts and hasattr(response.candidates[0].content.parts[0], 'text'):
                text = response.candidates[0].content.parts[0].text.strip()
            # 음식명만 파싱 (예: '이 이미지에는 [음식명]이 있습니다.'에서 [음식명]만 추출)
            if text:
                # 한 줄만 추출
                first_line = text.split('\n')[0]
                # '알 수 없는 음식' 처리
                if '알 수 없는 음식' in first_line:
                    food_name = '알 수 없는 음식'
                    confidence = 0.3
                else:
                    # 예: '이 이미지에는 피자가 있습니다.' → '피자'만 추출
                    import re
                    m = re.search(r'([가-힣a-zA-Z]+)', first_line)
                    if m:
                        food_name = m.group(1)
                        confidence = 0.9
                    else:
                        food_name = first_line.strip()
                        confidence = 0.9
            else:
                food_name = '알 수 없는 음식'
                confidence = 0.3
            calories = FOOD_CALORIE_MAP.get(food_name, FOOD_CALORIE_MAP['기타 음식'])
            return {
                'food_name': food_name,
                'calories': calories,
                'confidence': confidence,
                'ingredients': [],
                'nutrition': {}
            }
        except Exception as e:
            print(f"Gemini API Vision 오류: {e}")
            # 할당량 초과 오류인지 확인
            if "quota" in str(e).lower() or "429" in str(e):
                return {
                    'food_name': 'AI 할당량 초과',
                    'calories': 0,
                    'confidence': 0.0,
                    'ingredients': [],
                    'nutrition': {},
                    'error': 'API 할당량이 초과되었습니다. 수동으로 음식명을 입력해주세요.'
                }
            else:
                return {
                    'food_name': 'AI 분석 오류',
                    'calories': 0,
                    'confidence': 0.0,
                    'ingredients': [],
                    'nutrition': {},
                    'error': str(e)
                }

# AI 모델 인스턴스 생성
food_analyzer = FoodAnalysisModel()

bcrypt = Bcrypt(app)
JWT_SECRET = app.config.get('SECRET_KEY', 'jwt-secret')
JWT_ALGORITHM = 'HS256'
JWT_EXPIRE_HOURS = 24

# JWT 인증 데코레이터
def jwt_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        if not token:
            return jsonify({'success': False, 'error': '인증이 필요합니다.'}), 401
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            # JWT 페이로드에서 user_id를 추출하여 함수 인자로 전달
            kwargs['current_user_id'] = payload['user_id']
        except Exception:
            return jsonify({'success': False, 'error': '유효하지 않은 토큰입니다.'}), 401
        return f(*args, **kwargs)
    return decorated

# API 엔드포인트들

@app.route('/')
def index():
    return jsonify({'message': 'Diet Challenge API is running!'})

@app.route('/api/users', methods=['POST'])
def create_user():
    """새 사용자 생성"""
    data = request.get_json()
    
    try:
        new_user = User(
            username=data['username'],
            email=data['email']
        )
        db.session.add(new_user)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'user_id': new_user.id,
            'message': 'User created successfully'
        }), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/challenges', methods=['POST'])
@jwt_required
def create_challenge(current_user_id):
    """새 챌린지 생성 (단, 활성 챌린지가 없을 때만)"""
    # 이미 활성화된 챌린지가 있는지 확인 (만료되지 않은 챌린지만)
    existing_challenge = Challenge.query.filter(
        Challenge.user_id == current_user_id,
        Challenge.is_active == True,
        Challenge.end_date > datetime.utcnow()
    ).first()
    if existing_challenge:
        return jsonify({
            'success': False, 
            'error': '이미 진행 중인 챌린지가 있습니다. 새로운 챌린지를 시작하려면 기존 챌린지를 완료하거나 삭제해주세요.'
        }), 400

    data = request.get_json()
    try:
        target_calories = data['target_calories']
        start_date = datetime.utcnow()
        end_date = start_date + timedelta(days=7) # 7일 챌린지

        new_challenge = Challenge(
            user_id=current_user_id,
            target_calories=target_calories,
            start_date=start_date,
            end_date=end_date,
            is_active=True,
            status='active'
        )
        db.session.add(new_challenge)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'challenge_id': new_challenge.id,
            'message': 'Challenge created successfully'
        }), 201
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/challenges/<int:challenge_id>', methods=['GET', 'DELETE'])
@jwt_required
def manage_challenge(current_user_id, challenge_id):
    """챌린지 정보 조회 또는 삭제"""
    challenge = Challenge.query.filter_by(id=challenge_id, user_id=current_user_id).first_or_404()

    # 챌린지 상태 업데이트 (만료된 경우)
    if challenge.is_active and datetime.utcnow() > challenge.end_date:
        challenge.is_active = False
        # 챌린지 성공/실패 판정 로직 (주 5일 이상 목표 달성)
        daily_success_count = 0
        for i in range(7):
            check_date = challenge.start_date + timedelta(days=i)
            daily_calories = sum(r.calories for r in FoodRecord.query.filter(
                FoodRecord.challenge_id == challenge_id,
                db.func.date(FoodRecord.recorded_at) == check_date.date()
            ).all())
            if daily_calories <= challenge.target_calories:
                daily_success_count += 1
        
        # 챌린지 성공 기준: 7일 중 6일 이상 목표 달성 (85% 이상)
        success_threshold = 6  # 7일 중 6일 이상
        if daily_success_count >= success_threshold:
            challenge.status = 'success'
            # 주간 성공 스탬프 지급
            new_stamp = Stamp(
                user_id=current_user_id,
                challenge_id=challenge_id,
                stamp_type='weekly_success',
                description=f'주간 챌린지 성공! ({daily_success_count}/7일 달성)',
                earned_at=datetime.utcnow()
            )
            db.session.add(new_stamp)
            # 퍼펙트 위크 스탬프 (7일 모두 성공)
            if daily_success_count == 7:
                perfect_stamp = Stamp(
                    user_id=current_user_id,
                    challenge_id=challenge_id,
                    stamp_type='perfect_week',
                    description='퍼펙트 위크 달성!',
                    earned_at=datetime.utcnow()
                )
                db.session.add(perfect_stamp)
        else:
            challenge.status = 'fail'
        db.session.commit()

    if request.method == 'GET':
        return jsonify({
            'success': True,
            'challenge': {
                'id': challenge.id,
                'target_calories': challenge.target_calories,
                'start_date': challenge.start_date.isoformat(),
                'end_date': challenge.end_date.isoformat(),
                'is_active': challenge.is_active,
                'status': challenge.status
            }
        })

    elif request.method == 'DELETE':
        # 연결된 모든 데이터 삭제 후 챌린지 삭제
        try:
            FoodRecord.query.filter_by(challenge_id=challenge_id).delete()
            Stamp.query.filter_by(challenge_id=challenge_id).delete()
            db.session.delete(challenge)
            db.session.commit()
            return jsonify({'success': True, 'message': '챌린지가 성공적으로 삭제되었습니다.'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': f'챌린지 삭제 중 오류 발생: {str(e)}'}), 500

@app.route('/api/challenges/<int:challenge_id>/today-records', methods=['GET'])
@jwt_required
def get_today_records(current_user_id, challenge_id):
    """특정 챌린지의 오늘 음식 기록 조회"""
    # 챌린지 권한 확인
    challenge = Challenge.query.filter_by(id=challenge_id, user_id=current_user_id).first()
    if not challenge:
        return jsonify({'success': False, 'error': '챌린지를 찾을 수 없거나 권한이 없습니다.'}), 404

    today = datetime.now().date()
    records = FoodRecord.query.filter(
        FoodRecord.challenge_id == challenge_id,
        db.func.date(FoodRecord.recorded_at) == today
    ).order_by(FoodRecord.recorded_at.desc()).all()
    
    # N+1 쿼리 문제 해결을 위해 사용자 정보를 미리 조회
    user_ids = list(set(record.user_id for record in records))
    users = {user.id: user for user in User.query.filter(User.id.in_(user_ids)).all()}
    
    records_data = []
    for record in records:
        user = users.get(record.user_id)
        records_data.append({
            'id': record.id,
            'food_name': record.food_name,
            'calories': record.calories,
            'weight': record.weight,
            'image_url': f'/uploads/{record.image_path}' if record.image_path else None,
            'comment': record.comment,  # 사용자 메모
            'advice': record.advice,  # Gemini 조언
            'created_at': record.recorded_at.isoformat(),
            'recorded_at': record.recorded_at.isoformat(),  # 호환성을 위해 둘 다 포함
            'user_name': user.username if user else 'Unknown'
        })
    
    return jsonify({
        'success': True,
        'records': records_data
    })

@app.route('/api/challenges/<int:challenge_id>/records', methods=['POST'])
@jwt_required
def create_challenge_record(current_user_id, challenge_id):
    """특정 챌린지에 음식 기록 생성"""
    # 챌린지 권한 확인
    challenge = Challenge.query.filter_by(id=challenge_id, user_id=current_user_id).first()
    if not challenge:
        return jsonify({'success': False, 'error': '챌린지를 찾을 수 없거나 권한이 없습니다.'}), 404

    if not challenge.is_active:
        return jsonify({'success': False, 'error': '종료된 챌린지에는 기록할 수 없습니다.'}), 400

    try:
        data = request.get_json()
        food_name = data.get('food_name', '').strip()
        comment = data.get('comment', '').strip()
        image_data = data.get('image_data')
        input_calories = data.get('calories')  # 프론트에서 보낸 값

        # 입력값 검증
        if not food_name:
            return jsonify({'success': False, 'error': '음식 이름이 필요합니다.'}), 400
        
        if len(food_name) > 200:
            return jsonify({'success': False, 'error': '음식 이름이 너무 깁니다. (최대 200자)'}), 400
        
        if input_calories is not None:
            try:
                input_calories = int(input_calories)
                if input_calories < 0 or input_calories > 10000:
                    return jsonify({'success': False, 'error': '칼로리는 0-10000 사이의 값이어야 합니다.'}), 400
            except (ValueError, TypeError):
                return jsonify({'success': False, 'error': '칼로리는 숫자여야 합니다.'}), 400
        else:
            input_calories = 0  # None인 경우 0으로 설정

        # 이미지 저장
        filename = None
        if image_data:
            try:
                # base64 디코딩
                import base64
                image_bytes = base64.b64decode(image_data)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_food.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                
                # AI 분석 수행
                ai_result = food_analyzer.analyze_food_image(filepath)
            except Exception as e:
                print(f"이미지 처리 오류: {e}")
                ai_result = None
        else:
            ai_result = None

        # 칼로리 결정 로직: 프론트엔드 입력값 우선, AI 분석 결과는 백업
        if input_calories is not None and input_calories > 0:
            # 프론트엔드에서 칼로리를 입력한 경우
            calories = input_calories
        elif ai_result and 'calories' in ai_result and ai_result['food_name'] != 'AI 분석 실패':
            # AI 분석 결과가 있는 경우
            calories = ai_result['calories']
        else:
            # 기본 칼로리 값 사용
            calories = FOOD_CALORIE_MAP.get(food_name, FOOD_CALORIE_MAP['기타 음식'])
        
        print(f"DEBUG: 음식명={food_name}, 입력칼로리={input_calories}, 최종칼로리={calories}")

        # 오늘 총 칼로리 계산 (새로 추가할 칼로리는 제외)
        today = datetime.now().date()
        today_records = FoodRecord.query.filter(
            FoodRecord.challenge_id == challenge_id,
            db.func.date(FoodRecord.recorded_at) == today
        ).all()
        total_calories = sum(record.calories for record in today_records)

        # Gemini LLM 조언 생성
        advice = get_gemini_advice(food_name, calories, 100, ai_result, challenge.target_calories, total_calories)

        # 음식 기록 저장
        new_record = FoodRecord(
            user_id=current_user_id,
            challenge_id=challenge_id,
            food_name=food_name,
            calories=calories,
            weight=100,  # 기본값
            image_path=filename,
            comment=comment,  # 사용자 메모 추가
            ai_analysis_result=json.dumps(ai_result) if ai_result else None,
            advice=advice
        )
        db.session.add(new_record)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '음식이 성공적으로 기록되었습니다.',
            'record_id': new_record.id
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': f'음식 기록 생성 중 오류 발생: {str(e)}'}), 500

@app.route('/api/my-challenges', methods=['GET'])
@jwt_required
def get_my_challenges(current_user_id):
    """로그인한 사용자의 모든 챌린지 목록 반환"""
    challenges = Challenge.query.filter_by(user_id=current_user_id).order_by(Challenge.start_date.desc()).all()
    challenges_data = []
    for challenge in challenges:
        challenges_data.append({
            'id': challenge.id,
            'target_calories': challenge.target_calories,
            'start_date': challenge.start_date.isoformat(),
            'is_active': challenge.is_active
        })
    return jsonify({'success': True, 'challenges': challenges_data})

# Gemini LLM 예시 함수 (실제 API 연동)
def get_gemini_advice(food_name, calories, weight, ai_result, goal_calories, total_calories):
    from config import Config
    api_key = Config.GEMINI_API_KEY
    genai.configure(api_key=api_key)

    # 남은 칼로리 계산 (음수면 0)
    remaining_calories = max(goal_calories - total_calories, 0)
    
    # 프롬프트: 최대한 단순하고 직접적으로 구체적인 조언을 요청
    prompt = f"""
    당신은 전문 영양사입니다.
    사용자가 오늘 '{food_name}' ({weight}g, {calories}kcal)를 섭취했습니다.
    오늘의 목표 칼로리는 {goal_calories}kcal이며, 현재까지 총 {total_calories}kcal를 섭취했습니다.
    
    이 정보를 바탕으로, 다음 질문에 2~3문장으로 간결하게 답해주세요:
    1. '{food_name}'의 주요 영양적 특징과 다이어트 시 섭취에 대한 구체적인 조언.
    2. 남은 {remaining_calories}kcal 내에서 섭취할 수 있는 다음 식사(예: 저녁 또는 간식) 메뉴를 1~2가지 구체적으로 추천해 주세요.
    """
    
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt, stream=False)

        # --- Gemini LLM 디버그 로그 ---
        print(f"\n--- Gemini LLM 프롬프트 ---\n{prompt}\n---\n")
        print(f"--- Gemini LLM 응답 (원본) ---\n{response}\n---\n")
        # robust하게 텍스트 추출
        advice_text = ""
        # 1. response.text
        if hasattr(response, 'text') and response.text:
            advice_text = response.text.strip()
        # 2. response.candidates[0].text
        elif hasattr(response, 'candidates') and response.candidates and hasattr(response.candidates[0], 'text') and response.candidates[0].text:
            advice_text = response.candidates[0].text.strip()
        # 3. 기타 fallback (향후 확장 가능)
        # 디버그 로그
        print(f"--- Gemini LLM 최종 추출 조언 ---\n{advice_text}\n---\n")
        # 응답이 비어있거나 너무 짧으면 Fallback 메시지 반환
        if not advice_text or len(advice_text) < 20:
            return "Gemini가 이 음식에 대한 충분한 조언을 생성하지 못했습니다. 일반적인 식단 조절 원칙을 참고하세요."
        return advice_text
    except Exception as e:
        print(f"Gemini API 오류: {e}")
        # 할당량 초과 오류인지 확인
        if "quota" in str(e).lower() or "429" in str(e):
            return "API 할당량이 초과되었습니다. 일반적인 식단 조절 원칙을 참고하세요."
        else:
            return "Gemini API 호출 중 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."

# 기존 /api/food-records API 제거 - /api/challenges/<id>/records로 통합됨

@app.route('/api/food-records/<int:record_id>', methods=['PUT', 'DELETE'])
@jwt_required
def manage_food_record(current_user_id, record_id):
    """음식 기록 수정 또는 삭제"""
    record = FoodRecord.query.filter_by(id=record_id, user_id=current_user_id).first_or_404()

    if request.method == 'PUT':
        data = request.get_json()
        record.food_name = data.get('food_name', record.food_name)
        record.calories = data.get('calories', record.calories)
        record.weight = data.get('weight', record.weight)
        # 이미지 경로는 수정하지 않음 (새 이미지 업로드는 별도 로직 필요)
        try:
            db.session.commit()
            return jsonify({'success': True, 'message': '음식 기록이 성공적으로 업데이트되었습니다.'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 500

    elif request.method == 'DELETE':
        try:
            db.session.delete(record)
            db.session.commit()
            return jsonify({'success': True, 'message': '음식 기록이 성공적으로 삭제되었습니다.'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/food-records', methods=['GET'])
@jwt_required
def get_food_records_query(current_user_id):
    """특정 사용자/챌린지의 음식 기록 조회 (최신순)"""
    challenge_id = request.args.get('challenge_id', type=int)
    query = FoodRecord.query.filter_by(user_id=current_user_id) # JWT의 user_id로 필터링
    if challenge_id:
        query = query.filter_by(challenge_id=challenge_id)
    records = query.order_by(FoodRecord.recorded_at.desc()).all()
    records_data = []
    for record in records:
        records_data.append({
            'id': record.id,
            'food_name': record.food_name,
            'calories': record.calories,
            'weight': record.weight,
            'recorded_at': record.recorded_at.isoformat(),
            'created_at': record.recorded_at.isoformat(),  # 호환성을 위해 둘 다 포함
            'image_path': record.image_path,
            'comment': record.comment,  # 사용자 메모
            'ai_analysis': json.loads(record.ai_analysis_result) if record.ai_analysis_result else None,
            'advice': record.advice
        })
    return jsonify({'success': True, 'records': records_data})

@app.route('/api/food-records/all', methods=['GET'])
@jwt_required
def get_all_food_records(current_user_id):
    """공개 설정된 사용자들의 음식 기록을 최신순으로 반환"""
    # 공개 설정된 사용자들의 기록만 조회
    records = db.session.query(FoodRecord).join(User).filter(
        User.is_public == True
    ).order_by(FoodRecord.recorded_at.desc()).all()
    records_data = []
    for record in records:
        records_data.append({
            'id': record.id,
            'user_id': record.user_id,
            'challenge_id': record.challenge_id,
            'food_name': record.food_name,
            'calories': record.calories,
            'weight': record.weight,
            'recorded_at': record.recorded_at.isoformat(),
            'created_at': record.recorded_at.isoformat(),  # 호환성을 위해 둘 다 포함
            'image_path': record.image_path,
            'comment': record.comment,  # 사용자 메모
            'ai_analysis': json.loads(record.ai_analysis_result) if record.ai_analysis_result else None,
            'advice': record.advice
        })
    return jsonify({'success': True, 'records': records_data})

@app.route('/api/stamps/<int:user_id>', methods=['GET'])
@jwt_required
def get_user_stamps(current_user_id, user_id):
    """사용자의 스탬프 조회 (자신의 것만)"""
    if current_user_id != user_id:
        return jsonify({'success': False, 'error': '권한이 없습니다.'}), 403

    stamps = Stamp.query.filter_by(user_id=current_user_id).order_by(Stamp.earned_at.desc()).all()
    
    stamps_data = []
    for stamp in stamps:
        stamp_data = {
            'id': stamp.id,
            'stamp_type': stamp.stamp_type,
            'description': stamp.description,
            'earned_at': stamp.earned_at.isoformat()
        }
        stamps_data.append(stamp_data)
    
    return jsonify({
        'success': True,
        'stamps': stamps_data
    })

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """업로드된 파일 제공"""
    if not filename:
        return jsonify({'error': 'No filename provided'}), 400
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/api/analyze-food', methods=['POST'])
@jwt_required
def analyze_food(current_user_id):
    """음식 사진 AI 분석 (파일 업로드 또는 base64 데이터)"""
    try:
        # 파일 업로드 방식
        if 'image' in request.files:
            file = request.files['image']
            if file and file.filename and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_{filename}"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # AI 분석 수행
                result = food_analyzer.analyze_food_image(filepath)
                
                if result:
                    return jsonify({
                        'success': True,
                        'analysis': result,
                        'image_path': filename
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'AI analysis failed'
                    }), 500
        
        # base64 데이터 방식
        elif request.is_json:
            data = request.get_json()
            image_data = data.get('image_data')
            
            if not image_data:
                return jsonify({'success': False, 'error': 'No image data provided'}), 400
            
            try:
                # base64 디코딩
                import base64
                image_bytes = base64.b64decode(image_data)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"{timestamp}_analysis.jpg"
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                
                with open(filepath, 'wb') as f:
                    f.write(image_bytes)
                
                # AI 분석 수행
                result = food_analyzer.analyze_food_image(filepath)
                
                if result:
                    return jsonify({
                        'success': True,
                        'analysis': result,
                        'image_path': filename
                    })
                else:
                    return jsonify({
                        'success': False,
                        'error': 'AI analysis failed'
                    }), 500
                    
            except Exception as e:
                return jsonify({
                    'success': False,
                    'error': f'Image processing error: {str(e)}'
                }), 500
        
        else:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Analysis error: {str(e)}'
        }), 500

@app.route('/index.html')
def serve_index():
    return send_from_directory('.', 'index.html')

@app.route('/my-challenges.html')
def serve_my_challenges():
    return send_from_directory('.', 'my-challenges.html')

@app.route('/challenge-detail.html')
def serve_challenge_detail():
    return send_from_directory('.', 'challenge-detail.html')

@app.route('/api/calories-today', methods=['GET'])
@jwt_required
def get_today_calories(current_user_id):
    """특정 사용자의 오늘 섭취 칼로리 합계, 목표 칼로리, 기록 수를 반환"""
    challenge_id = request.args.get('challenge_id', type=int)
    if not challenge_id:
        return jsonify({'success': False, 'error': 'challenge_id가 필요합니다.'}), 400

    # 해당 챌린지가 현재 사용자의 것인지 확인
    challenge = Challenge.query.filter_by(id=challenge_id, user_id=current_user_id).first()
    if not challenge:
        return jsonify({'success': False, 'error': '챌린지를 찾을 수 없거나 권한이 없습니다.'}), 404

    today = date.today()
    records = FoodRecord.query.filter(
        FoodRecord.user_id == current_user_id,
        FoodRecord.challenge_id == challenge_id,
        db.func.date(FoodRecord.recorded_at) == today
    ).all()
    
    total_calories = sum(r.calories for r in records)
    records_count = len(records)
    goal_calories = challenge.target_calories

    return jsonify({
        'success': True, 
        'total_calories': total_calories, 
        'goal_calories': goal_calories,
        'records_count': records_count
    })

@app.route('/api/users/<int:user_id>', methods=['GET', 'PUT'])
@jwt_required
def manage_user_profile(current_user_id, user_id):
    """사용자 프로필 조회 및 업데이트"""
    if current_user_id != user_id:
        return jsonify({'success': False, 'error': '권한이 없습니다.'}), 403

    user = User.query.get_or_404(user_id)

    if request.method == 'GET':
        return jsonify({
            'success': True,
            'user': {
                'id': user.id,
                'username': user.username,
                'email': user.email,
                'is_public': user.is_public,
                'created_at': user.created_at.isoformat()
            }
        })

    elif request.method == 'PUT':
        data = request.get_json()
        
        # 사용자명 업데이트
        if 'username' in data and data['username'].strip() != user.username:
            new_username = data['username'].strip()
            if User.query.filter_by(username=new_username).first():
                return jsonify({'success': False, 'error': '이미 존재하는 사용자명입니다.'}), 400
            user.username = new_username

        # 이메일 업데이트
        if 'email' in data and data['email'].strip().lower() != user.email:
            new_email = data['email'].strip().lower()
            if User.query.filter_by(email=new_email).first():
                return jsonify({'success': False, 'error': '이미 존재하는 이메일입니다.'}), 400
            user.email = new_email

        # 비밀번호 업데이트
        if 'new_password' in data:
            old_password = data.get('old_password')
            new_password = data['new_password']
            if not bcrypt.check_password_hash(user.password_hash, old_password):
                return jsonify({'success': False, 'error': '기존 비밀번호가 일치하지 않습니다.'}), 400
            user.password_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')

        # 공개 상태 업데이트
        if 'is_public' in data:
            user.is_public = bool(data['is_public'])

        try:
            db.session.commit()
            return jsonify({'success': True, 'message': '프로필이 성공적으로 업데이트되었습니다.'})
        except Exception as e:
            db.session.rollback()
            return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/stamps/count', methods=['GET'])
@jwt_required
def get_stamp_count(current_user_id):
    user_id = request.args.get('user_id', type=int)
    if not user_id or user_id != current_user_id:
        return jsonify({'success': False, 'error': '권한이 없거나 잘못된 사용자 ID입니다.'}), 403
    count = Stamp.query.filter_by(user_id=user_id).count()
    return jsonify({'success': True, 'count': count})

@app.route('/api/users/<int:user_id>/public_status', methods=['PUT'])
@jwt_required
def update_user_public_status(current_user_id, user_id):
    """사용자의 공개 상태(is_public)를 업데이트"""
    if current_user_id != user_id:
        return jsonify({'success': False, 'error': '권한이 없습니다.'}), 403

    user = User.query.get_or_404(user_id)
    data = request.get_json()
    new_status = data.get('is_public')

    if new_status is None or not isinstance(new_status, bool):
        return jsonify({'success': False, 'error': 'is_public 값은 boolean이어야 합니다.'}), 400

    try:
        user.is_public = new_status
        db.session.commit()
        return jsonify({'success': True, 'message': '공개 상태가 업데이트되었습니다.', 'is_public': user.is_public})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/challenges/process_expired', methods=['POST'])
@jwt_required
def process_expired_challenges(current_user_id):
    """만료된 챌린지를 처리하고 스탬프를 지급"""
    try:
        # 현재 사용자의 활성 챌린지 중 만료된 챌린지 찾기
        expired_challenges = Challenge.query.filter(
            Challenge.user_id == current_user_id,
            Challenge.is_active == True,
            Challenge.end_date < datetime.utcnow()
        ).all()

        for challenge in expired_challenges:
            challenge.is_active = False
            
            # 챌린지 성공/실패 판정 로직 (주 5일 이상 목표 달성)
            daily_success_count = 0
            for i in range(7):
                check_date = challenge.start_date + timedelta(days=i)
                # 해당 날짜의 총 칼로리 계산
                daily_calories = sum(r.calories for r in FoodRecord.query.filter(
                    FoodRecord.challenge_id == challenge.id,
                    db.func.date(FoodRecord.recorded_at) == check_date.date()
                ).all())
                
                if daily_calories <= challenge.target_calories:
                    daily_success_count += 1
            
            if daily_success_count >= 5:
                challenge.status = 'success'
                # 주간 성공 스탬프 지급
                new_stamp = Stamp(
                    user_id=current_user_id,
                    challenge_id=challenge.id,
                    stamp_type='weekly_success',
                    description='주간 챌린지 성공!',
                    earned_at=datetime.utcnow()
                )
                db.session.add(new_stamp)
                # 퍼펙트 위크 스탬프 (7일 모두 성공)
                if daily_success_count == 7:
                    perfect_stamp = Stamp(
                        user_id=current_user_id,
                        challenge_id=challenge.id,
                        stamp_type='perfect_week',
                        description='퍼펙트 위크 달성!',
                        earned_at=datetime.utcnow()
                    )
                    db.session.add(perfect_stamp)
            else:
                challenge.status = 'fail'
            db.session.commit()
        
        return jsonify({'success': True, 'message': f'{len(expired_challenges)}개의 챌린지가 처리되었습니다.'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': f'만료된 챌린지 처리 중 오류 발생: {str(e)}'}), 500

# 에러 핸들러
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

# 회원가입
@app.route('/api/register', methods=['POST'])
def register():
    data = request.get_json()
    username = data.get('username', '').strip()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    is_public = data.get('is_public', False) # is_public 필드 추가

    if not username or not email or not password:
        return jsonify({'success': False, 'error': '모든 항목을 입력해 주세요.'}), 400
    if User.query.filter((User.username == username) | (User.email == email)).first():
        return jsonify({'success': False, 'error': '이미 존재하는 사용자명 또는 이메일입니다.'}), 400
    pw_hash = bcrypt.generate_password_hash(password).decode('utf-8')
    new_user = User(username=username, email=email, password_hash=pw_hash, is_public=is_public)
    db.session.add(new_user)
    db.session.commit()
    return jsonify({'success': True, 'message': '회원가입 성공'})

# 로그인
@app.route('/api/login', methods=['POST'])
def login():
    data = request.get_json()
    email = data.get('email', '').strip().lower()
    password = data.get('password', '')
    user = User.query.filter_by(email=email).first()
    if not user or not bcrypt.check_password_hash(user.password_hash, password):
        return jsonify({'success': False, 'error': '이메일 또는 비밀번호가 올바르지 않습니다.'}), 401
    payload = {
        'user_id': user.id,
        'exp': datetime.utcnow() + timedelta(hours=JWT_EXPIRE_HOURS)
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    return jsonify({'success': True, 'token': token, 'user_id': user.id, 'username': user.username, 'is_public': user.is_public})

@app.route('/register.html')
def serve_register():
    return send_from_directory('.', 'register.html')

@app.route('/login.html')
def serve_login():
    return send_from_directory('.', 'login.html')

@app.route('/social-feed.html')
def serve_social_feed():
    return send_from_directory('.', 'social-feed.html')

@app.route('/profile.html')
def serve_profile():
    return send_from_directory('.', 'profile.html')

@app.route('/api/food-records/public', methods=['GET'])
@jwt_required
def get_public_food_records(current_user_id):
    """공개 설정된 사용자들의 음식 기록을 반환"""
    # is_public이 True인 사용자들의 FoodRecord만 가져옴
    public_records = db.session.query(FoodRecord, User).join(User).filter(User.is_public == True).order_by(FoodRecord.recorded_at.desc()).limit(50).all()
    
    records_data = []
    for record, user in public_records:
        records_data.append({
            'id': record.id,
            'user_id': record.user_id,
            'username': user.username, # 사용자 이름 추가
            'food_name': record.food_name,
            'calories': record.calories,
            'weight': record.weight,
            'recorded_at': record.recorded_at.isoformat(),
            'image_path': record.image_path,
            'ai_analysis': json.loads(record.ai_analysis_result) if record.ai_analysis_result else None,
            'advice': record.advice
        })
    return jsonify({'success': True, 'records': records_data})

if __name__ == '__main__':
    print("Starting Diet Challenge API Server...")
    print(f"GPU Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name()}")
    app.run(debug=True, host='0.0.0.0', port=5000) 