from app import app, db, User, Challenge, FoodRecord, Stamp
from datetime import datetime, timedelta
import random

def init_database():
    """데이터베이스 초기화 및 샘플 데이터 생성"""
    with app.app_context():
        # 기존 데이터 삭제
        db.drop_all()
        db.create_all()
        
        print("Database initialized successfully!")
        
        # 샘플 사용자 생성
        users = [
            User(username='김다이어터', email='kim@example.com'),
            User(username='이건강', email='lee@example.com'),
            User(username='박운동', email='park@example.com'),
            User(username='최영양', email='choi@example.com')
        ]
        
        for user in users:
            db.session.add(user)
        db.session.commit()
        
        print(f"Created {len(users)} sample users")
        
        # 샘플 챌린지 생성
        challenges = [
            Challenge(user_id=1, target_calories=1000),
            Challenge(user_id=2, target_calories=1500),
            Challenge(user_id=3, target_calories=2000),
            Challenge(user_id=4, target_calories=1200)
        ]
        
        for challenge in challenges:
            db.session.add(challenge)
        db.session.commit()
        
        print(f"Created {len(challenges)} sample challenges")
        
        # 샘플 음식 기록 생성
        food_data = [
            {'name': '닭가슴살 샐러드', 'calories': 320, 'weight': 250},
            {'name': '사과 1개', 'calories': 80, 'weight': 150},
            {'name': '오트밀', 'calories': 150, 'weight': 100},
            {'name': '김치찌개', 'calories': 350, 'weight': 300},
            {'name': '바나나', 'calories': 105, 'weight': 120},
            {'name': '계란 2개', 'calories': 140, 'weight': 100},
            {'name': '현미밥', 'calories': 200, 'weight': 150},
            {'name': '브로콜리', 'calories': 55, 'weight': 100}
        ]
        
        # 오늘과 어제 날짜
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        
        for i, challenge in enumerate(challenges):
            # 각 챌린지에 대해 3-5개의 음식 기록 생성
            num_records = random.randint(3, 5)
            selected_foods = random.sample(food_data, num_records)
            
            for j, food in enumerate(selected_foods):
                # 시간을 랜덤하게 분산
                record_time = today.replace(
                    hour=random.randint(6, 21),
                    minute=random.randint(0, 59)
                ) - timedelta(hours=j*3)
                
                record = FoodRecord(
                    user_id=challenge.user_id,
                    challenge_id=challenge.id,
                    food_name=food['name'],
                    calories=food['calories'],
                    weight=food['weight'],
                    recorded_at=record_time,
                    ai_analysis_result='{"food_name": "%s", "calories": %d, "confidence": 0.9}' % (food['name'], food['calories'])
                )
                db.session.add(record)
        
        db.session.commit()
        print("Created sample food records")
        
        # 샘플 스탬프 생성
        stamp_types = [
            ('daily_goal', '일일 목표 달성'),
            ('weekly_goal', '주간 목표 달성'),
            ('streak_3', '3일 연속 달성'),
            ('streak_7', '7일 연속 달성'),
            ('first_record', '첫 음식 기록'),
            ('perfect_day', '완벽한 하루')
        ]
        
        for user in users:
            # 각 사용자에게 2-4개의 스탬프 부여
            num_stamps = random.randint(2, 4)
            selected_stamps = random.sample(stamp_types, num_stamps)
            
            for stamp_type, description in selected_stamps:
                stamp = Stamp(
                    user_id=user.id,
                    challenge_id=random.choice(challenges).id,
                    stamp_type=stamp_type,
                    description=description,
                    earned_at=today - timedelta(days=random.randint(1, 30))
                )
                db.session.add(stamp)
        
        db.session.commit()
        print("Created sample stamps")
        
        print("\n=== Database Initialization Complete ===")
        print(f"Users: {len(users)}")
        print(f"Challenges: {len(challenges)}")
        print(f"Food Records: {len(FoodRecord.query.all())}")
        print(f"Stamps: {len(Stamp.query.all())}")

if __name__ == '__main__':
    init_database() 