#!/usr/bin/env python3
"""
데이터베이스 마이그레이션 스크립트
FoodRecord 테이블에 comment 필드 추가
"""

import sqlite3
import os

def migrate_database():
    """데이터베이스 마이그레이션 실행"""
    db_path = 'diet_challenge.db'
    
    if not os.path.exists(db_path):
        print("데이터베이스 파일이 존재하지 않습니다.")
        return
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # comment 컬럼이 존재하는지 확인
        cursor.execute("PRAGMA table_info(food_record)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if 'comment' not in columns:
            print("comment 컬럼을 추가합니다...")
            cursor.execute("ALTER TABLE food_record ADD COLUMN comment TEXT")
            print("✅ comment 컬럼이 성공적으로 추가되었습니다.")
        else:
            print("✅ comment 컬럼이 이미 존재합니다.")
        
        conn.commit()
        conn.close()
        print("마이그레이션이 완료되었습니다.")
        
    except Exception as e:
        print(f"마이그레이션 중 오류 발생: {e}")
        if conn:
            conn.close()

if __name__ == "__main__":
    migrate_database() 