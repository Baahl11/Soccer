#!/usr/bin/env python3
"""
Script para crear usuario premium usando SQLite (temporal)
"""

import sys
import os
from datetime import datetime, timedelta
import sqlite3
import hashlib
import secrets

def create_premium_user_sqlite():
    """Crear usuario premium en SQLite"""
    
    print("🚀 Creando acceso premium con SQLite...")
    
    try:
        # Conectar a SQLite
        db_path = "fastapi_backend/soccer_predictions.db"
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Password hasheado para 'admin123' (bcrypt)
        hashed_password = "$2b$12$LQv3c1yqBwLFaT6aVyUDeaVmjjzYH4mfkKCE.hFG8g6XyZK4zN7CG"
        
        print("📊 Creando tablas si no existen...")
        
        # Crear tabla users
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                full_name TEXT,
                hashed_password TEXT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT TRUE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Crear tabla subscription_tiers
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscription_tiers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT UNIQUE NOT NULL,
                display_name TEXT NOT NULL,
                price REAL NOT NULL,
                duration_days INTEGER NOT NULL,
                max_predictions_per_day INTEGER NOT NULL,
                features TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Crear tabla subscriptions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                tier_id INTEGER NOT NULL,
                start_date TIMESTAMP NOT NULL,
                end_date TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                payment_status TEXT DEFAULT 'completed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (tier_id) REFERENCES subscription_tiers (id),
                UNIQUE(user_id)
            )
        """)
        
        print("✅ Tablas creadas")
        
        # Crear o actualizar usuario admin
        cursor.execute("""
            INSERT OR REPLACE INTO users 
            (email, username, full_name, hashed_password, is_active, is_verified, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'admin@soccerpredictions.com',
            'admin',
            'Administrator Premium',
            hashed_password,
            True,
            True,
            datetime.utcnow().isoformat(),
            datetime.utcnow().isoformat()
        ))
        
        # Obtener ID del usuario
        cursor.execute("SELECT id FROM users WHERE email = ?", ('admin@soccerpredictions.com',))
        user_result = cursor.fetchone()
        user_id = user_result[0]
        print(f"✅ Usuario creado/actualizado con ID: {user_id}")
        
        # Crear tier premium
        features_json = '["daily_predictions", "advanced_filters", "value_bets", "live_alerts", "phone_support"]'
        cursor.execute("""
            INSERT OR REPLACE INTO subscription_tiers 
            (name, display_name, price, duration_days, max_predictions_per_day, features, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'premium',
            'Premium',
            99.99,
            30,
            50,
            features_json,
            datetime.utcnow().isoformat(),
            datetime.utcnow().isoformat()
        ))
        
        # Obtener ID del tier
        cursor.execute("SELECT id FROM subscription_tiers WHERE name = ?", ('premium',))
        tier_result = cursor.fetchone()
        tier_id = tier_result[0]
        print(f"✅ Tier premium disponible con ID: {tier_id}")
        
        # Crear suscripción premium
        end_date = datetime.utcnow() + timedelta(days=365)
        cursor.execute("""
            INSERT OR REPLACE INTO subscriptions 
            (user_id, tier_id, start_date, end_date, is_active, payment_status, created_at, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            tier_id,
            datetime.utcnow().isoformat(),
            end_date.isoformat(),
            True,
            'completed',
            datetime.utcnow().isoformat(),
            datetime.utcnow().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        print("\n✅ ¡ACCESO PREMIUM CONFIGURADO!")
        print("=" * 50)
        print("📧 Email: admin@soccerpredictions.com")
        print("🔑 Password: admin123")
        print("👑 Tier: Premium")
        print("⏰ Válido hasta:", end_date.strftime("%Y-%m-%d"))
        print("🔓 Acceso a TODAS las funcionalidades premium")
        print("=" * 50)
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

def create_auth_helper():
    """Crear script helper para autenticación"""
    
    auth_script = '''#!/usr/bin/env python3
"""
Helper para obtener token de autenticación fácilmente
"""
import requests
import json

def get_token():
    """Obtener token JWT"""
    print("🔑 Obteniendo token de autenticación...")
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/v1/auth/login",
            data={
                "username": "admin@soccerpredictions.com",
                "password": "admin123"
            },
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            data = response.json()
            token = data.get("access_token")
            print("\\n✅ TOKEN OBTENIDO:")
            print(f"Bearer {token}")
            print("\\n📋 Para usar en /docs:")
            print("1. Ve a http://127.0.0.1:8000/docs")
            print("2. Click en 'Authorize'")
            print("3. Ingresa la línea completa de arriba")
            return token
        else:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se puede conectar al servidor")
        print("🔧 Asegúrate de que FastAPI esté corriendo en http://127.0.0.1:8000")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    get_token()
'''
    
    with open('get_token.py', 'w', encoding='utf-8') as f:
        f.write(auth_script)
    
    print("✅ Helper de autenticación creado: get_token.py")

if __name__ == "__main__":
    print("🎯 CONFIGURANDO ACCESO PREMIUM...")
    print("=" * 50)
    
    success = create_premium_user_sqlite()
    
    if success:
        create_auth_helper()
        print("\n🚀 INSTRUCCIONES DE USO:")
        print("1️⃣  Ejecuta: python get_token.py")
        print("2️⃣  Ve a: http://127.0.0.1:8000/docs")
        print("3️⃣  Haz click en 'Authorize' (botón verde)")
        print("4️⃣  Pega el token que obtuviste en paso 1")
        print("5️⃣  ¡Ya puedes usar todos los endpoints premium!")
        print("\n🔄 Usuario creado para futuras pruebas:")
        print("📧 Email: admin@soccerpredictions.com")
        print("🔑 Password: admin123")
    else:
        print("❌ Error en la configuración")
