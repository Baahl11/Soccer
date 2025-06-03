#!/usr/bin/env python3
"""
Script simplificado para crear usuario premium sin problemas de relaciones.
"""

import sys
import os
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

# Configuración de base de datos directa
DATABASE_URL = "postgresql://postgres:password@localhost:5432/soccer_predictions"

def create_premium_access():
    """Crear usuario premium directamente con SQL"""
    
    print("🚀 Creando acceso premium directo...")
    
    try:
        # Conectar a la base de datos
        engine = create_engine(DATABASE_URL)
        Session = sessionmaker(bind=engine)
        session = Session()
        
        # Password hasheado para 'admin123'
        hashed_password = "$2b$12$LQv3c1yqBwLFaT6aVyUDeaVmjjzYH4mfkKCE.hFG8g6XyZK4zN7CG"
        
        print("📊 Creando tablas si no existen...")
        
        # Crear usuario admin si no existe
        result = session.execute(text("""
            INSERT INTO users (email, username, full_name, hashed_password, is_active, is_verified, created_at, updated_at)
            VALUES ('admin@soccerpredictions.com', 'admin', 'Administrator Premium', :password, true, true, :now, :now)
            ON CONFLICT (email) DO UPDATE SET 
                hashed_password = :password,
                is_active = true,
                is_verified = true,
                updated_at = :now
            RETURNING id
        """), {
            'password': hashed_password,
            'now': datetime.utcnow()
        })
        
        user_result = result.fetchone()
        if user_result:
            user_id = user_result[0]
            print(f"✅ Usuario creado/actualizado con ID: {user_id}")
        else:
            # Obtener ID del usuario existente
            result = session.execute(text("SELECT id FROM users WHERE email = 'admin@soccerpredictions.com'"))
            user_id = result.fetchone()[0]
            print(f"✅ Usuario existente encontrado con ID: {user_id}")
        
        # Crear tier premium si no existe
        session.execute(text("""
            INSERT INTO subscription_tiers (name, display_name, price, duration_days, max_predictions_per_day, features, created_at, updated_at)
            VALUES ('premium', 'Premium', 99.99, 30, 50, '["daily_predictions", "advanced_filters", "value_bets", "live_alerts", "phone_support"]', :now, :now)
            ON CONFLICT (name) DO UPDATE SET 
                display_name = 'Premium',
                price = 99.99,
                updated_at = :now
        """), {'now': datetime.utcnow()})
        
        # Obtener ID del tier premium
        result = session.execute(text("SELECT id FROM subscription_tiers WHERE name = 'premium'"))
        tier_id = result.fetchone()[0]
        print(f"✅ Tier premium disponible con ID: {tier_id}")
        
        # Crear/actualizar suscripción premium
        end_date = datetime.utcnow() + timedelta(days=365)
        session.execute(text("""
            INSERT INTO subscriptions (user_id, tier_id, start_date, end_date, is_active, payment_status, created_at, updated_at)
            VALUES (:user_id, :tier_id, :start_date, :end_date, true, 'completed', :now, :now)
            ON CONFLICT (user_id) DO UPDATE SET 
                tier_id = :tier_id,
                end_date = :end_date,
                is_active = true,
                payment_status = 'completed',
                updated_at = :now
        """), {
            'user_id': user_id,
            'tier_id': tier_id,
            'start_date': datetime.utcnow(),
            'end_date': end_date,
            'now': datetime.utcnow()
        })
        
        session.commit()
        
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
        session.rollback()
        return False
    finally:
        session.close()

def create_test_token():
    """Crear un script para obtener token de autenticación fácilmente"""
    
    token_script = '''#!/usr/bin/env python3
"""
Script para obtener token JWT de autenticación
"""
import requests
import json

def get_auth_token():
    """Obtener token de autenticación"""
    
    login_data = {
        "username": "admin@soccerpredictions.com",
        "password": "admin123"
    }
    
    try:
        response = requests.post(
            "http://127.0.0.1:8000/api/v1/auth/login",
            data=login_data,
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )
        
        if response.status_code == 200:
            token_data = response.json()
            token = token_data.get("access_token")
            print("✅ TOKEN OBTENIDO:")
            print(f"Bearer {token}")
            print("\\n📋 Copia esta línea completa y úsala en el botón 'Authorize' de /docs")
            return token
        else:
            print(f"❌ Error en login: {response.status_code}")
            print(response.text)
            return None
            
    except Exception as e:
        print(f"❌ Error conectando: {e}")
        return None

if __name__ == "__main__":
    print("🔑 OBTENIENDO TOKEN DE AUTENTICACIÓN...")
    print("=" * 50)
    get_auth_token()
'''
    
    with open('get_auth_token.py', 'w', encoding='utf-8') as f:
        f.write(token_script)
    
    print("✅ Script de autenticación creado: get_auth_token.py")

if __name__ == "__main__":
    success = create_premium_access()
    if success:
        create_test_token()
        print("\n🚀 PASOS PARA USAR TU ACCESO PREMIUM:")
        print("1️⃣  Ve a: http://127.0.0.1:8000/docs")
        print("2️⃣  Busca POST /api/v1/auth/login")
        print("3️⃣  Haz click en 'Try it out'")
        print("4️⃣  Ingresa:")
        print("    📧 username: admin@soccerpredictions.com")
        print("    🔑 password: admin123")
        print("5️⃣  Haz click en 'Execute'")
        print("6️⃣  Copia el 'access_token' de la respuesta")
        print("7️⃣  Haz click en 'Authorize' (arriba)")
        print("8️⃣  Ingresa: Bearer <tu_token>")
        print("9️⃣  ¡Ya tienes acceso premium completo!")
        print("\\n🔄 O ejecuta: python get_auth_token.py")
    else:
        print("❌ Error en la configuración.")
