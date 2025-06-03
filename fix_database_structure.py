#!/usr/bin/env python3
"""
Script para verificar y arreglar la estructura de la base de datos SQLite
"""

import sqlite3
import os
from datetime import datetime, timedelta

def check_database_structure():
    """Verificar estructura actual de la base de datos"""
    
    db_path = "fastapi_backend/soccer_predictions.db"
    
    print("🔍 Verificando estructura de la base de datos...")
    
    if not os.path.exists(db_path):
        print(f"❌ Base de datos no encontrada en: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Obtener lista de tablas
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        print(f"📊 Tablas encontradas: {tables}")
        
        # Verificar estructura de cada tabla relevante
        for table in ['users', 'subscriptions', 'subscription_tiers']:
            if table in tables:
                cursor.execute(f"PRAGMA table_info({table})")
                columns = cursor.fetchall()
                print(f"\n📋 Estructura de {table}:")
                for col in columns:
                    print(f"  - {col[1]} ({col[2]})")
            else:
                print(f"\n❌ Tabla {table} no existe")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def recreate_database():
    """Recrear la base de datos con estructura correcta"""
    
    db_path = "fastapi_backend/soccer_predictions.db"
    
    print("🔧 Recreando base de datos con estructura correcta...")
    
    try:
        # Eliminar base de datos existente si existe
        if os.path.exists(db_path):
            os.remove(db_path)
            print("🗑️ Base de datos anterior eliminada")
        
        # Crear nueva base de datos
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        print("📊 Creando tablas...")
        
        # Crear tabla subscription_tiers primero (referenciada por otras)
        cursor.execute("""
            CREATE TABLE subscription_tiers (
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
        print("✅ Tabla subscription_tiers creada")
        
        # Crear tabla users
        cursor.execute("""
            CREATE TABLE users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT UNIQUE NOT NULL,
                username TEXT UNIQUE NOT NULL,
                full_name TEXT,
                hashed_password TEXT NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                is_verified BOOLEAN DEFAULT TRUE,
                subscription_tier_id INTEGER DEFAULT 1,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (subscription_tier_id) REFERENCES subscription_tiers (id)
            )
        """)
        print("✅ Tabla users creada")
        
        # Crear tabla subscriptions
        cursor.execute("""
            CREATE TABLE subscriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                subscription_tier_id INTEGER NOT NULL,
                start_date TIMESTAMP NOT NULL,
                end_date TIMESTAMP NOT NULL,
                is_active BOOLEAN DEFAULT TRUE,
                payment_status TEXT DEFAULT 'completed',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id),
                FOREIGN KEY (subscription_tier_id) REFERENCES subscription_tiers (id),
                UNIQUE(user_id)
            )
        """)
        print("✅ Tabla subscriptions creada")
        
        # Insertar tier básico
        cursor.execute("""
            INSERT INTO subscription_tiers 
            (name, display_name, price, duration_days, max_predictions_per_day, features)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'basic',
            'Basic',
            0.0,
            365,
            5,
            '["basic_predictions"]'
        ))
        
        # Insertar tier premium
        cursor.execute("""
            INSERT INTO subscription_tiers 
            (name, display_name, price, duration_days, max_predictions_per_day, features)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'premium',
            'Premium',
            99.99,
            30,
            50,
            '["daily_predictions", "advanced_filters", "value_bets", "live_alerts", "phone_support"]'
        ))
        
        print("✅ Tiers de suscripción insertados")
        
        # Crear usuario premium admin
        hashed_password = "$2b$12$LQv3c1yqBwLFaT6aVyUDeaVmjjzYH4mfkKCE.hFG8g6XyZK4zN7CG"
        
        cursor.execute("""
            INSERT INTO users 
            (email, username, full_name, hashed_password, is_active, is_verified, subscription_tier_id)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            'admin@soccerpredictions.com',
            'admin',
            'Administrator Premium',
            hashed_password,
            True,
            True,
            2  # Premium tier
        ))
        
        user_id = cursor.lastrowid
        print(f"✅ Usuario admin creado con ID: {user_id}")
        
        # Crear suscripción premium activa
        end_date = datetime.now() + timedelta(days=365)
        cursor.execute("""
            INSERT INTO subscriptions 
            (user_id, subscription_tier_id, start_date, end_date, is_active, payment_status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            2,  # Premium tier
            datetime.now().isoformat(),
            end_date.isoformat(),
            True,
            'completed'
        ))
        
        print("✅ Suscripción premium creada")
        
        conn.commit()
        conn.close()
        
        print("\n✅ ¡BASE DE DATOS RECREADA EXITOSAMENTE!")
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
Helper para obtener token de autenticación
"""
import requests
import json

def get_auth_token():
    """Obtener token JWT para acceso premium"""
    print("🔑 Obteniendo token de autenticación...")
    print("🔗 Conectando a: http://127.0.0.1:8000")
    
    try:
        # Hacer login
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
            print("\\n✅ TOKEN OBTENIDO EXITOSAMENTE!")
            print("=" * 60)
            print(f"Bearer {token}")
            print("=" * 60)
            print("\\n📋 INSTRUCCIONES PARA USAR:")
            print("1. Ve a: http://127.0.0.1:8000/docs")
            print("2. Haz click en el botón 'Authorize' (🔒)")
            print("3. Pega exactamente esta línea:")
            print(f"   Bearer {token}")
            print("4. Haz click en 'Authorize'")
            print("5. ¡Ahora tienes acceso premium completo!")
            
            print("\\n🎯 ENDPOINTS PREMIUM DISPONIBLES:")
            print("- POST /api/v1/predictions/value-bets")
            print("- GET /api/v1/predictions/premium")
            print("- GET /api/v1/matches/detailed")
            print("- POST /api/v1/payments/subscribe")
            
            return token
        else:
            print(f"❌ Error en login: {response.status_code}")
            print("📄 Respuesta:", response.text)
            
    except requests.exceptions.ConnectionError:
        print("❌ Error: No se puede conectar al servidor FastAPI")
        print("🔧 Solución:")
        print("1. Asegúrate de que el servidor esté corriendo")
        print("2. En otra terminal ejecuta: uvicorn app.main:app --reload")
        print("3. Verifica que esté en: http://127.0.0.1:8000")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    get_auth_token()
'''
    
    with open('get_premium_token.py', 'w', encoding='utf-8') as f:
        f.write(auth_script)
    
    print("✅ Helper de autenticación creado: get_premium_token.py")

if __name__ == "__main__":
    print("🎯 CONFIGURACIÓN COMPLETA DE ACCESO PREMIUM")
    print("=" * 60)
    
    # Verificar estructura actual
    check_database_structure()
    
    print("\n🔧 Procediendo con recreación de base de datos...")
    success = recreate_database()
    
    if success:
        create_auth_helper()
        
        print("\n🚀 ¡CONFIGURACIÓN COMPLETADA!")
        print("=" * 60)
        print("🎯 PASOS PARA USAR TU ACCESO PREMIUM:")
        print("1️⃣  Ejecuta: python get_premium_token.py")
        print("2️⃣  Copia el token que aparece")
        print("3️⃣  Ve a: http://127.0.0.1:8000/docs")
        print("4️⃣  Haz click en 'Authorize'")
        print("5️⃣  Pega: Bearer <tu_token>")
        print("6️⃣  ¡Disfruta del acceso premium completo!")
        
        print("\n🔐 CREDENCIALES PERMANENTES:")
        print("📧 Email: admin@soccerpredictions.com")
        print("🔑 Password: admin123")
        print("👑 Nivel: Premium (365 días)")
        
    else:
        print("❌ Error en la configuración de la base de datos")
