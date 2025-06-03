import sqlite3
import os
from datetime import datetime, timedelta

db_path = "fastapi_backend/soccer_predictions.db"

print("🔧 Recreando base de datos para acceso premium...")

# Eliminar BD existente
if os.path.exists(db_path):
    os.remove(db_path)
    print("✅ BD anterior eliminada")

# Crear nueva BD
conn = sqlite3.connect(db_path)
cursor = conn.cursor()

# Crear subscription_tiers
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

# Crear users
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

# Crear subscriptions  
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

# Insertar tiers
cursor.execute("""
    INSERT INTO subscription_tiers 
    (name, display_name, price, duration_days, max_predictions_per_day, features)
    VALUES ('basic', 'Basic', 0.0, 365, 5, '["basic_predictions"]')
""")

cursor.execute("""
    INSERT INTO subscription_tiers 
    (name, display_name, price, duration_days, max_predictions_per_day, features)
    VALUES ('premium', 'Premium', 99.99, 30, 50, '["daily_predictions", "advanced_filters", "value_bets", "live_alerts", "phone_support"]')
""")
print("✅ Tiers insertados")

# Crear usuario admin premium
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

# Crear suscripción premium
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

print("\n🎉 ¡ACCESO PREMIUM CONFIGURADO!")
print("=" * 50)
print("📧 Email: admin@soccerpredictions.com")
print("🔑 Password: admin123")
print("👑 Tier: Premium")
print("⏰ Válido hasta:", end_date.strftime("%Y-%m-%d"))
print("=" * 50)
