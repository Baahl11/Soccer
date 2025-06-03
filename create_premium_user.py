#!/usr/bin/env python3
"""
Script para crear un usuario premium de prueba en el sistema de Soccer Predictions.
Esto te dará acceso completo a todas las funcionalidades premium.
"""

import sys
import os
import asyncio
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from passlib.context import CryptContext

# Agregar el directorio del proyecto al path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'fastapi_backend'))

from app.core.database import SessionLocal, engine
from app.models.user import User, SubscriptionTier
from app.models.subscription import Subscription
from app.core.database import Base

# Configurar el contexto de encriptación para passwords
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def hash_password(password: str) -> str:
    """Encriptar password"""
    return pwd_context.hash(password)

def create_premium_user():
    """Crear un usuario premium de prueba"""
    
    print("🚀 Creando usuario premium de prueba...")
    
    # Crear las tablas si no existen
    print("📊 Verificando tablas de la base de datos...")
    Base.metadata.create_all(bind=engine)
    
    # Crear sesión de base de datos
    db = SessionLocal()
    
    try:
        # Verificar si ya existe el usuario
        existing_user = db.query(User).filter(User.email == "admin@soccerpredictions.com").first()
        if existing_user:
            print("⚠️  Usuario admin ya existe. Actualizando a premium...")
            user = existing_user
        else:
            # Crear nuevo usuario admin
            print("👤 Creando nuevo usuario administrador...")
            user = User(
                email="admin@soccerpredictions.com",
                username="admin",
                full_name="Administrator Premium",
                hashed_password=hash_password("admin123"),
                is_active=True,
                is_verified=True,
                created_at=datetime.utcnow()
            )
            db.add(user)
            db.flush()  # Para obtener el ID
        
        # Verificar/crear tier premium
        premium_tier = db.query(SubscriptionTier).filter(SubscriptionTier.name == "premium").first()
        if not premium_tier:
            print("🎯 Creando tier premium...")
            premium_tier = SubscriptionTier(
                name="premium",
                display_name="Premium",
                price=99.99,
                duration_days=30,
                max_predictions_per_day=50,
                features=["daily_predictions", "advanced_filters", "value_bets", "live_alerts", "phone_support"],
                created_at=datetime.utcnow()
            )
            db.add(premium_tier)
            db.flush()
        
        # Verificar si ya tiene suscripción activa
        existing_subscription = db.query(Subscription).filter(
            Subscription.user_id == user.id,
            Subscription.is_active == True
        ).first()
        
        if existing_subscription:
            print("💳 Actualizando suscripción existente a premium...")
            existing_subscription.tier_id = premium_tier.id
            existing_subscription.end_date = datetime.utcnow() + timedelta(days=365)  # 1 año
            existing_subscription.is_active = True
            existing_subscription.updated_at = datetime.utcnow()
        else:
            print("💳 Creando nueva suscripción premium...")
            subscription = Subscription(
                user_id=user.id,
                tier_id=premium_tier.id,
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=365),  # 1 año de acceso
                is_active=True,
                payment_status="completed",
                created_at=datetime.utcnow()
            )
            db.add(subscription)
        
        # Confirmar cambios
        db.commit()
        
        print("\n✅ ¡Usuario premium creado/actualizado exitosamente!")
        print("=" * 50)
        print("📧 Email: admin@soccerpredictions.com")
        print("🔑 Password: admin123")
        print("👑 Tier: Premium")
        print("⏰ Válido hasta:", (datetime.utcnow() + timedelta(days=365)).strftime("%Y-%m-%d"))
        print("🔓 Acceso a TODAS las funcionalidades premium")
        print("=" * 50)
        
        # Crear también un usuario básico para comparar
        basic_user = db.query(User).filter(User.email == "user@test.com").first()
        if not basic_user:
            print("\n👤 Creando usuario básico adicional para pruebas...")
            
            # Verificar/crear tier básico
            basic_tier = db.query(SubscriptionTier).filter(SubscriptionTier.name == "basic").first()
            if not basic_tier:
                basic_tier = SubscriptionTier(
                    name="basic",
                    display_name="Basic",
                    price=19.99,
                    duration_days=30,
                    max_predictions_per_day=10,
                    features=["daily_predictions", "basic_filters", "email_support"],
                    created_at=datetime.utcnow()
                )
                db.add(basic_tier)
                db.flush()
            
            basic_user = User(
                email="user@test.com",
                username="testuser",
                full_name="Test User Basic",
                hashed_password=hash_password("test123"),
                is_active=True,
                is_verified=True,
                created_at=datetime.utcnow()
            )
            db.add(basic_user)
            db.flush()
            
            basic_subscription = Subscription(
                user_id=basic_user.id,
                tier_id=basic_tier.id,
                start_date=datetime.utcnow(),
                end_date=datetime.utcnow() + timedelta(days=30),
                is_active=True,
                payment_status="completed",
                created_at=datetime.utcnow()
            )
            db.add(basic_subscription)
            db.commit()
            
            print("📧 Usuario básico: user@test.com")
            print("🔑 Password: test123")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creando usuario premium: {e}")
        db.rollback()
        return False
    finally:
        db.close()

def verify_user_access():
    """Verificar el acceso del usuario creado"""
    print("\n🔍 Verificando acceso del usuario premium...")
    
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == "admin@soccerpredictions.com").first()
        if user:
            subscription = db.query(Subscription).filter(
                Subscription.user_id == user.id,
                Subscription.is_active == True
            ).first()
            
            if subscription:
                tier = db.query(SubscriptionTier).filter(SubscriptionTier.id == subscription.tier_id).first()
                print(f"✅ Usuario verificado: {user.email}")
                print(f"✅ Tier activo: {tier.name if tier else 'Unknown'}")
                print(f"✅ Válido hasta: {subscription.end_date}")
                print(f"✅ Funcionalidades: {tier.features if tier else []}")
                return True
            else:
                print("❌ No se encontró suscripción activa")
                return False
        else:
            print("❌ Usuario no encontrado")
            return False
    finally:
        db.close()

if __name__ == "__main__":
    print("🎯 CONFIGURADOR DE USUARIO PREMIUM")
    print("=" * 50)
    
    success = create_premium_user()
    if success:
        verify_user_access()
        print("\n🚀 ¡Listo! Ya puedes hacer login con:")
        print("   📧 Email: admin@soccerpredictions.com")
        print("   🔑 Password: admin123")
        print("\n🔗 Ve a: http://127.0.0.1:8000/docs")
        print("   1. Busca el endpoint POST /api/v1/auth/login")
        print("   2. Haz login con las credenciales de arriba")
        print("   3. Copia el token JWT que te devuelve")
        print("   4. Usa 'Bearer <tu_token>' en el botón 'Authorize'")
        print("   5. ¡Ya tienes acceso premium completo!")
    else:
        print("❌ Error en la configuración. Revisa los logs arriba.")
