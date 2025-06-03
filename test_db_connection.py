#!/usr/bin/env python3
"""
Test simple de conexión a base de datos
"""

import sys
from sqlalchemy import create_engine, text

DATABASE_URL = "postgresql://postgres:password@localhost:5432/soccer_predictions"

def test_connection():
    """Test básico de conexión"""
    print("🔍 Probando conexión a PostgreSQL...")
    
    try:
        print(f"🔗 URL: {DATABASE_URL}")
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1 as test"))
            row = result.fetchone()
            print(f"✅ Conexión exitosa! Resultado: {row[0]}")
            
            # Ver qué tablas existen
            result = conn.execute(text("""
                SELECT table_name FROM information_schema.tables 
                WHERE table_schema = 'public'
            """))
            tables = [row[0] for row in result.fetchall()]
            print(f"📊 Tablas encontradas: {tables}")
            
            return True
            
    except Exception as e:
        print(f"❌ Error de conexión: {e}")
        print(f"🔍 Tipo de error: {type(e)}")
        return False

if __name__ == "__main__":
    success = test_connection()
    if success:
        print("✅ Base de datos lista para usar")
    else:
        print("❌ Problema con la base de datos")
