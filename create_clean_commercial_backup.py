#!/usr/bin/env python3
"""
Simple Commercial Backup - Master Pipeline Soccer Prediction
Crea un backup limpio solo de archivos comerciales críticos.
"""

import os
import zipfile
import shutil
from pathlib import Path
from datetime import datetime

def create_clean_commercial_backup():
    """Crear backup limpio solo con archivos comerciales críticos."""
    
    print("🎯 CLEAN COMMERCIAL BACKUP - MASTER PIPELINE")
    print("=" * 60)
    
    # Configurar rutas
    project_root = Path(__file__).parent.resolve()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_name = f"master_pipeline_commercial_clean_{timestamp}"
    backup_dir = project_root / "backups"
    backup_dir.mkdir(exist_ok=True)
    backup_path = backup_dir / f"{backup_name}.zip"
    
    print(f"📁 Project Root: {project_root}")
    print(f"📦 Backup: {backup_path.name}")
    
    # Archivos comerciales críticos a incluir
    commercial_files = {
        # Core System
        "master_prediction_pipeline_simple.py": "Sistema principal comercial",
        "app.py": "Servidor Flask API",
        
        # Data Integration
        "team_form.py": "Integración datos reales de equipos",
        "data.py": "Integración API datos",
        
        # Advanced Components
        "real_time_injury_analyzer.py": "Análisis lesiones en tiempo real",
        "market_value_analyzer.py": "Análisis valor de mercado",
        "auto_model_calibrator.py": "Auto-calibración modelo",
        "referee_analyzer.py": "Análisis estadístico árbitros",
        
        # Documentation
        "README.md": "Documentación principal",
        "API_DOCUMENTATION.md": "Documentación API",
        "TECHNICAL_ARCHITECTURE.md": "Arquitectura técnica",
        "TECHNICAL_SUMMARY.md": "Resumen técnico",
        "MASTER_PIPELINE_COMMERCIAL_DOCS.md": "Documentación comercial",
        "COMMERCIAL_IMPLEMENTATION_COMPLETION_REPORT.md": "Reporte de implementación",
        
        # Supporting Files
        "requirements.txt": "Dependencias Python",
        "config.py": "Configuración del sistema",
        
        # Backup System
        "backup_manager.py": "Sistema de respaldo",
        "create_master_pipeline_backup.py": "Script backup original",
    }
    
    # Optional supporting files (if they exist)
    optional_files = [
        "fixture_predictor.py",
        "weather_analyzer.py", 
        "league_analyzer.py",
        "prediction_engine.py",
        "model_trainer.py",
        "utils.py",
        "constants.py",
        "CHANGELOG.md",
        "LICENSE",
        ".env.example"
    ]
    
    try:
        files_included = 0
        total_size = 0
        
        print(f"\n🔄 Creando backup comercial limpio...")
        
        with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            # Incluir archivos críticos
            print(f"\n📋 Archivos críticos:")
            for filename, description in commercial_files.items():
                file_path = project_root / filename
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    zf.write(file_path, filename)
                    files_included += 1
                    total_size += file_size
                    print(f"   ✅ {filename} ({file_size/1024:.1f} KB) - {description}")
                else:
                    print(f"   ⚠️  {filename} - NO ENCONTRADO")
            
            # Incluir archivos opcionales si existen
            print(f"\n📄 Archivos opcionales:")
            for filename in optional_files:
                file_path = project_root / filename
                if file_path.exists():
                    file_size = file_path.stat().st_size
                    zf.write(file_path, filename)
                    files_included += 1
                    total_size += file_size
                    print(f"   ✅ {filename} ({file_size/1024:.1f} KB)")
                    
        # Verificar backup
        backup_size_mb = backup_path.stat().st_size / (1024 * 1024)
        
        print(f"\n✅ BACKUP COMERCIAL COMPLETADO")
        print(f"📦 Archivo: {backup_path.name}")
        print(f"📁 Ubicación: {backup_path.parent}")
        print(f"📊 Archivos incluidos: {files_included}")
        print(f"💾 Tamaño total: {total_size/1024:.1f} KB")
        print(f"💾 Tamaño comprimido: {backup_size_mb:.2f} MB")
        
        # Verificar integridad
        print(f"\n🔍 Verificando integridad...")
        with zipfile.ZipFile(backup_path, 'r') as zf:
            test_result = zf.testzip()
            if test_result is None:
                print("✅ Backup verificado correctamente")
            else:
                print(f"❌ Error en archivo: {test_result}")
                return False
        
        # Mostrar contenido del backup
        print(f"\n📋 CONTENIDO DEL BACKUP:")
        with zipfile.ZipFile(backup_path, 'r') as zf:
            for info in sorted(zf.infolist(), key=lambda x: x.filename):
                size_kb = info.file_size / 1024
                compressed_kb = info.compress_size / 1024
                compression_ratio = (1 - info.compress_size / info.file_size) * 100 if info.file_size > 0 else 0
                print(f"   📄 {info.filename}")
                print(f"      💾 {size_kb:.1f} KB → {compressed_kb:.1f} KB ({compression_ratio:.1f}% compresión)")
        
        print(f"\n🎉 BACKUP COMERCIAL EXITOSO")
        print(f"   ✅ Sistema Master Pipeline respaldado")
        print(f"   ✅ Solo archivos comerciales críticos")
        print(f"   ✅ Sin .venv, cache, o archivos temporales")
        print(f"   ✅ Verificación de integridad exitosa")
        print(f"   📍 Ubicación: {backup_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR AL CREAR BACKUP:")
        print(f"   {str(e)}")
        if backup_path.exists():
            backup_path.unlink()
        return False

if __name__ == "__main__":
    print("🏆 MASTER PIPELINE COMMERCIAL BACKUP")
    print("Sistema de Backup Limpio para Archivos Comerciales")
    print("=" * 60)
    
    response = input("\n¿Crear backup comercial limpio? (s/N): ").strip().lower()
    
    if response in ['s', 'si', 'yes', 'y']:
        success = create_clean_commercial_backup()
        
        if success:
            print(f"\n🎯 BACKUP COMERCIAL CREADO EXITOSAMENTE")
            print(f"   Solo archivos críticos del sistema comercial")
            print(f"   Listo para distribución o respaldo")
        else:
            print(f"\n💥 ERROR EN EL BACKUP COMERCIAL")
    else:
        print("📦 Operación cancelada")
        
    input("\nPresiona ENTER para salir...")
