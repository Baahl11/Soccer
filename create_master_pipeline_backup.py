#!/usr/bin/env python3
"""
Master Pipeline Project Backup Script
Crea un backup completo del proyecto Soccer Prediction excluyendo archivos innecesarios.
"""

import sys
import os
from pathlib import Path
from datetime import datetime

# Add current directory to path for imports
sys.path.append(str(Path(__file__).parent))

from backup_manager import BackupManager

def create_master_pipeline_backup():
    """Crear backup completo del Master Pipeline Soccer Prediction System."""
    
    print("🚀 MASTER PIPELINE PROJECT BACKUP")
    print("=" * 50)
    
    # Configurar rutas
    project_root = Path(__file__).parent.resolve()
    backup_name = f"master_pipeline_commercial_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    print(f"📁 Project Root: {project_root}")
    print(f"📦 Backup Name: {backup_name}")
      # Patrones de exclusión para proyecto comercial
    exclude_patterns = [
        # Virtual environments
        ".venv/**",
        ".env/**", 
        "venv/**",
        "env/**",
        
        # Python cache
        "**/__pycache__/**",
        "**/.pytest_cache/**",
        "**/*.pyc",
        "**/*.pyo",
        "**/*.pyd",
        "**/*.egg-info/**",
        
        # Git y control de versiones
        "**/.git/**",
        "**/.gitignore",
        
        # Cache y archivos temporales
        "cache/**",
        "**/*_cache/**",
        "logs/**",
        "*.log",
        "**/*.tmp",
        "**/*.temp",
        
        # Modelos y archivos grandes que pueden regenerarse
        "models/**",
        "**/*.pkl",
        "**/*.joblib",
        "**/*.model",
        
        # Datos de entrenamiento y results
        "data/**",
        "results/**",
        
        # Tests (opcional para backup comercial)
        "tests/**",
        
        # Bases de datos (excluir DBs grandes)
        "**/*.db",
        "**/*.sqlite3",
        "**/*.sqlite",
        
        # Archivos de sistema
        "**/.DS_Store",
        "**/Thumbs.db",
        "**/*.swp",
        "**/*.swo",
        
        # Directorios de build y dist
        "build/**",
        "dist/**",
        "node_modules/**",
        
        # Archivos de configuración específicos
        "**/.vscode/**",
        "**/.idea/**",
        
        # Documentación de implementación (opcional)
        "implementation/**",
        "documentation/**",
        "src/**",
        "templates/**",
        
        # Backups anteriores
        "backups/**",
        
        # Archivos muy grandes que pueden regenerarse
        "**/*.zip",
        "**/*.tar.gz",
        "**/*.rar",
    ]
    
    # Archivos críticos del Master Pipeline a incluir específicamente
    critical_files = [
        "master_prediction_pipeline_simple.py",  # Sistema principal
        "app.py",                                # Flask server
        "team_form.py",                          # Datos reales de equipos
        "data.py",                               # Integración API
        "real_time_injury_analyzer.py",         # Componente lesiones
        "market_value_analyzer.py",             # Componente mercado
        "auto_model_calibrator.py",             # Auto-calibración
        "referee_analyzer.py",                  # Análisis árbitro
        "README.md",                            # Documentación principal
        "API_DOCUMENTATION.md",                 # Docs API
        "TECHNICAL_ARCHITECTURE.md",           # Arquitectura técnica
        "MASTER_PIPELINE_COMMERCIAL_DOCS.md",  # Docs comerciales
        "COMMERCIAL_IMPLEMENTATION_COMPLETION_REPORT.md"  # Reporte final
    ]
    
    try:
        # Inicializar BackupManager
        backup_mgr = BackupManager(
            project_root=project_root,
            exclude_patterns=exclude_patterns
        )
        
        print(f"\n📊 Configuración de Backup:")
        print(f"   - Patrones de exclusión: {len(exclude_patterns)}")
        print(f"   - Archivos críticos: {len(critical_files)}")
        
        # Verificar archivos críticos
        missing_critical = []
        for critical_file in critical_files:
            file_path = project_root / critical_file
            if not file_path.exists():
                missing_critical.append(critical_file)
        
        if missing_critical:
            print(f"\n⚠️  Archivos críticos no encontrados:")
            for missing in missing_critical:
                print(f"   - {missing}")
        
        # Crear backup
        print(f"\n🔄 Creando backup...")
        manifest = backup_mgr.create_backup()
        
        # Mostrar resultados
        backup_path = Path(manifest['backup_path'])
        backup_size = backup_path.stat().st_size / (1024 * 1024)  # MB
        
        print(f"\n✅ BACKUP COMPLETADO EXITOSAMENTE")
        print(f"📦 Archivo: {backup_path.name}")
        print(f"📁 Ubicación: {backup_path.parent}")
        print(f"📊 Archivos incluidos: {len(manifest['files'])}")
        print(f"💾 Tamaño: {backup_size:.2f} MB")
        
        # Verificar backup
        print(f"\n🔍 Verificando integridad del backup...")
        if backup_mgr.verify_backup(backup_path):
            print("✅ Backup verificado correctamente")
        else:
            print("❌ Error en la verificación del backup")
            return False
        
        # Mostrar archivos incluidos más importantes
        print(f"\n📋 Archivos críticos incluidos:")
        for file_info in manifest['files']:
            file_path = file_info['path']
            if any(critical in file_path for critical in critical_files):
                size_kb = file_info['size'] / 1024
                print(f"   ✅ {file_path} ({size_kb:.1f} KB)")
        
        # Mostrar archivos Python principales
        print(f"\n🐍 Archivos Python principales:")
        python_files = [f for f in manifest['files'] 
                       if f['path'].endswith('.py') and '/' not in f['path']]
        for file_info in sorted(python_files, key=lambda x: x['size'], reverse=True)[:10]:
            size_kb = file_info['size'] / 1024
            print(f"   📄 {file_info['path']} ({size_kb:.1f} KB)")
        
        # Mostrar documentación incluida
        print(f"\n📚 Documentación incluida:")
        doc_files = [f for f in manifest['files'] 
                    if f['path'].endswith('.md')]
        for file_info in sorted(doc_files, key=lambda x: x['path']):
            size_kb = file_info['size'] / 1024
            print(f"   📖 {file_info['path']} ({size_kb:.1f} KB)")
        
        print(f"\n🎯 RESUMEN DEL BACKUP:")
        print(f"   Estado: EXITOSO ✅")
        print(f"   Versión: Master Pipeline v2.1 Enhanced")
        print(f"   Fecha: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"   Archivos: {len(manifest['files'])} incluidos")
        print(f"   Tamaño: {backup_size:.2f} MB")
        print(f"   Ubicación: {backup_path}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ ERROR AL CREAR BACKUP:")
        print(f"   {str(e)}")
        return False

def list_existing_backups():
    """Listar backups existentes."""
    
    try:
        project_root = Path(__file__).parent.resolve()
        backup_mgr = BackupManager(project_root=project_root)
        
        backups = backup_mgr.list_backups()
        
        if not backups:
            print("📦 No se encontraron backups existentes")
            return
        
        print(f"\n📦 BACKUPS EXISTENTES ({len(backups)}):")
        print("-" * 50)
        
        for i, backup in enumerate(backups, 1):
            backup_path = Path(backup['backup_path'])
            if backup_path.exists():
                size_mb = backup_path.stat().st_size / (1024 * 1024)
                timestamp = backup['timestamp']
                file_count = len(backup['files'])
                
                print(f"{i}. {backup_path.name}")
                print(f"   📅 Fecha: {timestamp}")
                print(f"   📊 Archivos: {file_count}")
                print(f"   💾 Tamaño: {size_mb:.2f} MB")
                print("")
        
    except Exception as e:
        print(f"❌ Error listando backups: {e}")

if __name__ == "__main__":
    print("🏆 MASTER PIPELINE BACKUP SYSTEM")
    print("Sistema de Backup para Soccer Prediction Commercial")
    print("=" * 60)
    
    # Mostrar backups existentes
    list_existing_backups()
    
    # Preguntar si crear nuevo backup
    response = input("\n¿Crear nuevo backup del Master Pipeline? (s/N): ").strip().lower()
    
    if response in ['s', 'si', 'yes', 'y']:
        success = create_master_pipeline_backup()
        
        if success:
            print(f"\n🎉 BACKUP CREADO EXITOSAMENTE")
            print(f"   El proyecto Master Pipeline ha sido respaldado completamente")
            print(f"   Todos los archivos comerciales están incluidos")
            print(f"   Archivos .venv y temporales fueron excluidos")
        else:
            print(f"\n💥 ERROR EN EL BACKUP")
            print(f"   Revisa los logs para más detalles")
            sys.exit(1)
    else:
        print("📦 Operación cancelada")
        
    input("\nPresiona ENTER para salir...")
