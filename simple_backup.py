import zipfile
import os
from pathlib import Path
from datetime import datetime

print("Creating Master Pipeline Commercial Backup...")

# Setup paths
project_root = Path("c:/Users/gm_me/Soccer2/Soccer")
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
backup_dir = project_root / "backups"
backup_dir.mkdir(exist_ok=True)
backup_path = backup_dir / f"master_pipeline_commercial_{timestamp}.zip"

# Critical commercial files
files_to_backup = [
    "master_prediction_pipeline_simple.py",
    "app.py",
    "team_form.py", 
    "data.py",
    "real_time_injury_analyzer.py",
    "market_value_analyzer.py",
    "auto_model_calibrator.py",
    "referee_analyzer.py",
    "README.md",
    "API_DOCUMENTATION.md",
    "TECHNICAL_ARCHITECTURE.md",
    "TECHNICAL_SUMMARY.md", 
    "MASTER_PIPELINE_COMMERCIAL_DOCS.md",
    "COMMERCIAL_IMPLEMENTATION_COMPLETION_REPORT.md",
    "config.py",
    "backup_manager.py"
]

print(f"Backup path: {backup_path}")
print(f"Files to include: {len(files_to_backup)}")

with zipfile.ZipFile(backup_path, 'w', zipfile.ZIP_DEFLATED) as zf:
    included = 0
    for filename in files_to_backup:
        file_path = project_root / filename
        if file_path.exists():
            zf.write(file_path, filename)
            size_kb = file_path.stat().st_size / 1024
            print(f"✅ {filename} ({size_kb:.1f} KB)")
            included += 1
        else:
            print(f"⚠️ {filename} - NOT FOUND")

backup_size = backup_path.stat().st_size / (1024 * 1024)
print(f"\n✅ BACKUP COMPLETED")
print(f"📦 File: {backup_path.name}")
print(f"📊 Files included: {included}")
print(f"💾 Size: {backup_size:.2f} MB")

# Verify backup
with zipfile.ZipFile(backup_path, 'r') as zf:
    test_result = zf.testzip()
    if test_result is None:
        print("✅ Backup verified successfully")
    else:
        print(f"❌ Error in file: {test_result}")
