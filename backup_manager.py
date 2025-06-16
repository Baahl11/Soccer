import os
import json
import shutil
import zipfile
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Set, Union, Any

# Create logs directory if it doesn't exist
log_dir = Path('logs')
log_dir.mkdir(exist_ok=True)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backup_manager.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class BackupManager:
    """Handles creation, verification and restoration of project backups."""
    
    def __init__(self, project_root: Union[str, Path], 
                 backup_dir: Optional[Union[str, Path]] = None,
                 exclude_patterns: Optional[List[str]] = None):
        """
        Initialize backup manager.
        
        Args:
            project_root: Root directory of project to backup
            backup_dir: Directory to store backups (default: project_root/backups)
            exclude_patterns: List of glob patterns to exclude from backup
        """
        self.project_root = Path(project_root).resolve()
        if not self.project_root.exists():
            raise ValueError(f"Project root does not exist: {self.project_root}")
            
        # Setup backup directory
        self.backup_dir = Path(backup_dir) if backup_dir else self.project_root / "backups"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
          # Default exclude patterns
        self.exclude_patterns = set(exclude_patterns or [])
        self.exclude_patterns.add(".venv/**")  # Always exclude .venv
        self.exclude_patterns.add("**/__pycache__/**")
        self.exclude_patterns.add("**/.git/**")
        self.exclude_patterns.add(".git/**")  # Added explicit git exclusion
        self.exclude_patterns.add("cache/**")  # Exclude cache directories
        self.exclude_patterns.add("backups/**")
        self.exclude_patterns.add("logs/**")
        
        logger.info(f"BackupManager initialized with root: {self.project_root}")
        logger.info(f"Backup directory: {self.backup_dir}")
        
    def backup_model(self, model_path: str, metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        Create a backup of a model file with metadata.
        
        Args:
            model_path: Path to the model file to backup
            metadata: Optional metadata about the model
            
        Returns:
            bool: True if backup was successful
        """
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model file not found: {model_path}")
                return False
                
            # Create timestamp for backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = os.path.basename(model_path)
            backup_path = os.path.join(
                self.backup_dir, 
                f"{model_name}_{timestamp}"
            )
            
            # Copy model file
            shutil.copy2(model_path, backup_path)
            
            # Save metadata if provided
            if metadata:
                metadata_path = f"{backup_path}_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(metadata, f, indent=2)
            
            logger.info(f"Created backup: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            return False

    def create_backup(self, additional_files: Optional[List[str]] = None) -> Dict:
        """
        Create a new backup of the project.
        
        Args:
            additional_files: Optional list of specific files to include
            
        Returns:
            Dict with backup information including timestamp, path, and manifest
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_name = f"project_backup_{timestamp}"
        backup_path = self.backup_dir / f"{backup_name}.zip"
        manifest_path = self.backup_dir / f"{backup_name}_manifest.json"
        
        try:
            # Track included files and their sizes
            manifest = {
                "timestamp": timestamp,
                "backup_path": str(backup_path),
                "files": []
            }
            
            # Create backup archive
            with zipfile.ZipFile(backup_path, "w", zipfile.ZIP_DEFLATED) as zf:
                # First add explicit additional files if any
                if additional_files:
                    for file_path in additional_files:
                        abs_path = Path(file_path).resolve()
                        if abs_path.exists() and abs_path.is_file():
                            rel_path = abs_path.relative_to(self.project_root)
                            zf.write(abs_path, rel_path)
                            manifest["files"].append({
                                "path": str(rel_path),
                                "size": abs_path.stat().st_size
                            })
                  # Then add all project files except excluded ones
                for root, dirs, files in os.walk(self.project_root):
                    # Skip excluded directories completely
                    dirs[:] = [d for d in dirs if not any(
                        Path(root) / d == self.project_root / pattern.split('/')[0] or
                        str(Path(root) / d).endswith('.git') or
                        str(Path(root) / d).endswith('cache') or
                        str(Path(root) / d).endswith('__pycache__') or
                        str(Path(root) / d).endswith('.venv')
                        for pattern in self.exclude_patterns
                    )]
                    
                    for file in files:
                        abs_path = Path(root) / file
                        rel_path = abs_path.relative_to(self.project_root)
                        
                        # Skip if matches any exclude pattern
                        if any(rel_path.match(pattern) for pattern in self.exclude_patterns):
                            continue
                            
                        # Skip if already added via additional_files
                        if additional_files and str(abs_path) in additional_files:
                            continue
                            
                        # Add file to zip
                        zf.write(abs_path, rel_path)
                        manifest["files"].append({
                            "path": str(rel_path),
                            "size": abs_path.stat().st_size
                        })
            
            # Save manifest
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
                
            logger.info(f"Backup created: {backup_path}")
            logger.info(f"Files backed up: {len(manifest['files'])}")
            return manifest
            
        except Exception as e:
            logger.error(f"Error creating backup: {e}")
            if backup_path.exists():
                backup_path.unlink()
            if manifest_path.exists():
                manifest_path.unlink()
            raise
                
    def verify_backup(self, backup_path: Union[str, Path]) -> bool:
        """
        Verify integrity of a backup file.
        
        Args:
            backup_path: Path to backup zip file to verify
            
        Returns:
            True if backup is valid, False otherwise
        """
        backup_path = Path(backup_path)
        if not backup_path.exists():
            logger.error(f"Backup file not found: {backup_path}")
            return False
            
        try:
            # Check if manifest exists
            manifest_path = backup_path.with_name(
                backup_path.stem + "_manifest.json"
            )
            if not manifest_path.exists():
                logger.error(f"Manifest file not found: {manifest_path}")
                return False
                
            # Load manifest
            with open(manifest_path) as f:
                manifest = json.load(f)
                
            # Test zip file
            with zipfile.ZipFile(backup_path) as zf:
                # Test zip integrity
                test_result = zf.testzip()
                if test_result is not None:
                    logger.error(f"Corrupt file in zip: {test_result}")
                    return False
                    
                # Verify all files in manifest are in zip
                zip_files = set(zf.namelist())
                manifest_files = {f["path"] for f in manifest["files"]}
                missing = manifest_files - zip_files
                if missing:
                    logger.error(f"Files in manifest missing from zip: {missing}")
                    return False
                    
            logger.info(f"Backup verified successfully: {backup_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error verifying backup: {e}")
            return False
            
    def restore_backup(self, backup_path: Union[str, Path], 
                      target_dir: Optional[Union[str, Path]] = None,
                      overwrite: bool = False) -> bool:
        """
        Restore a backup to target directory.
        
        Args:
            backup_path: Path to backup zip file
            target_dir: Directory to restore to (default: new directory)
            overwrite: Whether to overwrite existing files
            
        Returns:
            True if restore successful, False otherwise
        """
        backup_path = Path(backup_path)
        if not self.verify_backup(backup_path):
            logger.error(f"Backup verification failed: {backup_path}")
            return False
            
        try:
            # Determine target directory
            if target_dir is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                target_dir = self.project_root.parent / f"restore_{timestamp}"
                
            target_dir = Path(target_dir)
            if target_dir.exists() and not overwrite:
                logger.error(f"Target directory exists: {target_dir}")
                return False
                
            # Create target directory
            target_dir.mkdir(parents=True, exist_ok=True)
            
            # Extract backup
            with zipfile.ZipFile(backup_path) as zf:
                zf.extractall(target_dir)
                
            logger.info(f"Backup restored to: {target_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Error restoring backup: {e}")
            if target_dir and target_dir.exists():
                shutil.rmtree(target_dir)
            return False
            
    def list_backups(self) -> List[Dict]:
        """
        List all backups with their details.
        
        Returns:
            List of dicts with backup information
        """
        backups = []
        
        try:
            for backup_file in self.backup_dir.glob("*_manifest.json"):
                with open(backup_file) as f:
                    manifest = json.load(f)
                    backups.append(manifest)
                    
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            
        return sorted(backups, key=lambda x: x["timestamp"], reverse=True)
        
    def get_latest_backup(self) -> Optional[Dict]:
        """
        Get information about the most recent backup.
        
        Returns:
            Dict with backup information or None if no backups exist
        """
        backups = self.list_backups()
        return backups[0] if backups else None
        
    def cleanup_old_backups(self, keep_days: int = 30) -> int:
        """
        Remove backups older than specified number of days.
        
        Args:
            keep_days: Number of days to keep backups for
            
        Returns:
            Number of backups removed
        """
        removed = 0
        cutoff = datetime.now().timestamp() - (keep_days * 86400)
        
        try:
            for backup_file in self.backup_dir.glob("project_backup_*"):
                if backup_file.stat().st_mtime < cutoff:
                    backup_file.unlink()
                    # Also remove manifest if it exists
                    manifest = backup_file.with_name(
                        backup_file.stem + "_manifest.json"
                    )
                    if manifest.exists():
                        manifest.unlink()
                    removed += 1
                    
            logger.info(f"Removed {removed} old backup(s)")
            return removed
            
        except Exception as e:
            logger.error(f"Error cleaning up old backups: {e}")
            return 0

if __name__ == "__main__":
    # Example usage
    try:
        backup_mgr = BackupManager(
            project_root=Path().resolve(),
            exclude_patterns=[
                "*.pyc",
                "*.pyo",
                "*.pyd",
                "**/__pycache__/**",
                "**/.pytest_cache/**",
                "**/*.egg-info/**",
                ".git/**",
                ".env/**",
                ".venv/**",
                "node_modules/**",
                "*.log",
                "*.db",
                "*.sqlite3"
            ]
        )
        
        # Create a backup
        manifest = backup_mgr.create_backup()
        print(f"Created backup: {manifest['backup_path']}")
        
        # List all backups
        backups = backup_mgr.list_backups()
        print(f"\nExisting backups: {len(backups)}")
        for backup in backups:
            print(f"- {backup['timestamp']}: {len(backup['files'])} files")
            
    except Exception as e:
        print(f"Error in example: {e}")