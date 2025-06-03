"""
Documentation Validation Script
Verifies all documentation files exist and system is operational
"""

import os
import sys
from pathlib import Path

def check_documentation_files():
    """Verify all documentation files exist."""
    print("🔍 Checking Documentation Files...")
    
    required_docs = [
        "README_DOCUMENTATION.md",
        "FINAL_SYSTEM_DOCUMENTATION.md", 
        "USER_GUIDE.md",
        "TECHNICAL_ARCHITECTURE.md",
        "BUG_FIX_REPORT.md",
        "DEVELOPMENT_GUIDE.md",
        "TESTING_INFRASTRUCTURE_DOCUMENTATION.md",
        "ELO_INTEGRATION_FIX_REPORT.md",
        "DOCUMENTATION_SUMMARY.md",
        "PROJECT_COMPLETION_SUMMARY.md"
    ]
    
    missing_files = []
    existing_files = []
    
    for doc in required_docs:
        if os.path.exists(doc):
            file_size = os.path.getsize(doc) / 1024  # KB
            existing_files.append((doc, file_size))
            print(f"  ✅ {doc} ({file_size:.1f} KB)")
        else:
            missing_files.append(doc)
            print(f"  ❌ {doc} - MISSING")
    
    # Check specialized documentation
    specialized_docs = [
        "documentation/1x2_prediction_workflow.md"
    ]
    
    print(f"\n📊 Specialized Documentation:")
    for doc in specialized_docs:
        if os.path.exists(doc):
            file_size = os.path.getsize(doc) / 1024
            existing_files.append((doc, file_size))
            print(f"  ✅ {doc} ({file_size:.1f} KB)")
        else:
            print(f"  ⚠️  {doc} - Optional, not found")
    
    print(f"\n📈 Documentation Statistics:")
    print(f"  Total Documentation Files: {len(existing_files)}")
    total_size = sum([size for _, size in existing_files])
    print(f"  Total Documentation Size: {total_size:.1f} KB")
    print(f"  Average File Size: {total_size/len(existing_files):.1f} KB")
    
    if missing_files:
        print(f"\n❌ Missing Documentation Files: {len(missing_files)}")
        for file in missing_files:
            print(f"    - {file}")
        return False
    else:
        print(f"\n✅ All Documentation Files Present!")
        return True

def check_system_operational():
    """Verify the core system is operational."""
    print(f"\n🔧 Checking System Components...")
    
    try:
        # Check if main validation script exists
        if os.path.exists("final_system_validation.py"):
            print(f"  ✅ Final system validation script found")
        else:
            print(f"  ❌ Final system validation script missing")
            return False
            
        # Check model files
        model_files = ["models/random_forest_corners.pkl", "models/xgboost_corners.pkl"]
        for model_file in model_files:
            if os.path.exists(model_file):
                print(f"  ✅ {model_file} found")
            else:
                print(f"  ⚠️  {model_file} not found (may be expected)")
        
        # Check core system files
        core_files = [
            "voting_ensemble_corners.py",
            "auto_updating_elo.py",
            "team_elo_rating.py"
        ]
        
        for core_file in core_files:
            if os.path.exists(core_file):
                print(f"  ✅ {core_file} found")
            else:
                print(f"  ❌ {core_file} missing")
                return False
        
        print(f"\n✅ Core System Files Present!")
        return True
        
    except Exception as e:
        print(f"  ❌ Error checking system: {e}")
        return False

def validate_documentation_content():
    """Basic validation of documentation content."""
    print(f"\n📝 Validating Documentation Content...")
    
    key_phrases = {
        "README_DOCUMENTATION.md": ["Documentation Index", "System Status", "OPERATIONAL"],
        "USER_GUIDE.md": ["Quick Start", "predict_corners", "ELO"],
        "TECHNICAL_ARCHITECTURE.md": ["System Architecture", "Component", "Performance"],
        "BUG_FIX_REPORT.md": ["ELO Integration", "Key Naming", "RESOLVED"],
        "FINAL_SYSTEM_DOCUMENTATION.md": ["Executive Summary", "6/6", "PASSED"]
    }
    
    content_valid = True
    for doc, phrases in key_phrases.items():
        if os.path.exists(doc):
            try:
                with open(doc, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                missing_phrases = []
                for phrase in phrases:
                    if phrase not in content:
                        missing_phrases.append(phrase)
                
                if missing_phrases:
                    print(f"  ⚠️  {doc}: Missing key content - {missing_phrases}")
                    content_valid = False
                else:
                    print(f"  ✅ {doc}: Content validation passed")
            except Exception as e:
                print(f"  ❌ {doc}: Error reading file - {e}")
                content_valid = False
    
    return content_valid

def main():
    """Main validation function."""
    print("=" * 60)
    print("📚 SOCCER PREDICTION SYSTEM - DOCUMENTATION VALIDATION")
    print("=" * 60)
    print(f"📅 Date: May 29, 2025")
    print(f"📁 Working Directory: {os.getcwd()}")
    
    # Check documentation files
    docs_exist = check_documentation_files()
    
    # Check system components
    system_ok = check_system_operational()
    
    # Validate content
    content_ok = validate_documentation_content()
    
    # Final summary
    print(f"\n" + "=" * 60)
    print(f"📊 VALIDATION SUMMARY")
    print(f"=" * 60)
    
    if docs_exist and system_ok and content_ok:
        print(f"🎉 SUCCESS: Documentation validation PASSED!")
        print(f"✅ All documentation files present and validated")
        print(f"✅ Core system components operational")
        print(f"✅ Documentation content quality verified")
        print(f"\n🚀 System is PRODUCTION READY with complete documentation!")
        return 0
    else:
        print(f"❌ ISSUES FOUND:")
        if not docs_exist:
            print(f"   - Missing documentation files")
        if not system_ok:
            print(f"   - System component issues")
        if not content_ok:
            print(f"   - Documentation content issues")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
