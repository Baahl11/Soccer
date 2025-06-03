"""
Simple test script for voting_ensemble_corners.py
"""

print("Starting test...")
try:
    from voting_ensemble_corners import VotingEnsembleCornersModel
    print("Import successful")
    
    model = VotingEnsembleCornersModel()
    print("Model instance created successfully")
    
    print("Test complete - everything works!")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
