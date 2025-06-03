# This script fixes indentation in the voting_ensemble_corners.py file

with open("c:\\Users\\gm_me\\Soccer\\voting_ensemble_corners.py.bak", "r") as f:
    content = f.read()

# Fix indentation and newlines around the problematic methods
fixed_content = content.replace(
    "        return features\n    def _predict_with_rf", 
    "        return features\n\n    def _predict_with_rf"
).replace(
    "            raise\n    def _predict_with_xgb",
    "            raise\n\n    def _predict_with_xgb"
)

# Save the fixed content back
with open("c:\\Users\\gm_me\\Soccer\\voting_ensemble_corners.py", "w") as f:
    f.write(fixed_content)

print("File fixed successfully")
