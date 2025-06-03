import re

# Lee el archivo original
with open('voting_ensemble_corners.py.original', 'r') as f:
    content = f.read()

# Corrige la línea con el problema de "return features    def _predict_with_rf"
content = content.replace('return features    def _predict_with_rf', 'return features\n    \ndef _predict_with_rf')

# Corrige la línea con el problema en "else:            features['is_windy']"
content = content.replace("else:            features['is_windy']", "else:\n            features['is_windy']")

# Corrige la indentación en el método _predict_with_xgb
content = content.replace('raise\n              def _predict_with_xgb', 'raise\n    \ndef _predict_with_xgb')

# Agrega el código para usar DataFrame en _predict_with_rf
content = re.sub(
    r'def _predict_with_rf\(self, features: Dict\[str, float\]\) -> Tuple\[float, float, float\]:\s*"""Make predictions using Random Forest model"""\s*# Convert features to format expected by RF model\s*X = np\.array\(\[list\(features\.values\(\)\)\]\)',
    'def _predict_with_rf(self, features: Dict[str, float]) -> Tuple[float, float, float]:\n        """Make predictions using Random Forest model"""\n        # Convert features to format expected by RF model with proper feature names\n        feature_values = list(features.values())\n        feature_names = list(features.keys())\n        X = pd.DataFrame([feature_values], columns=feature_names)',
    content
)

# Agrega el código para usar DataFrame en _predict_with_xgb
content = re.sub(
    r'def _predict_with_xgb\(self, features: Dict\[str, float\]\) -> Tuple\[float, float, float\]:\s*"""Make predictions using XGBoost model"""\s*# Convert features to format expected by XGBoost model\s*X = np\.array\(\[list\(features\.values\(\)\)\]\)',
    'def _predict_with_xgb(self, features: Dict[str, float]) -> Tuple[float, float, float]:\n        """Make predictions using XGBoost model"""\n        # Convert features to format expected by XGBoost model with proper feature names\n        feature_values = list(features.values())\n        feature_names = list(features.keys())\n        X = pd.DataFrame([feature_values], columns=feature_names)',
    content
)

# Guarda el archivo corregido
with open('voting_ensemble_corners.py.fixed', 'w') as f:
    f.write(content)

print("Archivo corregido guardado como voting_ensemble_corners.py.fixed")
