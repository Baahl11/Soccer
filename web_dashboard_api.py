#!/usr/bin/env python3
"""
Soccer Prediction Web Dashboard - REST API

This module provides a Flask-based REST API for the Soccer Prediction System
with beautiful web interface for visualizing JSON results in an intuitive way.
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List
import os
import sqlite3

# Import our prediction systems
from advanced_1x2_system import create_advanced_1x2_system
from enhanced_match_winner import predict_with_enhanced_system
from prediction_integration import make_integrated_prediction

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for frontend-backend communication

# Initialize prediction system
advanced_system = create_advanced_1x2_system()

def create_user_friendly_interpretation(probs, winner, confidence, home_team, away_team):
    """Create a user-friendly interpretation of the prediction results."""
    interpretation = {
        'summary': '',
        'confidence_explanation': '',
        'probability_analysis': '',
        'recommendation': ''
    }
    
    # Main summary
    outcome_map = {
        'home_win': f'{home_team} victory',
        'draw': 'a draw',
        'away_win': f'{away_team} victory'
    }
    
    interpretation['summary'] = f"The model predicts {outcome_map[winner]} with {confidence}% confidence."
    
    # Confidence explanation
    if confidence >= 70:
        interpretation['confidence_explanation'] = "High confidence - Strong statistical backing for this prediction."
    elif confidence >= 50:
        interpretation['confidence_explanation'] = "Medium confidence - Moderate statistical support."
    else:
        interpretation['confidence_explanation'] = "Low confidence - Close match with uncertain outcome."
    
    # Probability analysis
    home_prob = probs['home_win']
    draw_prob = probs['draw']
    away_prob = probs['away_win']
    
    if abs(home_prob - away_prob) <= 10:
        interpretation['probability_analysis'] = "Very evenly matched teams with similar winning chances."
    elif draw_prob >= 30:
        interpretation['probability_analysis'] = "High probability of a draw - defensive game expected."
    else:
        interpretation['probability_analysis'] = f"Clear advantage to the predicted winner."
    
    # Betting recommendation (educational)
    if confidence >= 65:
        interpretation['recommendation'] = f"Strong statistical support for {outcome_map[winner]}."
    elif confidence >= 45:
        interpretation['recommendation'] = "Moderate prediction strength - consider other factors."
    else:
        interpretation['recommendation'] = "Uncertain outcome - avoid high-risk decisions."
    
    return interpretation

@app.route('/')
def dashboard():
    """Main dashboard page."""
    return render_template('dashboard.html')

@app.route('/api/predict', methods=['POST'])
def predict_match():
    """
    API endpoint for match prediction.
    
    Expected JSON input:
    {
        "home_team_id": 33,
        "away_team_id": 40,
        "league_id": 39,
        "home_team_name": "Manchester United",
        "away_team_name": "Liverpool",
        "use_enhanced": true
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract required fields
        home_team_id = data.get('home_team_id')
        away_team_id = data.get('away_team_id')
        league_id = data.get('league_id', 39)  # Default to Premier League
        
        if not home_team_id or not away_team_id:
            return jsonify({'error': 'home_team_id and away_team_id are required'}), 400
        
        # Get team names for display
        home_team_name = data.get('home_team_name', f'Team {home_team_id}')
        away_team_name = data.get('away_team_name', f'Team {away_team_id}')
        
        # Make advanced prediction
        prediction = advanced_system.predict_match_advanced(
            home_team_id=int(home_team_id),
            away_team_id=int(away_team_id),
            league_id=int(league_id),
            use_calibration=True
        )
        
        # Format response for web display
        base_pred = prediction.get('base_prediction', {})
        probabilities = base_pred.get('probabilities', {})
        advanced_metrics = prediction.get('advanced_metrics', {})
          # Convert probabilities to percentages and round
        # Handle cases where probabilities might already be percentages or invalid
        home_prob = probabilities.get('home_win', 0)
        draw_prob = probabilities.get('draw', 0)
        away_prob = probabilities.get('away_win', 0)
        
        # If probabilities sum to more than 1.5, they're likely already percentages
        total_prob = home_prob + draw_prob + away_prob
        if total_prob > 1.5:
            # Already percentages, just round
            formatted_probs = {
                'home_win': round(home_prob, 1),
                'draw': round(draw_prob, 1),
                'away_win': round(away_prob, 1)
            }
        else:
            # Convert to percentages
            formatted_probs = {
                'home_win': round(home_prob * 100, 1),
                'draw': round(draw_prob * 100, 1),
                'away_win': round(away_prob * 100, 1)
            }
        
        # Determine winner and confidence
        max_prob_outcome = max(formatted_probs.items(), key=lambda x: x[1])
        winner = max_prob_outcome[0]
        confidence = max_prob_outcome[1]
        
        # Map outcome to display text
        outcome_text = {
            'home_win': f'{home_team_name} Wins',
            'draw': 'Draw',
            'away_win': f'{away_team_name} Wins'
        }
          # Confidence level text
        confidence_level = advanced_metrics.get('confidence_level', 'medium').title()
        
        # Create user-friendly interpretation
        interpretation = create_user_friendly_interpretation(
            formatted_probs, winner, confidence, home_team_name, away_team_name
        )
        
        response = {
            'prediction_id': prediction.get('prediction_id'),
            'timestamp': prediction.get('timestamp'),
            'match_info': {
                'home_team': home_team_name,
                'away_team': away_team_name,
                'league_id': league_id
            },
            'prediction': {
                'outcome': outcome_text[winner],
                'winner': winner,
                'confidence': confidence,
                'confidence_level': confidence_level
            },
            'probabilities': formatted_probs,
            'interpretation': interpretation,
            'advanced_metrics': {
                'entropy': round(advanced_metrics.get('entropy', 0), 3),
                'probability_spread': round(advanced_metrics.get('probability_spread', 0), 3),
                'draw_favorability': round(advanced_metrics.get('draw_favorability', 0), 3)
            },
            'system_info': prediction.get('system_info', {}),
            'status': 'success'
        }
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in prediction API: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/batch_predict', methods=['POST'])
def batch_predict():
    """
    API endpoint for batch match predictions.
    
    Expected JSON input:
    {
        "matches": [
            {
                "home_team_id": 33,
                "away_team_id": 40,
                "league_id": 39,
                "home_team_name": "Manchester United",
                "away_team_name": "Liverpool"
            },
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        matches = data.get('matches', [])
        
        if not matches:
            return jsonify({'error': 'No matches provided'}), 400
        
        results = []
        
        for match in matches:
            try:
                # Make prediction for each match
                prediction = advanced_system.predict_match_advanced(
                    home_team_id=int(match.get('home_team_id')),
                    away_team_id=int(match.get('away_team_id')),
                    league_id=int(match.get('league_id', 39))
                )
                
                # Format for web display
                base_pred = prediction.get('base_prediction', {})
                probabilities = base_pred.get('probabilities', {})
                
                formatted_probs = {
                    'home_win': round(probabilities.get('home_win', 0) * 100, 1),
                    'draw': round(probabilities.get('draw', 0) * 100, 1),
                    'away_win': round(probabilities.get('away_win', 0) * 100, 1)
                }
                
                max_prob_outcome = max(formatted_probs.items(), key=lambda x: x[1])
                winner = max_prob_outcome[0]
                
                result = {
                    'match_info': {
                        'home_team': match.get('home_team_name', f"Team {match.get('home_team_id')}"),
                        'away_team': match.get('away_team_name', f"Team {match.get('away_team_id')}")
                    },
                    'prediction': {
                        'winner': winner,
                        'confidence': max_prob_outcome[1]
                    },
                    'probabilities': formatted_probs
                }
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error predicting match {match}: {e}")
                continue
        
        return jsonify({
            'results': results,
            'total_matches': len(results),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error in batch prediction API: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/performance', methods=['GET'])
def get_performance_metrics():
    """Get system performance metrics."""
    try:
        metrics = advanced_system.evaluate_system_performance()
        
        # Format metrics for display
        formatted_metrics = {
            'accuracy': round(metrics.get('accuracy', 0) * 100, 1),
            'draw_precision': round(metrics.get('draw_precision', 0) * 100, 1),
            'draw_recall': round(metrics.get('draw_recall', 0) * 100, 1),
            'brier_score': round(metrics.get('brier_score', 0), 3),
            'total_predictions': metrics.get('total_predictions', 0),
            'calibration_rate': round(metrics.get('calibration_rate', 0) * 100, 1),
            'last_updated': metrics.get('evaluation_timestamp')
        }
        
        return jsonify({
            'metrics': formatted_metrics,
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error getting performance metrics: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/system_status', methods=['GET'])
def get_system_status():
    """Get current system status."""
    try:
        status = advanced_system.get_system_status()
        return jsonify({
            'status': status,
            'api_status': 'operational',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return jsonify({'error': str(e)}, 500)

@app.route('/api/recent_predictions', methods=['GET'])
def get_recent_predictions():
    """Get recent predictions from database."""
    try:
        limit = request.args.get('limit', 10, type=int)
        
        conn = sqlite3.connect(advanced_system.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT home_team_id, away_team_id, predicted_outcome,
                   home_win_prob, draw_prob, away_win_prob,
                   confidence, timestamp, calibrated
            FROM predictions 
            ORDER BY timestamp DESC 
            LIMIT ?
        ''', (limit,))
        
        results = cursor.fetchall()
        conn.close()
        
        predictions = []
        for row in results:
            predictions.append({
                'home_team_id': row[0],
                'away_team_id': row[1],
                'predicted_outcome': row[2],
                'probabilities': {
                    'home_win': round(row[3] * 100, 1),
                    'draw': round(row[4] * 100, 1),
                    'away_win': round(row[5] * 100, 1)
                },
                'confidence': round(row[6], 3),
                'timestamp': row[7],
                'calibrated': bool(row[8])
            })
        
        return jsonify({
            'predictions': predictions,
            'count': len(predictions),
            'status': 'success'
        })
        
    except Exception as e:
        logger.error(f"Error getting recent predictions: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

@app.route('/api/predict/formatted', methods=['POST'])
def predict_match_formatted():
    """
    API endpoint for beautifully formatted match prediction results.
    Returns human-readable JSON with detailed explanations.
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        # Extract required fields
        home_team_id = data.get('home_team_id')
        away_team_id = data.get('away_team_id')
        league_id = data.get('league_id', 39)
        
        if not home_team_id or not away_team_id:
            return jsonify({'error': 'home_team_id and away_team_id are required'}), 400
        
        # Get team names for display
        home_team_name = data.get('home_team_name', f'Team {home_team_id}')
        away_team_name = data.get('away_team_name', f'Team {away_team_id}')
        
        # Make prediction
        prediction = advanced_system.predict_match_advanced(
            home_team_id=int(home_team_id),
            away_team_id=int(away_team_id),
            league_id=int(league_id),
            use_calibration=True
        )
        
        # Format response
        base_pred = prediction.get('base_prediction', {})
        probabilities = base_pred.get('probabilities', {})
        advanced_metrics = prediction.get('advanced_metrics', {})
        
        # Handle probability formatting
        home_prob = probabilities.get('home_win', 0)
        draw_prob = probabilities.get('draw', 0)
        away_prob = probabilities.get('away_win', 0)
        
        total_prob = home_prob + draw_prob + away_prob
        if total_prob > 1.5:
            formatted_probs = {
                'home_win': round(home_prob, 1),
                'draw': round(draw_prob, 1),
                'away_win': round(away_prob, 1)
            }
        else:
            formatted_probs = {
                'home_win': round(home_prob * 100, 1),
                'draw': round(draw_prob * 100, 1),
                'away_win': round(away_prob * 100, 1)
            }
        
        # Get winner
        max_prob_outcome = max(formatted_probs.items(), key=lambda x: x[1])
        winner = max_prob_outcome[0]
        confidence = max_prob_outcome[1]
        
        # Create interpretation
        interpretation = create_user_friendly_interpretation(
            formatted_probs, winner, confidence, home_team_name, away_team_name
        )
        
        # Create beautiful formatted response
        formatted_response = {
            "🏆 SOCCER MATCH PREDICTION": {
                "📅 Match Details": {
                    "🏠 Home Team": home_team_name,
                    "🛣️ Away Team": away_team_name,
                    "🏟️ League ID": league_id,
                    "⏰ Prediction Time": prediction.get('timestamp', datetime.now().isoformat())
                },
                "🎯 Prediction Result": {
                    "🏅 Predicted Outcome": f"{home_team_name} Wins" if winner == 'home_win' 
                                          else "Draw" if winner == 'draw' 
                                          else f"{away_team_name} Wins",
                    "📊 Confidence Level": f"{confidence}% ({advanced_metrics.get('confidence_level', 'medium').title()})",
                    "💡 Summary": interpretation['summary']
                },
                "📈 Probability Breakdown": {
                    f"🏠 {home_team_name} Win": f"{formatted_probs['home_win']}%",
                    "🤝 Draw": f"{formatted_probs['draw']}%",
                    f"🛣️ {away_team_name} Win": f"{formatted_probs['away_win']}%"
                },
                "🧠 AI Analysis": {
                    "💯 Confidence Explanation": interpretation['confidence_explanation'],
                    "⚖️ Probability Analysis": interpretation['probability_analysis'],
                    "💼 Recommendation": interpretation['recommendation']
                },
                "🔬 Advanced Metrics": {
                    "🌀 Entropy (Uncertainty)": f"{round(advanced_metrics.get('entropy', 0), 3)} (lower = more certain)",
                    "📊 Probability Spread": f"{round(advanced_metrics.get('probability_spread', 0), 3)}",
                    "🤝 Draw Favorability": f"{round(advanced_metrics.get('draw_favorability', 0), 3)}"
                },
                "🖥️ System Information": {
                    "🤖 AI Version": prediction.get('system_info', {}).get('version', 'N/A'),
                    "⚙️ Enhanced System": "✅ Active" if prediction.get('system_info', {}).get('enhanced_system') else "❌ Inactive",
                    "🎯 Calibration": "✅ Enabled" if prediction.get('system_info', {}).get('calibration_enabled') else "❌ Disabled",
                    "⚖️ SMOTE Balancing": "✅ Applied" if prediction.get('system_info', {}).get('smote_balanced') else "❌ Not Applied"
                }
            }
        }
        
        return jsonify(formatted_response)
        
    except Exception as e:
        logger.error(f"Error in formatted prediction API: {e}")
        return jsonify({'error': str(e), 'status': 'error'}), 500

# Serve static files
@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files."""
    return send_from_directory('static', filename)

if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    print("🚀 Soccer Prediction Dashboard")
    print("=" * 40)
    print("📊 Web Interface: http://localhost:5000")
    print("🔗 API Endpoints:")
    print("   POST /api/predict - Single match prediction")
    print("   POST /api/batch_predict - Multiple match predictions")
    print("   GET /api/performance - System performance metrics")
    print("   GET /api/system_status - System status")
    print("   GET /api/recent_predictions - Recent predictions")
    print("=" * 40)
    
    # Run the Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)
