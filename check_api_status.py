#!/usr/bin/env python3
"""
Quick API Status Check Script
Verifica el estado del endpoint de predicciones
"""

import requests
import json
from datetime import datetime

def check_api_status():
    """Check the status of the API endpoint"""
    
    # Test URLs
    base_url = "http://127.0.0.1:5000"
    endpoints = [
        "/api/upcoming_predictions?auto_discovery=true&pretty=1",
        "/api/system_status",
        "/health"
    ]
    
    print("🔍 API Status Check")
    print("=" * 50)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Base URL: {base_url}")
    print()
    
    for endpoint in endpoints:
        url = base_url + endpoint
        try:
            print(f"Testing: {endpoint}")
            response = requests.get(url, timeout=10)
            
            if response.status_code == 200:
                print(f"  ✅ Status: {response.status_code} OK")
                
                # Try to parse JSON
                try:
                    data = response.json()
                    if 'success' in data:
                        print(f"  📊 Success: {data['success']}")
                    if 'status' in data:
                        print(f"  📈 Status: {data['status']}")
                    if 'component_analyses' in data:
                        components = data.get('component_analyses', {})
                        active_components = []
                        for component, info in components.items():
                            if isinstance(info, dict) and info.get('available', False):
                                active_components.append(component)
                        print(f"  🔧 Active Components: {len(active_components)} - {active_components}")
                        
                        # Check for fixture statistics specifically
                        if 'fixture_statistics' in components:
                            fs = components['fixture_statistics']
                            print(f"  📈 Fixture Statistics: Available={fs.get('available', False)}")
                            
                except json.JSONDecodeError:
                    print(f"  ⚠️  Response not JSON format")
                    print(f"  📝 Content preview: {response.text[:100]}...")
                    
            else:
                print(f"  ❌ Status: {response.status_code}")
                print(f"  📝 Response: {response.text[:200]}...")
                
        except requests.exceptions.ConnectionError:
            print(f"  ❌ Connection failed - Server not running?")
        except requests.exceptions.Timeout:
            print(f"  ⏰ Request timeout")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            
        print()

if __name__ == "__main__":
    check_api_status()
