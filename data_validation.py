from typing import Dict, Any, Optional, Union
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataValidator:
    @staticmethod
    def validate_fixture_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validates fixture data structure and content
        Returns: (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "Data must be a dictionary"

        required_fields = {
            'fixture_id': int,
            'teams': dict,
            'league': dict,
            'date': str
        }

        for field, field_type in required_fields.items():
            if field not in data:
                return False, f"Missing required field: {field}"
            if not isinstance(data[field], field_type):
                return False, f"Invalid type for {field}: expected {field_type.__name__}"

        # Validate date format
        try:
            datetime.strptime(data['date'], "%Y-%m-%d")
        except ValueError:
            return False, "Invalid date format, expected YYYY-MM-DD"

        return True, None

    @staticmethod
    def validate_fixture_id(fixture_id: Any) -> bool:
        """
        Validates that a fixture_id is a positive integer
        Returns: True if valid, False otherwise
        """
        if fixture_id is None:
            logger.warning("Fixture ID is None")
            return False
        
        try:
            # Convert to int if it's a string
            fixture_id_int = int(fixture_id)
            
            # Check if it's a positive integer
            if fixture_id_int <= 0:
                logger.warning(f"Fixture ID must be positive, got {fixture_id_int}")
                return False
                
            return True
        except (ValueError, TypeError):
            logger.warning(f"Fixture ID must be an integer, got {type(fixture_id)}: {fixture_id}")
            return False

    @staticmethod
    def validate_player_data(data: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validates player data structure and content
        Returns: (is_valid, error_message)
        """
        if not isinstance(data, dict):
            return False, "Data must be a dictionary"

        required_fields = {
            'player_id': int,
            'name': str,
            'statistics': list
        }

        for field, field_type in required_fields.items():
            if field not in data:
                return False, f"Missing required field: {field}"
            if not isinstance(data[field], field_type):
                return False, f"Invalid type for {field}: expected {field_type.__name__}"

        return True, None

    @staticmethod
    def validate_response(response: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validates API response structure
        Returns: (is_valid, error_message)
        """
        if "errors" in response:
            return False, str(response["errors"])
        
        if "response" not in response:
            return False, "Invalid API response structure"
            
        return True, None