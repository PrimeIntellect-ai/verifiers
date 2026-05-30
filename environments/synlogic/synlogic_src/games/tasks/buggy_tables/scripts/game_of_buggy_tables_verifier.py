from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
import re

class BuggyTableVerifier(Verifier):
    """
    Verifier for the BuggyTable game.
    Checks if the submitted answer matches the expected answer.
    """
    def extract_answer(self, answer: str) -> str:
        """
        Public method to extract and normalize an answer string from LLM output.
        Delegates to the private _extract_answer method.
        
        @param answer: The answer string to normalize
        @return: The normalized answer string
        """
        return self._extract_answer(answer)

    def verify(self, data: Data, test_answer: str) -> bool:
        """
        Verify whether the test answer is consistent with the expected answer
        for the buggy table query.
        
        @param data: Data object containing the expected answer
        @param test_answer: The answer provided by the LLM to verify
        @return: bool indicating whether the answer is correct
        """
        # Extract the expected answer from the Data object
        expected_answer = data.answer if data and hasattr(data, 'answer') else ""
        
        # For empty strings, compare directly
        if not expected_answer and not test_answer:
            return True
            
        # Extract and normalize both answers
        normalized_expected = self._extract_answer(expected_answer)
        normalized_test = self._extract_answer(test_answer)
        
        # Direct comparison of normalized answers
        return normalized_expected == normalized_test
        
    def _is_raw_numeric_answer(self, value: str) -> bool:
        """
        Check if a string represents a plain numeric answer without additional context.
        This is used to validate raw input format.
        
        @param value: The string to check
        @return: True if the string is a simple numeric value
        """
        # Remove whitespace
        value = value.strip()
        
        # Simple pattern match for a number (optionally with sign and decimal point)
        import re
        return bool(re.match(r'^-?\d+(\.\d+)?$', value))
        
    def _raw_has_exactly_two_decimals(self, value: str) -> bool:
        """
        Check if a raw numeric string has exactly 2 decimal places.
        This is used to validate the format of the raw answer.
        
        @param value: The string to check
        @return: True if the string has exactly 2 decimal places
        """
        # Remove whitespace
        value = value.strip()
        
        # Split on decimal point
        parts = value.replace('-', '', 1).split('.')
        
        # Check if there is exactly one decimal point and two digits after it
        return len(parts) == 2 and len(parts[1]) == 2
    
    def _is_numeric(self, value: str) -> bool:
        """
        Check if a string represents a valid number (including negative numbers and decimals).
        
        @param value: The string to check
        @return: True if the string represents a valid number
        """
        # Remove negative sign if present
        value = value.strip()
        if value.startswith('-'):
            value = value[1:]
        # Check if remaining string is a valid decimal number
        return value.replace('.', '', 1).isdigit()
    
    def _has_exactly_two_decimals(self, value: str) -> bool:
        """
        Check if a number string has exactly 2 decimal places.
        
        @param value: The number string to check
        @return: True if the number has exactly 2 decimal places
        """
        # Remove negative sign if present
        value = value.strip()
        if value.startswith('-'):
            value = value[1:]
            
        # Split into whole and decimal parts
        parts = value.split('.')
        if len(parts) != 2:
            return False
            
        # Check if decimal part has exactly 2 digits
        return len(parts[1]) == 2
    
    def _extract_answer(self, answer: str) -> str:
        """
        Extract and normalize an answer string from LLM output.
        Only finds values with exactly two decimal places.
        
        @param answer: The answer string to normalize
        @return: The normalized answer string
        """
        # Convert to string and normalize
        normalized = str(answer).strip() if answer is not None else ""
        
        # Try to find numbers with exactly two decimal places
        exact_matches = re.findall(r'-?\d+\.\d{2}\b', normalized)
        if exact_matches:
            return exact_matches[-1]  # Return the last match with exactly two decimals
            
        # If no exact two-decimal match found, return the original string
        return normalized