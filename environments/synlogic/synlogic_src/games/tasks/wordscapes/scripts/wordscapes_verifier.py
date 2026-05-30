"""
Wordscapes verifier module for the reasonreason framework.
"""

import json
import re
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END

debug_mode = False

class WordscapesVerifier(Verifier):
    """
    Verifier for Wordscapes game
    """
    def verify(self, data, test_solution: str):
        """
        Verify whether the test answer is consistent with the gold answer
        
        Args:
            data: WordscapesData
            test_solution: str containing the solution
            
        Returns:
            float: Score between 0 and 1
        """
        try:
            extracted_answer = self.extract_answer(test_solution)
            if not extracted_answer:
                print("Failed to extract answer from test solution")
                return False
            
            if debug_mode:
                for row in extracted_answer:
                    print(" ".join(cell if cell != " " else "_" for cell in row))
            
            # Get grid, across_words, and down_words from data
            grid = data.metadata["grid"]
            across_words = data.metadata["across_words"]
            down_words = data.metadata["down_words"]
            
            # Validate grid dimensions
            if len(extracted_answer) != len(grid):
                print(f"Grid height mismatch: expected {len(grid)}, got {len(extracted_answer)}")
                return False
            
            for i in range(len(grid)):
                if len(extracted_answer[i]) != len(grid[i]):
                    print(f"Grid width mismatch at row {i}: expected {len(grid[i])}, got {len(extracted_answer[i])}")
                    return False
            
            # Check if the answer respects the grid layout (X for letters, 0 for empty)
            for i in range(len(grid)):
                for j in range(len(grid[i])):
                    if grid[i][j] == "0" and extracted_answer[i][j].strip():
                        print(f"Expected empty space at position ({i},{j}), got '{extracted_answer[i][j]}'")
                        return False
                    if grid[i][j] == "X" and not extracted_answer[i][j].strip():
                        print(f"Expected letter at position ({i},{j}), got empty space")
                        return False
            
            # Verify across words
            for word in across_words:
                found = False
                for i in range(len(extracted_answer)):
                    row_str = ''.join(extracted_answer[i]).replace(' ', '').lower()
                    if word.lower() in row_str:
                        found = True
                        break
                if not found and word:
                    print(f"Across word '{word}' not found in the grid")
                    return 0
            
            # Verify down words
            for word in down_words:
                found = False
                for j in range(len(extracted_answer[0])):
                    col = []
                    for i in range(len(extracted_answer)):
                        if j < len(extracted_answer[i]):
                            col.append(extracted_answer[i][j])
                    col_str = ''.join(col).replace(' ', '').lower()
                    if word.lower() in col_str:
                        found = True
                        break
                if not found and word:  # Only check if word is not empty
                    print(f"Down word '{word}' not found in the grid")
                    return False
            
            # All checks passed
            return True
        except Exception as e:
            print(f"Error in verification: {str(e)}")
            return False
    
    def extract_answer(self, test_solution: str):
        """
        Extract the answer from the test solution
        
        Args:
            test_solution: str
            
        Returns:
            list: 2D grid of the answer or None if extraction fails
        """
        try:
            # Remove thoughts if present
            if THOUGHT_DELIMITER_START in test_solution and THOUGHT_DELIMITER_END in test_solution:
                # Extract only the part after the thoughts
                thought_end_pos = test_solution.rfind(THOUGHT_DELIMITER_END)
                if thought_end_pos >= 0:
                    test_solution = test_solution[thought_end_pos + len(THOUGHT_DELIMITER_END):]
            
            # Clean up the response and find the grid pattern
            # Look for a pattern like [[...]] or [[[...]]]
            grid_pattern = re.search(r'\[\s*\[(?:\s*\[)?(.+?)(?:\]\s*)?\]\s*\]', test_solution, re.DOTALL)
            if not grid_pattern:
                return None
                
            grid_text = grid_pattern.group(1)
            
            # Handle various formats
            rows = []
            
            # Check if rows are separated by commas
            split_rows = re.split(r'\],\s*\[', grid_text)
            
            for row_text in split_rows:
                # Clean the row text and extract characters
                row_text = row_text.strip().strip('[],')
                
                # Extract quoted characters: "X" or 'X' or just X
                chars = []
                
                # Look for quoted strings or standalone characters
                char_matches = re.findall(r'\"([^\"]*)\"|\'([^\']*)\'|([^,\s]+)', row_text)
                
                for match in char_matches:
                    # Take the first non-empty group from each match
                    char = next((x for x in match if x), "")
                    
                    # Handle numeric or empty values (0, "", '')
                    if char == "0" or char == "":
                        char = " "
                        
                    chars.append(char)
                
                if chars:  # Only add non-empty rows
                    rows.append(chars)
            
            # Make sure we have a valid grid
            if not rows or not all(rows):
                return None
                
            return rows
                
        except Exception as e:
            print(f"Error extracting answer: {str(e)}")
            return None
       