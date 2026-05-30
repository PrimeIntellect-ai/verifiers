from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
import re
import json


class KukurasuVerifier(Verifier):
    """
    Verifier for Kukurasu puzzle
    数独拼图验证器
    """
    def verify(self, data: Data, test_solution: str, **kwargs):
        try:
            grid = self.extract_answer(test_solution)
            # Extract metadata from the data
            row_sums = data.metadata["row_sums"]
            col_sums = data.metadata["col_sums"]
            print(row_sums, col_sums)
            n = data.metadata["n"]
            m = data.metadata["m"]
            
            # Check grid dimensions
            if len(grid) != n:
                return False
                
            for row in grid:
                if len(row) != m:
                    return False
                    
                # Check that each cell contains only "1" or "X"
                for cell in row:
                    if cell not in ["1", "X"]:
                        return False
            
            # Calculate row sums based on the answer grid
            calculated_row_sums = []
            for i, row in enumerate(grid):
                row_sum = 0
                for j, cell in enumerate(row):
                    if cell == "1":
                        # Weight is column position (1-indexed)
                        row_sum += (j + 1)
                calculated_row_sums.append(row_sum)
            
            # Calculate column sums based on the answer grid
            calculated_col_sums = []
            for j in range(m):
                col_sum = 0
                for i in range(n):
                    if grid[i][j] == "1":
                        # Weight is row position (1-indexed)
                        col_sum += (i + 1)
                calculated_col_sums.append(col_sum)
            
            # Check if calculated sums match the expected sums
            if calculated_row_sums != row_sums:
                return False
                
            if calculated_col_sums != col_sums:
                return False
                
            return True
            
        except Exception as e:
            # If any error occurs during verification, return False
            print(f"Verification error: {e}")
            return False
        
        
    def extract_answer(self, response: str):
        """Extract the answer grid from the model's response
        从模型的响应中提取答案网格"""
        # Look for a grid representation in the response
        # 在响应中寻找网格表示
        grid_pattern = r'\[\s*\[(?:\s*"[X1]"\s*,\s*)*\s*"[X1]"\s*\]\s*(?:,\s*\[(?:\s*"[X1]"\s*,\s*)*\s*"[X1]"\s*\]\s*)*\]'
        match = re.search(grid_pattern, response)
        
        if match:
            try:
                # Try to parse the grid as JSON
                # 尝试将网格解析为JSON
                grid_str = match.group(0)
                return json.loads(grid_str)
            except json.JSONDecodeError:
                pass
        
        return ""
