from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
import re
import ast
import numpy as np

class NumbrixVerifier(Verifier):
    """
    Numbrix 游戏的验证器
    验证提交的解答是否符合 Numbrix 游戏规则
    """
    def verify(self, data: Data, test_solution: str):
        try:
            # 提取答案网格
            test_grid = self.extract_answer(test_solution)
            if not test_grid:
                return False
            
            # 获取原始谜题和网格大小
            original_grid = data.metadata["grid"]
            n = len(original_grid)
            n_squared = n * n
            
            # 检查网格大小是否正确
            if len(test_grid) != n or any(len(row) != n for row in test_grid):
                return False
            
            # 检查是否包含所有数字 1 到 n²
            flattened_grid = [cell for row in test_grid for cell in row]
            if sorted(flattened_grid) != list(range(1, n_squared + 1)):
                return False
            
            # 检查是否保留了原始提示数字
            for i in range(n):
                for j in range(n):
                    if original_grid[i][j] != "X" and test_grid[i][j] != original_grid[i][j]:
                        return False
            
            # 检查连续数字是否正交相邻
            for num in range(1, n_squared):
                # 找到当前数字的位置
                current_pos = None
                next_pos = None
                for i in range(n):
                    for j in range(n):
                        if test_grid[i][j] == num:
                            current_pos = (i, j)
                        elif test_grid[i][j] == num + 1:
                            next_pos = (i, j)
                
                if current_pos is None or next_pos is None:
                    return False
                
                # 检查是否正交相邻（曼哈顿距离为1）
                i1, j1 = current_pos
                i2, j2 = next_pos
                manhattan_distance = abs(i1 - i2) + abs(j1 - j2)
                if manhattan_distance != 1:
                    return False
            
            return True
        except Exception as e:
            print(f"验证过程中出错: {e}")
            return False
        
    def extract_answer(self, test_solution: str, strict=False):
        """从模型回答中提取网格"""
        try:
            import ast
            import re
            # 尝试找到 Python 列表格式的答案
            # 寻找形如 [[1, 2, 3], [4, 5, 6], [7, 8, 9]] 的模式
            pattern = r'\[\s*\[\s*\d+.*?\]\s*\]'
            matches = re.finditer(pattern, test_solution, re.DOTALL)
            match = None
            
            # 获取最后一个匹配项
            for m in matches:
                match = m
            if not match:
                return None
            
            # 提取匹配的文本并尝试解析为 Python 对象
            grid_text = match.group(0)
            
            # 清理文本，确保它是有效的 Python 列表
            # 移除可能导致解析错误的字符
            grid_text = grid_text.replace("'", "").replace('"', "")
            
            # 解析为 Python 对象
            grid = ast.literal_eval(grid_text)
            
            # 确保是二维列表且所有元素都是整数
            if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
                return None
            
            if not all(isinstance(cell, int) for row in grid for cell in row):
                return None
            
            return grid
        except Exception as e:
            print(f"提取答案时出错: {e}")
            return None