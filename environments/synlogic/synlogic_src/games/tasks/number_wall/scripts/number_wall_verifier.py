from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
import re
import json
from collections import deque

class NumberWallVerifier(Verifier):
    """
    Verifier for Number Wall puzzle
    数字墙拼图验证器
    """
    def verify(self, data: Data, test_solution: str, **kwargs):
        try:
            # 提取答案网格
            solution_grid = self.extract_answer(test_solution)
            if not solution_grid:
                print("Failed to extract solution grid")
                return False
                
            # 提取元数据
            original_grid = data.metadata["grid"]
            n = data.metadata["n"]
            
            # 检查网格尺寸
            if len(solution_grid) != n:
                print(f"Solution grid has incorrect number of rows: {len(solution_grid)} != {n}")
                return False
                
            for row in solution_grid:
                if len(row) != n:
                    print(f"Solution grid has incorrect number of columns: {len(row)} != {n}")
                    return False
                    
                # 检查每个单元格只包含数字、"X"或"A"
                for cell in row:
                    if not (isinstance(cell, int) or cell in ["X", "A"]):
                        print(f"Invalid cell content: {cell}")
                        return False
            
            # 检查原始数字是否保留
            if not self._check_original_numbers(original_grid, solution_grid):
                print("Original numbers not preserved")
                return False
                
            # 检查墙壁布局是否有效（没有2×2或更大的连续墙块）
            if not self._check_wall_layout(solution_grid):
                print("Invalid wall layout (2x2 or larger continuous wall blocks found)")
                return False
                
            # 检查岛屿划分是否有效
            if not self._check_islands(solution_grid):
                print("Invalid island division")
                return False
                
            # 检查是否有斜线边
            if not self._check_diagonal_borders(solution_grid):
                print("Invalid solution: islands have diagonal borders")
                return False
                
            return True
            
        except Exception as e:
            # 如果验证过程中发生任何错误，返回False
            print(f"Verification error: {e}")
            return False
    
    def _check_original_numbers(self, original_grid, solution_grid):
        """检查原始数字是否在解决方案中保留"""
        for i in range(len(original_grid)):
            for j in range(len(original_grid[i])):
                if isinstance(original_grid[i][j], int):
                    if original_grid[i][j] != solution_grid[i][j]:
                        print(f"Original number at ({i},{j}) changed: {original_grid[i][j]} -> {solution_grid[i][j]}")
                        return False
        return True
    
    def _check_wall_layout(self, grid):
        """检查墙壁布局是否有效（没有2×2或更大的连续墙块）"""
        n = len(grid)
        for i in range(n - 1):
            for j in range(n - 1):
                if (grid[i][j] == "A" and grid[i][j+1] == "A" and
                    grid[i+1][j] == "A" and grid[i+1][j+1] == "A"):
                    print(f"Found 2x2 wall block at ({i},{j})")
                    return False
        return True
    
    def _check_islands(self, grid):
        """检查岛屿划分是否有效"""
        n = len(grid)
        visited = set()
        
        for i in range(n):
            for j in range(n):
                if (i, j) not in visited and grid[i][j] != "A":
                    # 发现一个新岛屿
                    island_cells = []
                    island_number = None
                    queue = deque([(i, j)])
                    visited.add((i, j))
                    
                    while queue:
                        r, c = queue.popleft()
                        island_cells.append((r, c))
                        
                        if isinstance(grid[r][c], int):
                            if island_number is not None:
                                # 岛屿有多个数字
                                print(f"Island contains multiple numbers: {island_number} and {grid[r][c]}")
                                return False
                            island_number = grid[r][c]
                        
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < n and 0 <= nc < n and 
                                (nr, nc) not in visited and 
                                grid[nr][nc] != "A"):
                                queue.append((nr, nc))
                                visited.add((nr, nc))
                    
                    if island_number is None:
                        # 岛屿没有数字
                        print(f"Island at ({i},{j}) has no number")
                        return False
                    
                    if len(island_cells) != island_number:
                        # 岛屿大小与数字不匹配
                        print(f"Island size ({len(island_cells)}) doesn't match number ({island_number})")
                        return False
        
        return True
    
    def _check_diagonal_borders(self, grid):
        """检查是否有斜线边（对角相邻的不同岛屿）"""
        n = len(grid)
        
        # 标记所有岛屿
        island_map = {}  # 映射格子坐标到岛屿ID
        island_id = 0
        visited = set()
        
        for i in range(n):
            for j in range(n):
                if grid[i][j] != "A" and (i, j) not in visited:
                    # 发现一个新岛屿
                    queue = deque([(i, j)])
                    visited.add((i, j))
                    
                    while queue:
                        r, c = queue.popleft()
                        island_map[(r, c)] = island_id
                        
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if (0 <= nr < n and 0 <= nc < n and 
                                grid[nr][nc] != "A" and (nr, nc) not in visited):
                                queue.append((nr, nc))
                                visited.add((nr, nc))
                    
                    island_id += 1
        
        # 检查斜线边
        for i in range(n - 1):
            for j in range(n - 1):
                # 检查2x2方格中的对角格子
                if (grid[i][j] != "A" and grid[i+1][j+1] != "A" and
                    grid[i][j+1] == "A" and grid[i+1][j] == "A"):
                    # 对角格子属于不同岛屿，存在斜线边
                    if island_map.get((i, j)) != island_map.get((i+1, j+1)):
                        print(f"Found diagonal border at ({i},{j}) and ({i+1},{j+1})")
                        return False
                
                # 检查另一对对角格子
                if (grid[i][j+1] != "A" and grid[i+1][j] != "A" and
                    grid[i][j] == "A" and grid[i+1][j+1] == "A"):
                    # 对角格子属于不同岛屿，存在斜线边
                    if island_map.get((i, j+1)) != island_map.get((i+1, j)):
                        print(f"Found diagonal border at ({i},{j+1}) and ({i+1},{j})")
                        return False
        
        return True
        
 
    def extract_answer(self, response: str):
        """从模型的响应中提取答案网格"""    
        # 在响应中寻找网格表示
        # 修改正则表达式以匹配字符串形式的数字
        grid_pattern = r'\[\s*\[(?:\s*(?:"[XA]"|\'[XA]\'|[0-9]+|"[0-9]+"|\'[0-9]+\')\s*,\s*)*\s*(?:"[XA]"|\'[XA]\'|[0-9]+|"[0-9]+"|\'[0-9]+\')\s*\]\s*(?:,\s*\[(?:\s*(?:"[XA]"|\'[XA]\'|[0-9]+|"[0-9]+"|\'[0-9]+\')\s*,\s*)*\s*(?:"[XA]"|\'[XA]\'|[0-9]+|"[0-9]+"|\'[0-9]+\')\s*\]\s*)*\]'
        matches = re.findall(grid_pattern, response)
        
        if matches:
            # 尝试解析最后一个匹配项
            grid_str = matches[-1]
            
            try:
                # 尝试清理字符串，替换可能导致问题的字符
                cleaned_grid_str = grid_str.replace('\n', '').replace('\r', '').strip()
                grid = json.loads(cleaned_grid_str)
                
                # 将字符串数字转换为整数
                for i in range(len(grid)):
                    for j in range(len(grid[i])):
                        if isinstance(grid[i][j], str) and grid[i][j].isdigit():
                            grid[i][j] = int(grid[i][j])
                
                return grid
            except json.JSONDecodeError as e:
                # 尝试使用 ast.literal_eval 作为备选方案
                try:
                    import ast
                    grid = ast.literal_eval(cleaned_grid_str)
                    
                    # 将字符串数字转换为整数
                    for i in range(len(grid)):
                        for j in range(len(grid[i])):
                            if isinstance(grid[i][j], str) and grid[i][j].isdigit():
                                grid[i][j] = int(grid[i][j])
                    
                    return grid
                except Exception as e2:
                    print(f"ast.literal_eval also failed: {e2}")
        else:
            print("No grid pattern found in the response")
        
        return None
