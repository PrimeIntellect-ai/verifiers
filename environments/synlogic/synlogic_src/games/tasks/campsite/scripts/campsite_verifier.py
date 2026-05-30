from base.data import Data
from base.verifier import Verifier
import re
import ast
from typing import List, Set, Tuple, Dict


class CampsiteVerifier(Verifier):
    """
    Verifier for Campsite game
    """
    def verify(self, data: Data, test_solution: str):
        try:
            test_answer = self.extract_answer(test_solution)
            original_grid = data.metadata["grid"]
            row_constraints = data.metadata["row_constraints"]
            col_constraints = data.metadata["col_constraints"]
            n = data.metadata["n"]
            m = data.metadata["m"]
            
            if not test_answer:
                return False
            
            if len(test_answer) != n or any(len(row) != m for row in test_answer):
                return False
            
            if not self._check_trees_unchanged(original_grid, test_answer):
                return False
            
            if not self._check_row_constraints(test_answer, row_constraints):
                return False
            
            if not self._check_col_constraints(test_answer, col_constraints):
                return False
            
            if not self._check_tents_not_adjacent(test_answer):
                return False
            
            if not self._check_tent_tree_matching(test_answer):
                return False
            
            return True
            
        except Exception as e:
            print(f"Verification error: {e}")
            return False
    
    def _extract_grid(self, test_answer: str) -> List[List[str]]:
        """从回答中提取网格"""
        grid_pattern = r'\[\s*\[.*?\]\s*\]'
        match = re.search(grid_pattern, test_answer, re.DOTALL)
        if match:
            try:
                grid_str = match.group(0)
                return ast.literal_eval(grid_str)
            except:
                pass
        
        return None
    
    def _check_trees_unchanged(self, original_grid: List[List[str]], test_answer: List[List[str]]) -> bool:
        """检查树木位置是否保持不变"""
        for i in range(len(original_grid)):
            for j in range(len(original_grid[0])):
                if original_grid[i][j] == 'T' and test_answer[i][j] != 'T':
                    return False
                if original_grid[i][j] != 'T' and test_answer[i][j] == 'T':
                    return False
        return True
    
    def _check_row_constraints(self, grid: List[List[str]], row_constraints: List[int]) -> bool:
        """检查行约束条件"""
        for i in range(len(grid)):
            tent_count = sum(1 for cell in grid[i] if cell == 'C')
            if tent_count != row_constraints[i]:
                return False
        return True
    
    def _check_col_constraints(self, grid: List[List[str]], col_constraints: List[int]) -> bool:
        """检查列约束条件"""
        for j in range(len(grid[0])):
            tent_count = sum(1 for i in range(len(grid)) if grid[i][j] == 'C')
            if tent_count != col_constraints[j]:
                return False
        return True
    
    def _check_tents_not_adjacent(self, grid: List[List[str]]) -> bool:
        """检查帐篷之间是否相邻（包括对角线）"""
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 'C':
                    # 检查周围8个方向是否有其他帐篷
                    for di in [-1, 0, 1]:
                        for dj in [-1, 0, 1]:
                            if di == 0 and dj == 0:
                                continue
                            ni, nj = i + di, j + dj
                            if 0 <= ni < n and 0 <= nj < m and grid[ni][nj] == 'C':
                                return False
        
        return True
    
    def _check_tent_tree_matching(self, grid: List[List[str]]) -> bool:
        """
        检查帐篷与树木的一一匹配关系:
        1. 每个帐篷必须与一棵树正交相邻
        2. 每棵树只能与一个帐篷匹配
        3. 每个帐篷只能与一棵树匹配
        4. 帐篷和树的数量必须相等
        """
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        
        tents = []
        trees = []
        for i in range(n):
            for j in range(m):
                if grid[i][j] == 'C':
                    tents.append((i, j))
                elif grid[i][j] == 'T':
                    trees.append((i, j))
        
        if len(tents) != len(trees):
            return False
        
        tent_to_trees = {}
        tree_to_tents = {}
        
        for tent_i, tent_j in tents:
            tent_to_trees[(tent_i, tent_j)] = []
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                tree_i, tree_j = tent_i + di, tent_j + dj
                if 0 <= tree_i < n and 0 <= tree_j < m and grid[tree_i][tree_j] == 'T':
                    tent_to_trees[(tent_i, tent_j)].append((tree_i, tree_j))
        
        for tree_i, tree_j in trees:
            tree_to_tents[(tree_i, tree_j)] = []
            for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                tent_i, tent_j = tree_i + di, tree_j + dj
                if 0 <= tent_i < n and 0 <= tent_j < m and grid[tent_i][tent_j] == 'C':
                    tree_to_tents[(tree_i, tree_j)].append((tent_i, tent_j))
        
        for tent in tents:
            if not tent_to_trees[tent]:
                return False
        
        tent_matched = {}
        tree_matched = {}
        
        def dfs(tent):
            for tree in tent_to_trees[tent]:
                if tree in visited:
                    continue
                visited.add(tree)
                
                if tree not in tree_matched or dfs(tree_matched[tree]):
                    tent_matched[tent] = tree
                    tree_matched[tree] = tent
                    return True
            return False
        
        for tent in tents:
            visited = set()
            if tent not in tent_matched:
                if not dfs(tent):
                    return False
        
        return len(tent_matched) == len(tents) and len(tree_matched) == len(trees)
    
    def extract_answer(self, test_solution: str):
        """从模型回答中提取解决方案"""
        grid_pattern = r'\[\s*\[.*?\]\s*\]'
        match = re.search(grid_pattern, test_solution, re.DOTALL)
        if match:
            try:
                grid_str = match.group(0)
                return ast.literal_eval(grid_str)
            except:
                pass
        return ""