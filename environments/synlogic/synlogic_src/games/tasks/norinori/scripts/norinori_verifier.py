from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
import re
from collections import defaultdict

class NorinoriVerifier(Verifier):
    """
    Norinori 游戏的验证器
    检查提交的答案是否符合 Norinori 游戏规则
    """
    
    def __init__(self):
        super().__init__()

    def verify(self, data: Data, test_solution: str):
        """
        验证 Norinori 游戏的答案
        
        参数:
        data -- 游戏数据，包含区域网格等信息
        test_solution -- 用户提交的答案，应为多米诺坐标列表
        
        返回:
        bool -- 答案是否正确
        """
        try:
            # 从游戏数据中获取区域网格
            region_grid = data.metadata["region_grid"]
            n = len(region_grid)
            
            # 解析答案
            dominoes = self._parse_answer(test_solution)
            if dominoes is None:
                return False
            
            # 检查多米诺形状
            if not self._check_domino_shapes(dominoes):
                return False
            
            # 创建覆盖网格
            covered = [[False for _ in range(n)] for _ in range(n)]
            for domino in dominoes:
                for i, j in domino:
                    # 转换为0-indexed
                    i -= 1
                    j -= 1
                    if i < 0 or i >= n or j < 0 or j >= n:
                        return False  # 坐标超出范围
                    if covered[i][j]:
                        return False  # 格子被多次覆盖
                    covered[i][j] = True
            
            # 检查多米诺之间是否相邻
            if not self._check_domino_adjacency(dominoes, n):
                return False
            
            # 检查每个区域是否恰好有两个格子被覆盖
            region_coverage = defaultdict(int)
            for i in range(n):
                for j in range(n):
                    if covered[i][j] and region_grid[i][j] != "X":
                        region_coverage[region_grid[i][j]] += 1
            
            for region, count in region_coverage.items():
                if count != 2:
                    return False
        
            # 检查所有阴影格子是否被覆盖
            for i in range(n):
                for j in range(n):
                    if region_grid[i][j] == "X" and not covered[i][j]:
                        return False
            
            return True
        except Exception as e:
            print(f"验证过程中出错: {e}")
            return False
        
    def _parse_answer(self, test_solution: str):
        """
        解析答案字符串，提取多米诺坐标
        
        参数:
        test_solution -- 答案字符串
        
        返回:
        list -- 多米诺坐标列表，如果格式不正确则返回None
        """
        try:
            # 使用正则表达式提取坐标对
            pattern = r'\[\((\d+),\s*(\d+)\),\s*\((\d+),\s*(\d+)\)\]'
            matches = re.findall(pattern, test_solution)
            
            if not matches:
                # 尝试另一种可能的格式
                pattern = r'\(\s*(\d+)\s*,\s*(\d+)\s*\)\s*,\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)'
                matches = re.findall(pattern, test_solution)
            
            dominoes = []
            for match in matches:
                i1, j1, i2, j2 = map(int, match)
                dominoes.append([(i1, j1), (i2, j2)])
            
            return dominoes
        except Exception as e:
            print(f"解析答案时出错: {e}")
            return None
    
    def _check_domino_shapes(self, dominoes):
        """
        检查所有多米诺是否都是1×2或2×1的形状
        
        参数:
        dominoes -- 多米诺坐标列表
        
        返回:
        bool -- 是否所有多米诺都符合形状要求
        """
        for domino in dominoes:
            if len(domino) != 2:
                return False
            
            (i1, j1), (i2, j2) = domino
            
            # 检查是否为1×2或2×1
            if not ((i1 == i2 and abs(j1 - j2) == 1) or 
                    (j1 == j2 and abs(i1 - i2) == 1)):
                return False
        
        return True
    
    def _check_domino_adjacency(self, dominoes, n):
        """
        检查多米诺之间是否相邻
        
        参数:
        dominoes -- 多米诺坐标列表
        n -- 网格大小
        
        返回:
        bool -- 是否所有多米诺都不相邻
        """
        # 创建一个网格来标记每个多米诺的位置
        grid = [[-1 for _ in range(n+2)] for _ in range(n+2)]  # 加2是为了处理边界
        
        for idx, domino in enumerate(dominoes):
            for i, j in domino:
                # 转换为0-indexed并考虑边界
                grid[i][j] = idx
        
        # 检查每个多米诺是否与其他多米诺相邻
        for idx, domino in enumerate(dominoes):
            for i, j in domino:
                for di, dj in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    ni, nj = i + di, j + dj
                    if 1 <= ni <= n and 1 <= nj <= n:  # 检查是否在网格内
                        if grid[ni][nj] != -1 and grid[ni][nj] != idx:
                            return False  # 发现相邻的多米诺
        
        return True
    
    def extract_answer(self, test_solution: str, strict=False):
        """
        从回答中提取答案
        
        参数:
        test_solution -- 用户的回答
        strict -- 是否严格模式
        
        返回:
        str -- 提取的答案
        """
        # 尝试找到答案部分
        answer_patterns = [
            r'\[\s*\[\s*\(\s*\d+\s*,\s*\d+\s*\)\s*,\s*\(\s*\d+\s*,\s*\d+\s*\)\s*\]',  # 寻找格式如 [[(1,2), (1,3)], ...] 的答案
            r'答案是\s*(.*?)\s*$',  # 中文格式
            r'answer is\s*(.*?)\s*$',  # 英文格式
            r'solution is\s*(.*?)\s*$'  # 另一种英文格式
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, test_solution, re.IGNORECASE | re.DOTALL)
            if matches:
                # 返回最后一个匹配项，通常是最终答案
                return matches[-1]
        
        # 如果没有找到明确的答案格式，返回整个解答
        return test_solution