import json
from typing import List, Dict, Tuple
from base.verifier import Verifier
from base.data import Data
import re

class ArrowMazeVerifier(Verifier):
    """
    箭头迷宫游戏验证器
    
    验证条件:
    1. 判断answer grid的大小是否和question grid一致
    2. 判断answer grid中数字格子是否和question grid中数字格子一致
    3. 判断question grid空格（"X"）在answer grid中是否被箭头填满
    4. 判断箭头符号是否合法：
       上（↑）、下（↓）、左（←）、右（→）或对角线方向（↖、↗、↘、↙）
    5. 判断answer grid中非空格（"X"）和非数字的部分，即预填的箭头，是否和question grid一致
    6. 迷宫有个隐藏的条件是所有箭头都能被射线箭头串覆盖到
    7. 每个数字起点发出的射线箭头串总长度等于该数字
    """
    
    # 定义合法的箭头符号
    VALID_ARROWS = {"↑", "↓", "←", "→", "↖", "↗", "↘", "↙"}
    
    # 定义箭头符号和其对应的方向
    ARROWS_DIRECTIONS = {
        "↑": (-1, 0),   # 上
        "↓": (1, 0),    # 下
        "←": (0, -1),   # 左
        "→": (0, 1),    # 右
        "↖": (-1, -1),  # 左上
        "↗": (-1, 1),   # 右上
        "↘": (1, 1),    # 右下
        "↙": (1, -1)    # 左下
    }
    
    def verify(self, data: Data, test_solution_str: str) -> bool:
        """
        验证箭头迷宫的答案是否正确
        
        @param data: 游戏数据
        @param test_solution_str: 测试答案字符串 (JSON格式的二维数组)
        @return: 答案是否正确
        """
        test_answer_str = self.extract_answer(test_solution_str)
        if not test_answer_str:
            print("答案为空，验证失败")
            return False
        
        try:
            # 解析测试答案
            test_answer = json.loads(test_answer_str)
            
            # 获取原始迷宫
            question_grid = data.metadata["maze"]
            
            # 检查答案是否符合要求
            if not self._verify_grid_size(test_answer, question_grid):
                print("答案网格大小与题目不匹配")
                return False
                
            if not self._verify_number_positions(test_answer, question_grid):
                print("答案中数字位置或值与题目不匹配")
                return False
                
            if not self._verify_all_blanks_filled(test_answer, question_grid):
                print("答案中有空格未被填满")
                return False
                
            if not self._verify_arrow_symbols(test_answer):
                print("答案中包含非法箭头符号")
                return False
                
            if not self._verify_prefilled_arrows(test_answer, question_grid):
                print("答案中预填箭头与题目不一致")
                return False
                
            if not self._verify_arrow_rays(test_answer):
                print("答案中存在未被射线覆盖的箭头")
                return False
                
            if not self._verify_number_rays(test_answer):
                print("答案中数字的射线箭头串总数不符合要求")
                return False
            
            # 所有验证都通过
            print("验证通过！")
            return True
            
        except Exception as e:
            print(f"验证过程中出错: {e}")
            return False
    
    def _verify_grid_size(self, test_answer: List[List[str]], question_grid: List[List[str]]) -> bool:
        """
        验证答案网格大小是否与题目一致
        
        @param test_answer: 测试答案网格
        @param question_grid: 题目网格
        @return: 网格大小是否一致
        """
        if len(test_answer) != len(question_grid):
            return False
            
        for i in range(len(test_answer)):
            if len(test_answer[i]) != len(question_grid[i]):
                return False
                
        return True
    
    def _verify_number_positions(self, test_answer: List[List[str]], question_grid: List[List[str]]) -> bool:
        """
        验证答案中数字位置和值是否与题目一致
        
        @param test_answer: 测试答案网格
        @param question_grid: 题目网格
        @return: 数字位置和值是否一致
        """
        for i in range(len(question_grid)):
            for j in range(len(question_grid[i])):
                if question_grid[i][j].isdigit():
                    if test_answer[i][j] != question_grid[i][j]:
                        return False
        return True
    
    def _verify_all_blanks_filled(self, test_answer: List[List[str]], question_grid: List[List[str]]) -> bool:
        """
        验证所有空格是否都被填满
        
        @param test_answer: 测试答案网格
        @param question_grid: 题目网格
        @return: 所有空格是否被填满
        """
        for i in range(len(question_grid)):
            for j in range(len(question_grid[i])):
                if question_grid[i][j] == "X" and test_answer[i][j] == "X":
                    return False
        return True
    
    def _verify_arrow_symbols(self, test_answer: List[List[str]]) -> bool:
        """
        验证箭头符号是否合法
        
        @param test_answer: 测试答案网格
        @return: 箭头符号是否合法
        """
        for i in range(len(test_answer)):
            for j in range(len(test_answer[i])):
                cell = test_answer[i][j]
                if not cell.isdigit() and cell != "X" and cell not in self.VALID_ARROWS:
                    return False
        return True
    
    def _verify_prefilled_arrows(self, test_answer: List[List[str]], question_grid: List[List[str]]) -> bool:
        """
        验证预填的箭头是否与题目一致
        
        @param test_answer: 测试答案网格
        @param question_grid: 题目网格
        @return: 预填箭头是否一致
        """
        for i in range(len(question_grid)):
            for j in range(len(question_grid[i])):
                cell = question_grid[i][j]
                if not cell.isdigit() and cell != "X":
                    if test_answer[i][j] != cell:
                        return False
        return True
    
    def _verify_arrow_rays(self, test_answer: List[List[str]]) -> bool:
        """
        验证所有箭头是否都能被射线箭头串覆盖到
        
        @param test_answer: 测试答案网格
        @return: 所有箭头是否都能被射线覆盖
        """
        n = len(test_answer)
        m = len(test_answer[0]) if n > 0 else 0
        
        # 创建覆盖标记数组
        covered = [[False for _ in range(m)] for _ in range(n)]
        
        # 标记数字位置为已覆盖
        for i in range(n):
            for j in range(m):
                if test_answer[i][j].isdigit():
                    covered[i][j] = True
        
        # 从每个数字出发，沿各个方向延伸射线，标记覆盖到的箭头
        for i in range(n):
            for j in range(m):
                if test_answer[i][j].isdigit():
                    # 检查所有方向
                    for arrow_symbol, (di, dj) in self.ARROWS_DIRECTIONS.items():
                        ni, nj = i + di, j + dj
                        # 沿该方向延伸，直到边界或非匹配箭头
                        while 0 <= ni < n and 0 <= nj < m and test_answer[ni][nj] == arrow_symbol:
                            covered[ni][nj] = True
                            ni += di
                            nj += dj
        
        # 检查所有箭头是否都被覆盖
        for i in range(n):
            for j in range(m):
                if test_answer[i][j] in self.VALID_ARROWS and not covered[i][j]:
                    return False
        
        return True
    
    def _verify_number_rays(self, test_answer: List[List[str]]) -> bool:
        """
        验证每个数字起点发出的射线箭头串总长度是否等于该数字
        
        @param test_answer: 测试答案网格
        @return: 每个数字的射线箭头串是否符合要求
        """
        n = len(test_answer)
        m = len(test_answer[0]) if n > 0 else 0
        
        for i in range(n):
            for j in range(m):
                if test_answer[i][j].isdigit():
                    number = int(test_answer[i][j])
                    arrow_count = self._count_arrow_rays(test_answer, i, j)
                    if arrow_count != number:
                        return False
        
        return True
    
    def _count_arrow_rays(self, grid: List[List[str]], i: int, j: int) -> int:
        """
        计算从数字出发的所有射线箭头串中箭头总数
        
        @param grid: 网格
        @param i: 数字行索引
        @param j: 数字列索引
        @return: 箭头总数
        """
        n = len(grid)
        m = len(grid[0]) if n > 0 else 0
        count = 0
        
        # 检查所有方向
        for arrow_symbol, (di, dj) in self.ARROWS_DIRECTIONS.items():
            ni, nj = i + di, j + dj
            ray_length = 0
            
            # 沿该方向计数连续的相同箭头
            while 0 <= ni < n and 0 <= nj < m and grid[ni][nj] == arrow_symbol:
                ray_length += 1
                ni += di
                nj += dj
            
            count += ray_length
        
        return count 
    
    def extract_answer(self, test_solution: str) -> str:
            """
            从模型的回答中提取答案
            
            @param test_solution: 模型的完整回答
            @return: 提取的答案 (JSON格式的二维数组)
            """
            if not test_solution:
                return ""
            # 尝试匹配Python代码块
            import re
            code_block_patterns = [
                r'```python\s*\n(.*?\[.*?\].*?)\n```',  # 标准Python代码块
                r'```\s*\n(.*?\[.*?\].*?)\n```',       # 无语言标记的代码块
                r'```(.*?\[.*?\].*?)```'               # 无换行的代码块
            ]
            
            for pattern in code_block_patterns:
                matches = re.findall(pattern, test_solution, re.DOTALL)
                if matches:
                    # 获取最后一个匹配项
                    code_block = matches[-1].strip()
                    try:
                        # 尝试解析为Python列表
                        grid = eval(code_block)
                        # 验证格式是否为二维数组
                        if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                            return json.dumps(grid)
                    except Exception as e:
                        print(f"解析代码块失败: {e}")
                        continue
            
            # 如果没有找到有效的代码块，尝试直接寻找列表
            list_pattern = r'\[\s*\[.*?\]\s*\]'
            matches = re.findall(list_pattern, test_solution, re.DOTALL)
            if matches:
                try:
                    # 尝试解析为Python列表
                    grid = eval(matches[-1])
                    # 验证格式是否为二维数组
                    if isinstance(grid, list) and all(isinstance(row, list) for row in grid):
                        return json.dumps(grid)
                except Exception as e:
                    print(f"解析列表失败: {e}")
            
            # 如果上述方法都失败，返回空字符串
            return ""