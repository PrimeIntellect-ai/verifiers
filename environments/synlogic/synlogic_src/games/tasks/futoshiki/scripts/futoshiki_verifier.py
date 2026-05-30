import numpy as np
from typing import List, Tuple, Dict, Optional
import re
from base.data import Data
from base.verifier import Verifier

class FutoshikiVerifier(Verifier):
    """Verifier for Futoshiki"""
    def __init__(self):
        """Initialize the verifier."""
        pass
    
    def verify(self, data: Data, test_solution: str) -> bool:
        """
        验证模型提供的回答是否正确：
        1. 数独规则：每行每列1-N各出现一次
        2. 预填充数字：模型保留了题目中的预填充数字
        3. 不等式约束：所有不等式约束都被满足
        
        注意：不需要和生成数据中的答案完全一致，只要满足上述三个条件即为正确。
        
        Args:
            data: 包含题目和元数据的Data对象
            test_solution: 模型给出的回答字符串
            
        Returns:
            bool: 回答是否正确
        """
        # 从test_solution中提取答案
        answer = self.extract_answer(test_solution)
        if answer is None:
            print("无法提取答案")
            return False
            
        # 获取网格大小
        grid_size = data.metadata["grid_size"]
        
        # 检查答案格式
        if not self.check_answer_format(answer, grid_size):
            print(f"答案格式不正确: 期望 {grid_size}x{grid_size} 矩阵，实际为 {answer.shape}")
            return False
            
        # 检查每行和每列是否包含1到N的所有数字（数独规则）
        if not self.check_rows_and_columns(answer, grid_size):
            print("答案不满足数独规则（每行每列包含1到N的所有数字）")
            return False
            
        # 检查预填充数字是否保留
        prefilled_coords = data.metadata["prefilled_coords"]
        
        # 从数据中提取原始网格并解析预填充数字
        question = data.question
        prefilled_values = []
        grid_lines = []
        
        # 从题目中解析原始网格
        in_grid = False
        for line in question.split('\n'):
            if 'Current puzzle:' in line:
                in_grid = True
                continue
            elif in_grid and line.strip() and 'X' in line:
                grid_lines.append(line.strip())
            elif in_grid and grid_lines and line.strip() and 'X' not in line:
                in_grid = False
        
        # 提取预填充的值
        for (row, col) in prefilled_coords:
            # 确保网格行存在
            if row < len(grid_lines):
                grid_line = grid_lines[row].split()
                if col < len(grid_line) and grid_line[col] != 'X':
                    try:
                        prefilled_values.append(int(grid_line[col]))
                    except ValueError:
                        prefilled_values.append(None)
                else:
                    prefilled_values.append(None)
            else:
                prefilled_values.append(None)
        
        if not self.check_prefilled_numbers(answer, prefilled_coords, prefilled_values):
            print("答案未保留预填充数字")
            return False
            
        # 检查不等式约束
        constraints = data.metadata["constraints"]
        if not self.check_inequality_constraints(answer, constraints):
            print("答案不满足不等式约束")
            return False
            
        return True
    
    def extract_answer(self, test_solution: str) -> Optional[np.ndarray]:
        """Extract the answer from the test solution string."""
        try:
            # 移除所有空白字符
            test_solution = test_solution.strip()
            
            # 尝试从GPT响应中提取最后一个[[...]]格式的内容
            answer_matches = re.findall(r'\[\[([^\]]+)\]\]', test_solution)
            if answer_matches:
                # 获取最后一个匹配项，通常是最终答案
                answer_str = answer_matches[-1]
                
                # 分割行
                rows = answer_str.split(',')
                
                # 解析每一行
                grid = []
                for row in rows:
                    # 分割数字
                    numbers = re.findall(r'\d+', row)
                    # 转换为整数
                    row_numbers = [int(num) for num in numbers]
                    grid.append(row_numbers)
                
                # 检查是否是有效的网格
                if len(grid) > 0 and all(len(row) == len(grid) for row in grid):
                    return np.array(grid)
            
            # 如果上面的方法失败，尝试直接解析 [[...]] 格式
            if test_solution.startswith('[[') and test_solution.endswith(']]'):
                # 移除[[和]]
                test_solution = test_solution[2:-2]
                
                # 分割行
                rows = test_solution.split(',')
                
                # 解析每一行
                grid = []
                for row in rows:
                    # 分割数字
                    numbers = row.strip().split()
                    # 转换为整数
                    row_numbers = [int(num) for num in numbers]
                    grid.append(row_numbers)
                
                return np.array(grid)
            
            print(f"无法找到有效的答案格式，原始回答: {test_solution[:100]}...")
            return None
            
        except Exception as e:
            print(f"解析答案时出错: {e}，原始回答: {test_solution[:100]}...")
            return None
    
    def check_answer_format(self, answer: np.ndarray, grid_size: int) -> bool:
        """Check if the answer has the correct format."""
        # 检查形状
        if answer.shape != (grid_size, grid_size):
            return False
            
        # 检查数据类型
        if answer.dtype != np.int64:
            return False
            
        return True
    
    def check_rows_and_columns(self, answer: np.ndarray, grid_size: int) -> bool:
        """Check if each row and column contains all numbers from 1 to n."""
        # 检查每一行
        for i, row in enumerate(answer):
            if not self.check_sequence(row, grid_size):
                print(f"第 {i+1} 行不满足数独规则: {row}")
                return False
                
        # 检查每一列
        for i, col in enumerate(answer.T):
            if not self.check_sequence(col, grid_size):
                print(f"第 {i+1} 列不满足数独规则: {col}")
                return False
                
        return True
    
    def check_sequence(self, sequence: np.ndarray, grid_size: int) -> bool:
        """Check if a sequence contains all numbers from 1 to n."""
        # 创建1到n的集合
        required = set(range(1, grid_size + 1))
        # 创建序列中的数字集合
        actual = set(sequence)
        # 检查是否相等
        return required == actual
    
    def check_prefilled_numbers(self, answer: np.ndarray, prefilled_coords: List[Tuple[int, int]], prefilled_values: List[int]) -> bool:
        """Check if the prefilled numbers are preserved in the answer."""
        for (row, col), value in zip(prefilled_coords, prefilled_values):
            if value is not None and answer[row, col] != value:
                print(f"预填充位置 ({row+1},{col+1}) 的值被改变: 期望 {value}, 实际为 {answer[row, col]}")
                return False
        return True
    
    def check_inequality_constraints(self, answer: np.ndarray, constraints: List[Tuple[Tuple[int, int], Tuple[int, int], str]]) -> bool:
        """Check if all inequality constraints are satisfied."""
        for (coord1, coord2, sign) in constraints:
            row1, col1 = coord1
            row2, col2 = coord2
            num1 = answer[row1, col1]
            num2 = answer[row2, col2]
            
            if sign == '>':
                if num1 <= num2:
                    print(f"不等式约束不满足: ({row1+1},{col1+1}) > ({row2+1},{col2+1}), 实际值: {num1} <= {num2}")
                    return False
            elif sign == '<':
                if num1 >= num2:
                    print(f"不等式约束不满足: ({row1+1},{col1+1}) < ({row2+1},{col2+1}), 实际值: {num1} >= {num2}")
                    return False
                    
        return True 