from base.data import Data
from base.verifier import Verifier
import re
import ast

class SudokuVerifier(Verifier):
    """
    验证器用于检查数独游戏的答案是否正确
    """
    def verify(self, data: Data, test_solution: str):
        """
        验证模型提供的数独解答是否正确
        
        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的数独解答字符串，应该是一个9x9的二维数组
        @return: 回答是否正确的布尔值
        """
        try:
            test_answer = self.extract_answer(test_solution)
            # 将字符串转换为Python元组（二维数组）
            if not test_answer or test_answer == "":
                print("验证失败：答案为空")
                return False
            
            # 尝试将答案字符串转换为二维数组
            try:
                # 将字符串转换为Python数据结构
                sudoku_solution = ast.literal_eval(test_answer)
                
                # 检查解答是否为9x9的二维数组
                if (not isinstance(sudoku_solution, tuple) and not isinstance(sudoku_solution, list)) or len(sudoku_solution) != 9:
                    print(f"验证失败：解答格式不正确，应为9x9数组，实际为: {type(sudoku_solution)}, 长度: {len(sudoku_solution)}")
                    return False
                
                for row in sudoku_solution:
                    if (not isinstance(row, tuple) and not isinstance(row, list)) or len(row) != 9:
                        print(f"验证失败：解答格式不正确，行应为长度9的数组，实际为: {type(row)}, 长度: {len(row)}")
                        return False
                    
                    for num in row:
                        if not isinstance(num, int) or num < 1 or num > 9:
                            print(f"验证失败：解答包含非法数字 {num}，应为1-9的整数")
                            return False
            except (SyntaxError, ValueError) as e:
                print(f"验证失败：无法解析答案 '{test_answer}', 错误: {e}")
                return False
            
            # 从元数据中获取原始数独题目
            original_sudoku = data.metadata.get("original_sudoku", [])
            if not original_sudoku:
                print("验证失败：元数据中缺少原始数独题目")
                return False
            
            # 1. 验证解答是否符合数独规则
            if not self._is_valid_sudoku(sudoku_solution):
                print("验证失败：解答不符合数独规则")
                return False
            
            # 2. 验证解答是否与原始数独题目一致（已填数字部分）
            if not self._is_consistent_with_original(original_sudoku, sudoku_solution):
                print("验证失败：解答与原始数独不一致")
                return False
            
            print("验证成功：解答正确")
            return True
            
        except Exception as e:
            print(f"验证时出错: {e}")
            return False
    
    def _is_valid_sudoku(self, sudoku):
        """
        验证数独解答是否符合规则：每行、每列、每个3x3子网格包含1-9的数字且不重复
        
        @param sudoku: 数独解答，9x9的二维数组
        @return: 是否符合数独规则
        """
        # 检查每一行
        for row in sudoku:
            if set(row) != set(range(1, 10)):
                return False
        
        # 检查每一列
        for col in range(9):
            column = [sudoku[row][col] for row in range(9)]
            if set(column) != set(range(1, 10)):
                return False
        
        # 检查每个3x3子网格
        for box_row in range(0, 9, 3):
            for box_col in range(0, 9, 3):
                box = []
                for r in range(box_row, box_row + 3):
                    for c in range(box_col, box_col + 3):
                        box.append(sudoku[r][c])
                if set(box) != set(range(1, 10)):
                    return False
        
        return True
    
    def _is_consistent_with_original(self, original_sudoku, solution_sudoku):
        """
        验证解答是否与原始数独题目一致（已填数字部分）
        
        @param original_sudoku: 原始数独题目，9x9的二维数组，0或'X'表示空格
        @param solution_sudoku: 数独解答，9x9的二维数组
        @return: 是否与原始数独一致
        """
        for i in range(9):
            for j in range(9):
                original_value = original_sudoku[i][j]
                # 如果原始数独中的位置不是空的（不是0或'X'），则检查解答是否与之一致
                if original_value not in [0, 'X', 'x']:
                    if solution_sudoku[i][j] != int(original_value):
                        print(f"位置 ({i},{j}) 不一致: 原始值 {original_value}, 解答值 {solution_sudoku[i][j]}")
                        return False
        
        return True 
    
    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取数独解答
        
        @param test_solution: 模型的完整回答
        @return: 提取的答案（元组形式的字符串）
        """
        if not test_solution:
            return ""
        
        # 提取Python代码块中的内容
        code_block_pattern = r"```python\s*([\s\S]*?)\s*```"
        matches = re.findall(code_block_pattern, test_solution)
        
        if matches:
            # 取最后一个Python代码块
            python_code = matches[-1].strip()
            return python_code
        
        # 如果没有找到Python代码块，尝试找到任何类似于元组的结构
        tuple_pattern = r"\(\s*\(\s*\d+\s*,.*?\)\s*\)"
        matches = re.findall(tuple_pattern, test_solution, re.DOTALL)
        
        if matches:
            return matches[-1].strip()
        
        return ""