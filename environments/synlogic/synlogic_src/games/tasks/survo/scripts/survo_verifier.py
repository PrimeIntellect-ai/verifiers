import json
import numpy as np
from base.data import Data
from base.verifier import Verifier
import re

class SurvoVerifier(Verifier):
    """
    验证器用于检查Survo矩阵填充游戏的答案是否正确
    """
    def verify(self, data: Data, test_solution: str):
        """
        验证模型的回答是否正确
        
        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的矩阵答案字符串，格式为二维列表
        @return: 回答是否正确的布尔值
        """
        try:
            test_answer = self.extract_answer(test_solution)
            # 解析元数据
            metadata = data.metadata
            original_matrix = np.array(metadata["original_matrix"])
            candidate_numbers = metadata["candidate_numbers"]
            n = metadata["n"]
            
            # 解析模型给出的矩阵
            try:
                # 尝试直接解析JSON格式的矩阵
                test_matrix = json.loads(test_answer.replace("'", "\""))
                test_matrix = np.array(test_matrix)
            except:
                # 如果直接解析失败，尝试清理并手动解析
                test_answer = test_answer.strip()
                # 移除前后的括号和Python语法
                if test_answer.startswith("[") and test_answer.endswith("]"):
                    # 替换所有单引号为双引号以符合JSON格式
                    cleaned_answer = test_answer.replace("'", "\"")
                    try:
                        test_matrix = json.loads(cleaned_answer)
                        test_matrix = np.array(test_matrix)
                    except:
                        print(f"无法解析矩阵答案: {test_answer}")
                        return False
                else:
                    print(f"答案格式不正确: {test_answer}")
                    return False
            
            # 验证矩阵维度
            if test_matrix.shape != original_matrix.shape:
                print(f"矩阵维度不匹配: 期望 {original_matrix.shape}, 实际 {test_matrix.shape}")
                return False
            
            # 验证矩阵中的元素是否正确填充
            # 1. 检查原始矩阵中已填充的元素是否被保留
            for i in range(n):
                for j in range(n):
                    if original_matrix[i, j] != 0 and original_matrix[i, j] != test_matrix[i, j]:
                        print(f"原始矩阵中的元素被篡改: 位置 ({i}, {j}), 原值 {original_matrix[i, j]}, 新值 {test_matrix[i, j]}")
                        return False
            
            # 2. 收集填充的数字并验证它们是否与候选数字一致
            filled_numbers = []
            for i in range(n):
                for j in range(n):
                    if original_matrix[i, j] == 0:
                        filled_numbers.append(test_matrix[i, j])
            
            # 排序后比较是否一致
            sorted_filled = sorted(filled_numbers)
            sorted_candidates = sorted(candidate_numbers)
            
            if sorted_filled != sorted_candidates:
                print(f"填充的数字与候选数字不匹配: 填充 {sorted_filled}, 候选 {sorted_candidates}")
                return False
            
            # 3. 验证每行和每列的和是否正确
            # 检查每行的和
            for i in range(n-1):
                row_sum = sum(test_matrix[i, 0:n-1])
                expected_sum = test_matrix[i, n-1]
                if row_sum != expected_sum:
                    print(f"第 {i+1} 行的和不正确: 实际和 {row_sum}, 期望和 {expected_sum}")
                    return False
            
            # 检查每列的和
            for j in range(n-1):
                col_sum = sum(test_matrix[0:n-1, j])
                expected_sum = test_matrix[n-1, j]
                if col_sum != expected_sum:
                    print(f"第 {j+1} 列的和不正确: 实际和 {col_sum}, 期望和 {expected_sum}")
                    return False
            
            # 所有检查都通过
            print("验证结果: 正确")
            return True
            
        except Exception as e:
            print(f"验证时出错: {e}")
            return False 
        
    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取答案（矩阵）
        
        @param test_solution: 模型的完整回答
        @return: 提取的矩阵答案字符串
        """
        if not test_solution:
            return ""
        
        # 尝试提取Python代码块中的矩阵
        code_block_pattern = r'```python\s*([\s\S]*?)\s*```'
        code_matches = re.findall(code_block_pattern, test_solution)
        
        if code_matches:
            # 使用最后一个Python代码块
            matrix_str = code_matches[-1].strip()
            return matrix_str
        
        # 如果没有找到Python代码块，尝试提取任何代码块
        general_code_block = r'```([\s\S]*?)```'
        general_matches = re.findall(general_code_block, test_solution)
        
        if general_matches:
            # 使用最后一个代码块
            matrix_str = general_matches[-1].strip()
            return matrix_str
        
        # 如果没有找到代码块，尝试提取可能的矩阵表示
        matrix_pattern = r'\[\s*\[.*?\]\s*\]'
        matrix_matches = re.findall(matrix_pattern, test_solution, re.DOTALL)
        
        if matrix_matches:
            # 使用最后一个匹配的矩阵
            return matrix_matches[-1].strip()
        
        # 如果所有方法都失败，返回空字符串
        return ""