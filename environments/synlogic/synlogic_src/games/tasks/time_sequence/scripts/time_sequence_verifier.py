import json
import numpy as np
from base.data import Data
from base.verifier import Verifier
import re

class TimeSequenceVerifier(Verifier):
    """
    验证器用于验证 time sequence 的答案是否正确
    """
    def verify(self, data: Data, test_solution: str):
        """
        验证模型的回答是否正确
        
        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的答案，格式为数字列表
        @return: 回答是否正确的布尔值
        """
        try:
            test_answer = self.extract_answer(test_solution)
            # 解析元数据
            metadata = data.metadata
            true_answers = metadata['records']['answers']
            
            # 解析模型给出的列表
            try:
                test_list = json.loads(test_answer.replace("，", ","))
            except:
                print(f"无法解析答案: {test_answer}")
                return False
            
            try:
                if test_list[0]!=true_answers['answer_maxLen']:
                    print(f"最长会议时间不正确。model:{test_answer} *** true:[{true_answers['answer_maxLen']}, {true_answers['answer_nums']}]")
                    return False
                if test_list[1]!=true_answers['answer_nums']:
                    print(f"可选会议数量不正确。model:{test_answer} *** true:[{true_answers['answer_maxLen']}, {true_answers['answer_nums']}]")
                    return False
            except:
                print(f"答案判断出错了！")
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
        @return: 提取答案列表
        """
        if not test_solution:
            return ""
        
        # 尝试提取列表
        matrix_pattern = r'\[.*?\]'
        matrix_matches = re.findall(matrix_pattern, test_solution, re.DOTALL)
        if matrix_matches:
            # 使用最后一个匹配的列表
            print(matrix_matches)
            return matrix_matches[-1].strip()
        
        # 如果失败，返回空字符串
        return ""