import re
from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
import math_verify 


class OperationVerifier(Verifier):
    """
    验证器用于物品计数游戏的答案是否正确
    """
    def verify(self, data: Data, test_answer: str):
        try:
            ground_truth = math_verify.parse(data.answer)
            parsed_answer = math_verify.parse(test_answer)
            
            if parsed_answer is None:
                return False
            return math_verify.verify(parsed_answer, ground_truth)
        except Exception as e:
            print(f"解析答案错误:{e}")
            return False
    
    def extract_answer(self, answer_str):
        # 先找到最后一个\boxed{的位置
        last_box_index = answer_str.rfind("\\boxed{")
        
        if last_box_index == -1:
            return None
        
        # 从\boxed{开始截取到正确的闭合位置，处理嵌套括号
        start_index = last_box_index + len("\\boxed{")
        bracket_stack = 1  # 已经遇到了一个左括号
        end_index = start_index
        
        while end_index < len(answer_str) and bracket_stack > 0:
            if answer_str[end_index] == '{':
                bracket_stack += 1
            elif answer_str[end_index] == '}':
                bracket_stack -= 1
            end_index += 1
        
        if bracket_stack != 0:  # 括号不匹配
            return None
        
        # 提取\boxed{}内的内容
        latex_content = answer_str[start_index:end_index-1].strip()
        return latex_content