from abc import ABC, abstractmethod
from base.data import Data

class Verifier(ABC):
    """
    Base class for verifier
    """
    def __init__(self):
        pass
    
    @abstractmethod
    def verify(self, data: Data, test_answer: str):
        """
        Verify whether the test answer is consistent with the gold answer
        @param data: Data
        @param test_answer: str
        @return: bool
        """
        raise NotImplementedError("Verifier.verify() is not implemented")

    @abstractmethod
    def extract_answer(self, test_solution: str):
        """
        Extract the answer from the test solution
        @param test_solution: str
        @return: str
        """
        raise NotImplementedError("Verifier.extract_answer() is not implemented")

import re

THOUGHT_DELIMITER_START = "<think>"
THOUGHT_DELIMITER_END = "</think>"

def _extract_answer(text):
    # 定义正则表达式模式，匹配 <answer> 和 </answer> 之间的内容
    pattern = r'<answer>(.*?)</answer>'
    
    # 使用 re.search 查找第一个匹配项
    match = re.search(pattern, text, re.DOTALL)
    
    # 如果找到匹配项，返回匹配的内容
    if match:
        return match.group(1).strip()
    else:
        return None
    
def _extract_solution_with_thought(solution_str):
    
    model_output = solution_str
    
    if THOUGHT_DELIMITER_END in solution_str:
        model_output = solution_str.split(THOUGHT_DELIMITER_END)[1]
    
    predict_answer = _extract_answer(model_output)
    
    
    if predict_answer is not None:
        return predict_answer
    else:
        return ""


class ExactMatchVerifier(Verifier):
    """
    Verifier for Exact Match
    """
    def verify(self, data: Data, test_solution: str):
        try:
            test_answer = self.extract_answer(test_solution)
            ground_truth = data.answer
            correct = test_answer == ground_truth
            if correct:
                acc_score = 1.0
            else:
                acc_score = 0

            return acc_score
        except:
            return False
    
    def extract_answer(self, test_solution: str):
        return _extract_solution_with_thought(solution_str=test_solution)