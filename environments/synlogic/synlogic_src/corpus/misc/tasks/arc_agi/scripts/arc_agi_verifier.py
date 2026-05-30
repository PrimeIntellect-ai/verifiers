from base.data import Data
from base.verifier import Verifier
import re
import json


def _preprocess_answer(answer: str) -> str:
    """预处理答案文本"""
    if not answer:
        return None
        
    processed = answer.strip().lower()
    processed = processed.split("```json")[-1]
    processed = processed.split("```")[0].strip()
    try:
        processed = json.loads(processed)
    except:
        try:
            processed = eval(processed)
        except:
            return None

    return processed

class ArcAGIVerifier(Verifier):
    """ArcAGI任务的验证器"""
    
    def verify(self, data: Data, test_solution: str) -> float:
        try:
            # 提取和预处理模型答案
            test_answer = self.extract_answer(test_solution)
            
            # 获取并预处理参考答案
            ground_truth = data.answer
            
            # 进行模糊匹配
            correct = test_answer == ground_truth
            
            return 1.0 if correct else 0.0
        except:
            return 0.0
    
    def extract_answer(self, test_solution: str) -> str:
        """从完整答案中提取最终答案"""
        # if THOUGHT_DELIMITER_END in test_solution:
        #     test_solution = test_solution.split(THOUGHT_DELIMITER_END)[1]
            
        return _preprocess_answer(test_solution)

if __name__ == '__main__':
    # 测试用例
    test_cases = [
        ("<answer>```json\n[[7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 0, 0, 0, 0, 7, 7, 0], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 0, 0, 0, 0, 7, 7, 0], [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 7, 0, 7, 7, 0, 0, 0, 0]]\n```</answer>", [[7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 0, 0, 0, 0, 7, 7, 0], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 0, 7, 0, 0, 0, 7, 0, 7], [7, 7, 0, 0, 0, 0, 7, 7, 0], [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 0, 7, 7, 0, 7, 0, 0, 0], [7, 7, 0, 7, 7, 0, 0, 0, 0]]),
    ]
    
    verifier = ArcAGIVerifier()
    for test_input, reference in test_cases:
        result = verifier.extract_answer(test_input)
        print(f"Input: {test_input}")
        print(f"Reference: {reference}")
        print(f"Extracted: {result}")
        print(f"Match: {result == reference}\n")