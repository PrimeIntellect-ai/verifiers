import re
from base.data import Data
from base.verifier import Verifier

class WordSortingMistakeVerifier(Verifier):
    """
    验证器用于word sorting mistake的答案是否正确
    """
    def verify(self, data: Data, test_answer: str):
        try:
            ground_truth = data.answer if data.answer is not None else "No"
            parsed_answer = self.extract_answer(test_answer)
            
            if parsed_answer is None:
                return False
            
            if parsed_answer.isdigit():
                try:
                    return int(parsed_answer) == int(ground_truth)
                except Exception as e:
                    return False
            else:
                return parsed_answer.lower() == ground_truth.lower()
        except Exception as e:
            print(f"解析答案错误:{e}")
            return False
    
    def extract_answer(self, answer_str):
        # 先找到最后一个\boxed{的位置
        last_box_index = answer_str.rfind("\\boxed{")
        
        if last_box_index == -1:
            return None
            
        # 从最后一个\boxed{开始截取字符串
        last_box_substring = answer_str[last_box_index:]
        
        # 在截取的子字符串中进行正则匹配
        box_pattern = r'\\boxed\{([^}]*)\}'
        match = re.search(box_pattern, last_box_substring)
        
        if match:
            return match.group(1).strip()
        return None
        