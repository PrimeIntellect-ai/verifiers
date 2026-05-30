import re
from base.data import Data
from base.verifier import Verifier

class WordSortingVerifier(Verifier):
    """
    验证器用于单词排序游戏的答案是否正确
    """
    def str2list(self, answer_str):
        # 替换中文逗号为英文逗号，并删除所有空格
        answer_str = answer_str.replace("，", ",").replace(" ", "")
        return [w.strip() for w in answer_str.split(",")]

    def verify(self, data: Data, test_answer: str):
        try:
            ground_truth = self.str2list(data.answer)
            parsed_answer = self.str2list(self.extract_answer(test_answer))
            
            if parsed_answer is None:
                return False
            return parsed_answer == ground_truth

        except Exception as e:
            print(f"解析答案错误:{e}")
            return False
    
    def extract_answer(self, answer_str):
        # 先找到最后一个\Box{的位置
        last_box_index = answer_str.rfind("\\boxed{")
        
        if last_box_index == -1:
            return None
            
        # 从最后一个\Box{开始截取字符串
        last_box_substring = answer_str[last_box_index:]
        
        # 在截取的子字符串中进行正则匹配
        box_pattern = r'\\boxed\{([^}]*)\}'
        match = re.search(box_pattern, last_box_substring)
        
        if match:
            return match.group(1).strip()
        return None