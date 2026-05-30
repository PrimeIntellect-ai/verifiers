from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
import re


class DyckLanguageVerifier(Verifier):
    """
    验证器用于检查Dyck Language游戏的答案是否正确
    """
    def verify(self, data: Data, test_answer: str) -> bool:
        """
        验证模型的回答是否正确
        
        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的回答字符串
        @return: 回答是否正确的布尔值
        """
        try:
            # 获取元数据中的完整序列
            full_sequence = data.metadata["full_sequence"]
            
            print(f"验证: 模型答案='{test_answer}', 完整序列='{full_sequence}'")
            
            # 从模型回答中提取答案
            extracted_answer = self.extract_answer(test_answer)
            
            # 检查答案是否完全匹配
            is_correct = (extracted_answer == full_sequence)
            
            if is_correct:
                print("验证结果: 正确")
            else:
                print("验证结果: 错误")
                
            return is_correct
            
        except Exception as e:
            print(f"验证时出错: {e}")
            return False 
        
    def extract_answer(self, test_solution: str) -> str:
        """
        从模型的回答中提取括号序列答案
        
        @param test_solution: 模型的完整回答
        @return: 提取的答案
        """
        if not test_solution:
            return ""
        
        # print(f"原始回答:\n{test_solution}")
            
        def clean_text(text: str) -> str:
            """清理文本，处理转义字符和空白字符"""
            # 移除所有空白字符（包括换行符、制表符等）
            text = ''.join(text.split())
            
            # 处理转义序列
            text = text.replace('\\n', '')
            text = text.replace('\\t', '')
            text = text.replace('\\r', '')
            text = text.replace('\\\\', '\\')
            
            # 如果文本被引号包围，且引号不是括号序列的一部分，则移除外层引号
            if len(text) >= 2:
                if text.startswith('"') and text.endswith('"'):
                    text = text[1:-1]
                elif text.startswith("'") and text.endswith("'"):
                    text = text[1:-1]
            
            return text
        
        return clean_text(test_solution)

if __name__ == "__main__":
    test_response = '''填写后的完整序列应为“([])({})([()])”。\n\n检查一下长度是否正确：\n\n原序列长度为11字符，补充3个字符，总长度14。\n\n这样，整个序列是合法的。\n</think>\n\n<answer>([])({})([()])</answer>'''
    metadata = {"trace_id": "38aeede4-d5d7-4863-91d2-df1fd99f491b", "full_sequence": "([])({})([()])", "question_sequence": "([])({})([(", "n_types": 3, "total_length": 14, "fill_length": 3, "nesting_depth": 0}
    test_data = Data(question="", answer="", metadata=metadata)
    test_verifier = DyckLanguageVerifier()
    extracted_answer = test_verifier.extract_answer(test_response)
    print(extracted_answer)
    print(test_verifier.verify(data=test_data, test_answer=test_response))