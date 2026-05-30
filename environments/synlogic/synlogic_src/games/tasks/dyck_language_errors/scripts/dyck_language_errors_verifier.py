from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
import re

  
class DyckLanguageErrorsVerifier(Verifier):
    """
    验证器用于检查括号闭合错误识别游戏的答案是否正确
    """
    def verify(self, data: Data, test_answer: str):
        """
        验证模型的回答是否正确
        
        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的回答字符串
        @return: 回答是否正确的布尔值
        """
        try:
            test_answer = self.extract_answer(test_solution=test_answer)
            # 获取正确答案
            if data.metadata["is_valid"]:
                correct_answer = "-1"  # 合法序列对应-1
            else:
                correct_answer = str(data.metadata["first_error_pos"])
            
            print(f"验证: 模型答案='{test_answer}', 正确答案='{correct_answer}'")
            
            # 清理和标准化答案
            test_answer = test_answer.strip()
            
            # 检查-1答案（合法序列）
            if correct_answer == "-1":
                # 如果正确答案是-1（合法序列），只接受-1作为回答
                if test_answer == "-1":
                    is_correct = True
                else:
                    is_correct = False
            else:
                # 正确答案是位置数字，需要验证模型回答也是相同数字
                try:
                    is_correct = (int(test_answer) == int(correct_answer))
                except (ValueError, TypeError):
                    # 如果模型回答不是有效数字，验证失败
                    is_correct = False
            
            if is_correct:
                print("验证结果: 正确")
            else:
                print("验证结果: 错误")
                
            return is_correct
            
        except Exception as e:
            print(f"验证时出错: {e}")
            return False
            
    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取答案
        
        @param test_solution: 模型的完整回答
        @return: 提取的答案
        """
        answer_str = test_solution
        if answer_str is None:
            import re 
            # 清理回答文本
            solution = test_solution.strip() if test_solution else ""
            
            # 提取所有数字（包括负数）
            numbers = re.findall(r'-?\d+', solution)
            if numbers:
                # 优先返回"-1"（如果存在）
                if "-1" in numbers:
                    return "-1"
                # 否则返回找到的第一个非负整数
                for num in numbers:
                    if num.isdigit() and int(num) >= 0:
                        return num
                # 如果只有负数，返回第一个
                return numbers[0]
            
            # 检查是否表示合法
            
            
            # 默认返回空字符串
            return "" 
        elif any(keyword in answer_str.lower() for keyword in ["合法", "valid", "correct"]):
            return "-1"
        else:
            return answer_str

