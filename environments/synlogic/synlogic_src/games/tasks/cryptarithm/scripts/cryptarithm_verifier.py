import re
from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END



    
class CryptarithmVerifier(Verifier):
    """
    验证器用于检查密码算术游戏的答案是否正确
    """
    def verify(self, data: Data, test_solution: str):
        """
        验证模型的回答是否正确
        
        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的回答字符串
        @return: 回答是否正确的布尔值
        """
        try:
            test_answer = self.extract_answer(test_solution)
            # 获取元数据中的正确答案
            correct_answer = data.answer
            
            print(f"验证: 模型答案='{test_answer}', 正确答案='{correct_answer}'")
            
            # 清理答案字符串
            test_answer = test_answer.strip()
            
            # 标准化等式格式（移除空格）
            test_answer = test_answer.replace(" ", "")
            correct_answer = correct_answer.replace(" ", "")
            
            # 检查答案是否完全匹配
            is_correct = (test_answer == correct_answer)
            
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
        if not test_solution:
            return ""
        # 首先规范化空格和其他格式
        # 处理全大写的情况
        test_solution = test_solution.replace("THE ANSWER IS", "The answer is")
        test_solution = test_solution.replace("答案是：", "答案是:")
        test_solution = test_solution.replace("答案：", "答案:")
        
        # 尝试匹配数字等式模式（支持多个操作符和负数）
        # 例如：123 + 456 = 579 或 12 + 34 - 5 = 41 或 123 + 456 - 789 = -210
        equation_patterns = [
            r'(\d+(?:\s*(?:\+|\-|\*)\s*\d+)+\s*=\s*-?\d+)',  # 匹配包含多个操作符的等式，结果可能为负
            r'(\d+(?:\s*(?:\+|\-|\*)\s*\d+)+\s*=\s*-?\d+)[.。]*$',  # 结尾的等式，结果可能为负
            r'(\d+\s*(?:\+|\-|\*)\s*\d+\s*=\s*-?\d+)'  # 匹配简单等式，结果可能为负
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, test_solution)
            if matches:
                # 取最后一个匹配结果
                return matches[-1].strip()
        
        # 中文答案提取模式
        cn_patterns = [
            r'答案是[：:]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'答案[：:]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'我的答案是[：:]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'正确答案[：:]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'数字等式是[：:]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'数字等式为[：:]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'等式为[：:]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'等式是[：:]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'结果是[：:]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'结果为[：:]\s*([0-9\s\+\-\*=]+)[.。]*$'
        ]
        
        # 英文答案提取模式
        en_patterns = [
            r'[Tt]he answer is[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'[Tt]he answer[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'[Aa]nswer[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'[Mm]y answer is[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'[Tt]he final answer is[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'[Tt]he equation is[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'[Tt]he result is[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'[Tt]he numeric equation is[：:=]\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'[Tt]herefore,\s*([0-9\s\+\-\*=]+)[.。]*$',
            r'[Ss]o,\s*([0-9\s\+\-\*=]+)[.。]*$'
        ]
        
        # 尝试匹配所有模式
        patterns = cn_patterns + en_patterns
        
        for pattern in patterns:
            matches = re.findall(pattern, test_solution, re.DOTALL)
            if matches:
                answer = matches[-1].strip()
                # 移除美元符号（常用于标记LaTeX数学表达式）和句号
                answer = answer.replace("$", "").replace("。", "").replace(".", "")
                
                # 检查是否是有效等式（结果可能为负）
                if re.match(r'\d+(?:\s*(?:\+|\-|\*)\s*\d+)+\s*=\s*-?\d+', answer):
                    return answer
                
        # 如果上述模式都没有匹配到，尝试从最后一行提取等式
        lines = test_solution.strip().split('\n')
        for line in reversed(lines):  # 从最后一行开始向上查找
            equation_match = re.search(r'\d+(?:\s*(?:\+|\-|\*)\s*\d+)+\s*=\s*-?\d+', line)
            if equation_match:
                return equation_match.group(0)
        
        # 尝试从文本中提取任何看起来像等式的内容
        general_equation_pattern = r'\d+\s*(?:\+|\-|\*)\s*\d+\s*=\s*-?\d+'
        all_equations = re.findall(general_equation_pattern, test_solution)
        if all_equations:
            # 返回最后一个找到的等式
            return all_equations[-1]
        
        # 如果没有匹配到任何模式，返回空字符串
        return ""