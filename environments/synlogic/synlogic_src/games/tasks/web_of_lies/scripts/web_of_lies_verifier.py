import re
from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END

class WebOfLiesVerifier(Verifier):
    """
    验证器用于检查谎言之网游戏的答案是否正确
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
            # 获取预期答案和测试答案
            expected_answer = data.answer.lower()
            
            # 清理测试答案
            test_answer = test_answer.lower()
            
            # 提取预期答案中的真假值
            expected_truths = self._parse_answer(expected_answer)
            
            # 提取测试答案中的真假值
            test_truths = self._parse_answer(test_answer)
            
            print(f"验证: 预期答案={expected_truths}, 模型答案={test_truths}")
            
            # 检查答案列表长度是否匹配
            if len(expected_truths) != len(test_truths):
                print(f"验证失败: 答案长度不匹配，预期 {len(expected_truths)}，实际 {len(test_truths)}")
                return False
            
            # 检查每个位置的答案是否匹配
            for i, (expected, actual) in enumerate(zip(expected_truths, test_truths)):
                if expected != actual:
                    print(f"验证失败: 第 {i+1} 个答案不匹配，预期 {expected}，实际 {actual}")
                    return False
            
            print("验证成功: 所有答案匹配")
            return True
            
        except Exception as e:
            print(f"验证时出错: {e}")
            return False
    
    def _parse_answer(self, answer_str):
        """
        从答案字符串中解析出真假值列表
        
        @param answer_str: 答案字符串
        @return: 真假值列表，True表示说真话，False表示说谎话
        """
        # 尝试匹配英文答案格式 (yes/no)
        yes_pattern = r'yes|true|truth'
        no_pattern = r'no|false|lie'
        
        # 尝试匹配中文答案格式 (是/否)
        cn_yes_pattern = r'是|真话|真'
        cn_no_pattern = r'否|假话|假|谎'
        
        # 组合模式
        yes_patterns = f'({yes_pattern}|{cn_yes_pattern})'
        no_patterns = f'({no_pattern}|{cn_no_pattern})'
        
        # 根据答案字符串中的关键词确定真假值
        truths = []
        
        # 寻找所有可能的yes/no或是/否答案
        all_answers = re.findall(rf'{yes_patterns}|{no_patterns}', answer_str)
        
        for match in all_answers:
            # match是一个元组，需要找到非空的元素
            match_str = next((m for m in match if m), '')
            
            if re.search(yes_pattern, match_str) or re.search(cn_yes_pattern, match_str):
                truths.append(True)
            elif re.search(no_pattern, match_str) or re.search(cn_no_pattern, match_str):
                truths.append(False)
        
        return truths 
    
    def extract_answer(self, test_solution: str) -> str:
        """
        从模型的回答中提取答案
        
        @param test_solution: 模型的完整回答
        @return: 提取的答案
        """
        if not test_solution:
            return ""
        # 中文模式
        cn_patterns = [
            r'答案是[：:]\s*\*\*([^*]+)\*\*[.。]*$',  # 匹配"答案是：**是，否，是**"格式
        ]
        
        # 英文模式
        en_patterns = [
            r'[Tt]he answer is[：:=]\s*\*\*([^*]+)\*\*[.。]*$',  # 匹配"The answer is: **yes, no, yes**"格式
        ]
        
        # 尝试匹配所有模式
        patterns = cn_patterns + en_patterns
        
        for pattern in patterns:
            matches = re.findall(pattern, test_solution, re.DOTALL)
            if matches:
                return matches[-1].strip()
        
        # 如果上面的模式都没匹配到，尝试更宽松的匹配
        # 查找最后一行中的加粗文本
        lines = test_solution.strip().split('\n')
        if lines:
            last_line = lines[-1].strip()
            bold_match = re.search(r'\*\*([^*]+)\*\*', last_line)
            if bold_match:
                return bold_match.group(1).strip()
            
            # 尝试匹配"答案是"或"The answer is"后面的文本
            answer_match = re.search(r'(?:答案是|[Tt]he answer is)[：:=]?\s*(.*?)(?:[.。]|$)', last_line)
            if answer_match:
                return answer_match.group(1).strip()
        
        # 如果没有找到格式化的答案，尝试直接匹配yes/no或是/否序列
        yes_no_pattern = r'(?:\b(?:yes|no|是|否)\b[,，\s]*)+' 
        matches = re.findall(yes_no_pattern, test_solution.lower())
        if matches:
            return matches[-1].strip()
        
        # 如果没有匹配到任何模式，返回空字符串
        return ""