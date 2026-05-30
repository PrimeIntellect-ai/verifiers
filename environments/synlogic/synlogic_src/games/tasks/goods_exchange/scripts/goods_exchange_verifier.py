import re
from base.data import Data
from base.verifier import Verifier

class GoodsExchangeVerifier(Verifier):
    """
    验证器用于检查物品交换游戏的答案是否正确
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
            correct_answer = data.metadata["owns_after"]
            
            print(f"验证: 模型答案='{test_answer}', 正确答案='{correct_answer}'")
            
            # 解析模型答案
            model_ownership = self._parse_answer(test_answer)
            # 解析正确答案
            correct_ownership = self._parse_answer(correct_answer)
            
            # 比较两个答案是否完全一致
            is_correct = self._compare_answers(model_ownership, correct_ownership)
            
            if is_correct:
                print("验证结果: 正确")
            else:
                print("验证结果: 错误")
                # 打印详细的不匹配信息
                self._print_difference(model_ownership, correct_ownership)
                
            return is_correct
            
        except Exception as e:
            print(f"验证时出错: {e}")
            return False
    
    def _parse_answer(self, answer_str):
        """
        解析答案字符串为物品归属字典
        
        @param answer_str: 答案字符串，格式为"(('人1','物品1'),('人2','物品2'),...)"或"(人1,物品1),(人2,物品2),..."
        @return: 归属关系字典 {人: 物品}
        """
        if not answer_str:
            return {}
            
        result = {}
        try:
            # 预处理：只处理最外层的空格，保留内部结构
            answer_str = answer_str.strip()
            
            # 尝试使用 eval 解析 Python tuple 格式
            pairs = eval(answer_str)
            if isinstance(pairs, tuple):
                for pair in pairs:
                    if isinstance(pair, tuple) and len(pair) == 2:
                        person, item = pair
                        # 处理每个值中的空格：移除两端空格
                        result[person.strip()] = item.strip()
                return result
        except Exception as e:
            # 如果 eval 失败，记录错误并尝试解析旧格式
            print(f"Eval解析失败: {e}，尝试手动解析")
            
            # 移除最外层的括号（如果有）
            if answer_str.startswith('('):
                answer_str = answer_str[1:]
            if answer_str.endswith(')'):
                answer_str = answer_str[:-1]
            
            # 更健壮的手动解析逻辑
            person_item_pairs = []
            current_pair = ""
            bracket_count = 0
            
            # 更智能地分割答案字符串
            for char in answer_str:
                if char == '(':
                    bracket_count += 1
                    current_pair += char
                elif char == ')':
                    bracket_count -= 1
                    current_pair += char
                    if bracket_count == 0:
                        person_item_pairs.append(current_pair)
                        current_pair = ""
                elif char == ',' and bracket_count == 0:
                    # 跳过顶层逗号
                    continue
                else:
                    current_pair += char
            
            # 处理每一对
            for pair in person_item_pairs:
                pair = pair.strip()
                # 移除括号
                if pair.startswith('('):
                    pair = pair[1:]
                if pair.endswith(')'):
                    pair = pair[:-1]
                    
                # 拆分人和物品
                try:
                    # 使用更健壮的分割方法
                    parts = []
                    quote_count = 0
                    current = ""
                    
                    for char in pair:
                        if char in "\"'" and (len(current) == 0 or current[-1] != '\\'):
                            quote_count = 1 - quote_count
                        
                        if char == ',' and quote_count == 0:
                            parts.append(current.strip())
                            current = ""
                        else:
                            current += char
                    
                    if current:
                        parts.append(current.strip())
                    
                    if len(parts) >= 2:
                        person = parts[0].strip().strip("'\"")
                        item = parts[1].strip().strip("'\"")
                        result[person] = item
                except Exception as e:
                    print(f"解析对 '{pair}' 时出错: {e}")
                    
        return result
    
    def _compare_answers(self, model_ownership, correct_ownership):
        """
        比较两个归属关系字典是否相同
        
        @param model_ownership: 模型回答的归属关系
        @param correct_ownership: 正确的归属关系
        @return: 是否完全一致
        """
        # 检查人数是否相同
        if len(model_ownership) != len(correct_ownership):
            return False
        
        # 创建小写人名到原始人名的映射
        model_lower_to_original = {person.lower(): person for person in model_ownership}
        
        # 检查每个人的物品是否一致
        for person in correct_ownership:
            # 如果模型答案中没有这个人（不区分大小写）
            if person.lower() not in model_lower_to_original:
                return False
                
            # 获取模型答案中对应的原始人名
            model_person = model_lower_to_original[person.lower()]
                
            # 如果人的物品不匹配（不区分大小写）
            if model_ownership[model_person].lower() != correct_ownership[person].lower():
                return False
                
        return True
    
    def _print_difference(self, model_ownership, correct_ownership):
        """
        打印两个归属关系之间的差异
        
        @param model_ownership: 模型回答的归属关系
        @param correct_ownership: 正确的归属关系
        """
        print("\n差异详情:")
        
        # 创建小写人名到原始人名的映射
        model_lower_to_original = {person.lower(): person for person in model_ownership}
        correct_lower_to_original = {person.lower(): person for person in correct_ownership}
        
        # 检查正确答案中的每个人
        for person in correct_ownership:
            person_lower = person.lower()
            if person_lower not in model_lower_to_original:
                print(f"  - 模型答案中缺少: {person}")
            else:
                model_person = model_lower_to_original[person_lower]
                if model_ownership[model_person].lower() != correct_ownership[person].lower():
                    print(f"  - {person}: 模型答案={model_ownership[model_person]}, 正确答案={correct_ownership[person]}")
        
        # 检查模型答案中的额外人员
        for person in model_ownership:
            if person.lower() not in correct_lower_to_original:
                print(f"  - 模型答案中多余: {person}") 
                
    def extract_answer(self, text):
        """从文本中提取答案。
        
        Args:
            text (str): 输入文本
            
        Returns:
            str: 提取的答案，格式为 "(('人1','物品1'),('人2','物品2'),...)"
        """
        if not text:
            return ""
        
        # 尝试从 Python markdown 代码块中提取
        code_block_pattern = r'```python\s*\n(.*?)\n```'
        code_blocks = re.findall(code_block_pattern, text, re.DOTALL)
        if code_blocks:
            # 使用最后一个代码块
            last_block = code_blocks[-1].strip()
            if last_block.startswith("(") and last_block.endswith(")"):
                return last_block
        return ""