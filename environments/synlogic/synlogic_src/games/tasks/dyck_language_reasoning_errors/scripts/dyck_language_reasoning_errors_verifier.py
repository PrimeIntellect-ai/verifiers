from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
import re

    
class DyckLanguageReasoningErrorsVerifier(Verifier):
    """
    Dyck语言推理错误识别验证器
    """
    def verify(self, data: Data, test_answer: str):
        """
        验证模型的回答是否正确
        
        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的答案字符串
        @return: 回答是否正确的布尔值
        """
        try:
            test_answer = self.extract_answer(test_solution=test_answer)
            # 获取元数据中的正确答案
            correct_indices = data.metadata["error_indices"]
            # 格式化为正确的答案字符串格式
            expected_answer = self._format_answer(correct_indices)
            
            print(f"验证: 模型答案='{test_answer}', 正确答案='{expected_answer}'")
            
            # 检查不明确的答案
            if "不确定" in test_answer or "不知道" in test_answer or "unclear" in test_answer.lower():
                print("验证结果: 错误")
                return False
            
            # 清理模型答案，允许一定的格式变化
            cleaned_test_answer = self._standardize_answer(test_answer)
            
            if not correct_indices and (cleaned_test_answer == "" or cleaned_test_answer.lower() in ["无问题", "no", "无错误", "no error", "no errors", "no mistakes", "all correct"]):
                # 如果没有错误，且模型回答是空字符串或表示无问题，则正确
                is_correct = True
            else:
                # 将两个答案转换为数字集合进行比较
                test_error_indices = self._extract_error_indices(cleaned_test_answer)
                expected_error_indices = set(correct_indices)
                
                # 检查两个集合是否相同
                is_correct = test_error_indices == expected_error_indices
            
            if is_correct:
                print("验证结果: 正确")
            else:
                print("验证结果: 错误")
                
            return is_correct
            
        except Exception as e:
            print(f"验证时出错: {e}")
            return False
    
    def _standardize_answer(self, answer: str) -> str:
        """
        标准化答案字符串
        
        @param answer: 原始答案字符串
        @return: 标准化后的答案字符串
        """
        # 如果答案为空或仅包含空白字符
        if not answer or answer.strip() == "":
            return ""
            
        # 如果答案表示没有错误
        if answer.lower() in ["无问题", "no", "无错误", "no error", "no errors", "no mistakes", "all correct"]:
            return ""
        
        # 替换中文逗号为英文逗号
        answer = answer.replace('，', ',')
        # 移除所有非数字和逗号的字符
        answer = re.sub(r'[^0-9,]', '', answer)
        
        return answer
    
    def _extract_error_indices(self, answer: str) -> set:
        """
        从答案字符串中提取错误索引集合
        
        @param answer: 答案字符串
        @return: 错误索引集合
        """
        if not answer or answer.strip() == "":
            return set()
        
        # 先替换中文逗号为英文逗号
        answer = answer.replace('，', ',')
        
        # 分割逗号分隔的数字字符串，并转换为整数集合
        try:
            return set(int(idx) for idx in answer.split(',') if idx and idx.strip().isdigit())
        except ValueError:
            return set()
    
    def _format_answer(self, error_indices: list) -> str:
        """
        格式化答案字符串
        
        @param error_indices: 错误步骤的索引列表
        @return: 格式化的答案字符串
        """
        if not error_indices:
            return ""
        
        # 按照数字大小排序
        sorted_indices = sorted(error_indices)
        
        # 转换为字符串，使用英文逗号分隔
        return ",".join(map(str, sorted_indices)) 
    
    def extract_answer(self, test_solution: str) -> str:
        """
        从模型的回答中提取错误索引
        
        @param test_solution: 模型的完整回答
        """
        answer_str = test_solution
        return answer_str

if __name__ == "__main__":
    test_response = '''<think>\nOkay, so I'm trying to figure out if there are any errors in the reasoning process provided for checking the proper closure of brackets in the sequence: {}[][][]{}{}([]). Let me go through each thought step by step and see where things might have gone wrong.\n\nStarting with Thought 2: the initial stack is empty, which makes sense because we haven't processed any characters yet. So that seems correct.\n\nThought 3: The first character is '{', so it's pushed onto the stack. The stack becomes '{'. Hmm, but in the thought, it's written as '{[', which is incorrect. The stack after the first character should only have '{'. So this thought is wrong because it adds an extra '[' that wasn't there yet. Wait, no, actually, the input sequence starts with '{', so maybe there's a mistake in the thought. Let me recheck. Oh, looking again, no, the input sequence is {}[][][]{}{}([)]}, so the first two characters are '{' and '}'. So after processing '{', the stack should be '{', not '{['. So Thought 3 is wrong because it mistakenly pushes '[' instead of just '{'.\n\nMoving to Thought 4: Next character is '}' which correctly matches the top '{' in the stack, so the stack becomes empty. This part is correct.\n\nThought 5: The next character is '[', so stack becomes '['. Correct.\n\nThought 6: Next is ']', which pops the '[', so stack is empty. Correct.\n\nThought 7: Next is '[', stack becomes '['. Correct.\n\nThought 8: Next is ']', so stack is empty. Correct.\n\nThought 9: Next is '[', stack becomes '['. Correct.\n\nThought 10: Next is ']' which pops the '[', so stack should be empty again. However, in the thought, it says the stack becomes ']', which is incorrect. So Thought 10 is wrong because it added a ']' instead of popping it off. Wait no, the character is ']', which should pop the top '[', so the stack should go from '[' to empty. But the thought says the stack becomes ']' which is wrong. So Thought 10 is incorrect.\n\nThought 11: Next is '{', stack is now '{'. Correct.\n\nThought 12: Next is '}', popping the '{', stack is empty. Correct.\n\nThought 13: Next is '{', stack becomes '{'. Correct.\n\nThought 14: Next is '}', stack is empty. Correct.\n\nThought 15: Next is '(', stack becomes '('. Correct.\n\nThought 16: Next is '[', so stack becomes '(['. Correct.\n\nThought 17: Next is ']', which pops the '[', so stack becomes '('. Correct.\n\nThought 18: Next is ')', which pops the '(', so stack is empty. Correct.\n\nThought 19: After processing all characters, stack is empty. So the sequence is balanced.\n\nWait, but the input sequence actually ends with ']})', right? The sequence given is {}[][][]{}{}([]). So after processing '(' and '[', we have '(', then '['. Then when we process ']', it pops the '[', stack becomes '('. Then next is ')', which pops '(', stack is empty. So everything is correct up to that point.\n\nBut in the earlier steps, the issue was in Thought 10. Because when processing the fifth character, which is '[', followed by ']', the stack should go back to empty. So Thought 10 says the stack becomes ']', which is incorrect. It should go back to empty.\n\nWait, let me recount the steps:\n\nThe input is {}[][][]{}{}([]). So let's break it down:\n\n1. '{' : stack becomes '{'\n2. '}' : stack empty\n3. '[' : stack becomes '['\n4. ']' : stack empty\n5. '[' : '['\n6. ']' : stack empty\n7. '[' : '['\n8. ']' : stack empty\n9. '{' : '{'\n10. '}' : stack empty\n11. '{' : '{'\n12. '}' : stack empty\n13. '(' : '('\n14. '[' : '(['\n15. ']' : '('\n16. ')' : stack empty.\n\nWait, but in the original thought process, after step 10, the character is ']' which pops the '[', making the stack empty, but the thought says stack is ']'. So that's definitely incorrect.\n\nAlso, in step 3, the thought says after '{' the stack is '{[', which is incorrect because the next character is '}', not another '['. So step 3 is wrong because it added an extra '[' instead of just '{' and then processing '}' in step 4 correctly.\n\nSo the errors are in Thought 3 and Thought 10.\n</think>\n\nThe errors are in Thought 3 and Thought 10 because Thought 3 incorrectly adds an extra '[' instead of just '{' and Thought 10 mistakenly leaves the stack as ']' instead of empty after popping.\n\n<answer>[3,20]</answer>'''
    metadata = {"trace_id": "77db72eb-a9db-46cd-96ea-5a49eba78792", "dyck_sequence": "{}[][][]{}{}([])", "thoughts": ["Thought 1: 我们应该逐个处理输入并跟踪栈的配置。", "Thought 2: 栈: 空", "Thought 3: { ; 栈: {[", "Thought 4: } ; 栈: 空", "Thought 5: [ ; 栈: [", "Thought 6: ] ; 栈: 空", "Thought 7: [ ; 栈: [", "Thought 8: ] ; 栈: 空", "Thought 9: [ ; 栈: [", "Thought 10: ] ; 栈: ]", "Thought 11: { ; 栈: {", "Thought 12: } ; 栈: 空", "Thought 13: { ; 栈: {", "Thought 14: } ; 栈: 空", "Thought 15: ( ; 栈: (", "Thought 16: [ ; 栈: ([", "Thought 17: ] ; 栈: (", "Thought 18: ) ; 栈: 空", "Thought 19: 现在，我们已经到达结尾。最终栈是空的。"], "error_indices": [3, 10], "n_types": 3, "total_length": 15, "n_errors": 2}
    test_data = Data(question="", answer="", metadata=metadata)
    test_verifier = DyckLanguageReasoningErrorsVerifier()
    extracted_answer = test_verifier.extract_answer(test_response)
    print(extracted_answer)
    print(test_verifier.verify(data=test_data, test_answer=test_response))