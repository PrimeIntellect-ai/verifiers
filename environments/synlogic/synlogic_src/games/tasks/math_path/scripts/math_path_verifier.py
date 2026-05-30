import re
import json
import numpy as np
from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END


class MathPathVerifier(Verifier):
    """
    验证器用于检查math_path填充游戏的答案是否正确
    """
    def verify(self, data: Data, test_answer: str):
        """
        验证模型的回答是否正确
        
        @param data: 包含问题、元数据等信息的Data对象
        @param test_answer: 模型给出的运算表达式
        @return: 回答是否正确的布尔值
        """
        try:
            test_answer = self.extract_answer(test_solution=test_answer)
        except Exception as e:
            print(f"答案抽取时出错: {e}")
            return False 

        try:
            # 解析元数据
            metadata = data.metadata
            ref_expr = metadata["ref_expr"]
            query_expr = metadata["query_expr"]

            # 验证数字是否被篡改，数字是否在0-9之间。
            test_tmp = test_answer.replace(' ', '').strip()
            query_tmp = query_expr.replace(' ', '').strip()
            ref_tmp = ref_expr.replace(' ', '').strip()
            query_nums = [x for x in query_tmp if '0'<=x<='9' or x=='?']
            test_nums = [x for x in test_tmp if '0'<=x<='9']
            if len(query_nums)!=len(test_nums):
                print(f"所填数字数量不匹配！原始：{ref_tmp}，query：{query_tmp}，模型：{test_tmp}")
                return False
            else:
                for ind, x in enumerate(query_nums):
                    if x=='?':
                        continue
                    if x!=test_nums[ind]:
                        print(f"表达式数字被篡改！原始：{ref_tmp}，query：{query_tmp}，模型：{test_tmp}")
                        return False
                    
            query_symbols = [x for x in query_tmp if x in ['+', '-', '*', '/', '%']]
            test_symbols = [x for x in test_tmp if x in ['+', '-', '*', '/', '%']]
            if len(query_symbols)!=len(test_symbols):
                print(f"表达式运算符号数量不匹配！原始：{ref_tmp}，query：{query_tmp}，模型：{test_tmp}")
                return False
            else:
                for ind, x in enumerate(query_symbols):
                    if x!=test_symbols[ind]:
                        print(f"表达式运算符号被篡改！原始：{ref_tmp}，query：{query_tmp}，模型：{test_tmp}")
                        return False
            
            # 验证回答中的等式是否成立
            try:
                tmp = test_tmp.replace('=', '==')
                if not eval(tmp):
                    print(f"等式不成立！原始：{ref_tmp}，query：{query_tmp}，模型：{test_tmp}")
                    return False
            except:
                print(f"运算表达式错误！原始：{ref_tmp}，query：{query_tmp}，模型：{test_tmp}")
                return False
            
            
            # 所有检查都通过
            print("验证结果: 正确")
            return True
            
        except Exception as e:
            print(f"验证时出错: {e}")
            return False 
        

    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取答案（字符表达式）
        
        @param test_solution: 模型的完整回答
        @return: 提取的矩阵答案字符串
        """
        if not test_solution:
            return ""
        # 尝试提取Python代码块中的矩阵
        code_block_pattern = r'\[\[(.*?)\]\]'
        code_matches = re.findall(code_block_pattern, test_solution)
        
        if code_matches:
            # 使用最后一个匹配内容
            operation_expression = code_matches[-1].strip()
            return operation_expression
        
        # 如果所有方法都失败，返回空字符串
        return ""
        
