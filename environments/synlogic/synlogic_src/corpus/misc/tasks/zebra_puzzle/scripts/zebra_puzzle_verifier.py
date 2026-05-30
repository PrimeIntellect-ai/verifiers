from base.data import Data
from base.verifier import Verifier
import re
from sympy.parsing.latex import parse_latex


import json

def _verifier(answer_str, user_response_str, type_safe=True):
    """验证用户响应是否与答案一致。

    Args:
        answer_str (str): 标准答案字符串（可能是 JSON 序列化的字典或其他类型）。
        user_response_str (str): 用户响应字符串。
        type_safe (bool): 是否要求类型严格一致（默认 True）。

    Returns:
        bool: 验证结果是否一致。
    """
    try:
        user_response_str = user_response_str.split("```json")[-1].split("```")[0].strip()
    except:
        return False
    # 尝试解析答案字符串为 JSON 对象
    try:
        answer_obj = json.loads(answer_str)
    except json.JSONDecodeError:
        # 如果解析失败，直接比较原始字符串
        # print("cannot load as dict, do string exact matching")
        return answer_str == user_response_str
    else:
        if isinstance(answer_obj, dict):
            # 尝试解析用户响应为 JSON 对象
            try:
                response_obj = json.loads(user_response_str)
            except json.JSONDecodeError:
                try:
                    response_obj = eval(user_response_str)
                except:
                    return False
            # 用户响应必须是字典类型才能继续比较
            if not isinstance(response_obj, dict):
                return False
            # 深度比较字典结构
            return _deep_compare(answer_obj, response_obj, type_safe)
        else:
            # 如果答案解析后不是字典，严格比较原始字符串
            return answer_str == user_response_str


def _deep_compare(a, b, type_safe):
    """递归比较两个对象的结构和内容。

    Args:
        a: 期望值（标准答案）
        b: 实际值（用户响应）
        type_safe: 是否要求类型严格一致
    """
    # 检查类型是否一致（如果启用类型安全）
    if type_safe and type(a) is not type(b):
        return False
    
    # 处理字典类型
    if isinstance(a, dict):
        if not isinstance(b, dict) or a.keys() != b.keys():
            return False
        return all(_deep_compare(a[k], b[k], type_safe) for k in a)
    
    # 处理列表类型
    elif isinstance(a, list):
        if not isinstance(b, list) or len(a) != len(b):
            return False
        return all(_deep_compare(x, y, type_safe) for x, y in zip(a, b))
    
    # 处理基本类型（根据类型安全性决定是否转换）
    else:
        return _compare_values(a, b, type_safe)


def _compare_values(a, b, type_safe):
    """比较两个值（对数值类型做宽容处理）"""
    if type_safe:
        return a == b
    else:
        # 尝试将字符串转为数字进行比较
        converted_a = _try_convert_number(a)
        converted_b = _try_convert_number(b)
        return converted_a == converted_b


def _try_convert_number(value):
    """尝试将字符串转为数字（int/float）"""
    if isinstance(value, str):
        try: return int(value)
        except: pass
        try: return float(value)
        except: pass
    return value

class ZebraPuzzleVerifier(Verifier):
    """
    Verifier for Zebra Puzzle
    """
    def verify(self, data: Data, test_solution: str):
        """The scoring function for zebra puzzle.

        Reference: Trung, Luong, et al. "Reft: Reasoning with reinforced fine-tuning." Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 2024.

        Args:
            solution_str: the solution text
            ground_truth: the ground truth
            format_score: the score for the format
            acc_score: the score for the correct answer
        """

        extracted_answer = self.extract_answer(test_solution)
        ground_truth = data.answer
        try:
            correct = _verifier(answer_str=ground_truth, user_response_str=extracted_answer, type_safe=True)
        except Exception as e:
            print(f"Gold: {ground_truth} Response: {extracted_answer} Error: {str(e)}")
            correct = False
        
        if correct:
            acc_score = 1.0
        else:
            acc_score = 0

        return acc_score
    
    def extract_answer(self, test_solution: str):
        return test_solution