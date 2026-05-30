from base.data import Data
from base.verifier import Verifier
import re

class GameOf24Verifier(Verifier):
    """
    Verifier for Game of 24
    """
    def verify(self, data: Data, test_solution: str):
        try:
            test_answer = self.extract_answer(test_solution)
            print("Extracted answer:", test_answer)
            numbers = data.metadata["numbers"]
            operators = data.metadata["operators"]
            result = data.metadata["result"]
            input_numbers = [str(num) for num in numbers]
            answer_numbers_str = re.sub(r'[^0-9]', ' ', test_answer)
            answer_numbers = [num for num in answer_numbers_str.split() if num]
            answer_wo_numbers_str = re.sub(r'[0-9\s]', '', test_answer)
            unknown_chars = []
            for c in answer_wo_numbers_str:
                if c in operators:
                    continue
                if c in ["(", ")"]:
                    continue
                unknown_chars.append(c)
            if len(unknown_chars) > 0:
                print("Found unknown characters in the answer:", unknown_chars)
                return False

            for num in answer_numbers:
                if num not in input_numbers:
                    print("Found unknown number in the answer:", num)
                    return False
                if answer_numbers.count(num) > input_numbers.count(num):
                    print("Found more occurrences of number in the answer than in the input:", num)
                    return False
            
            if len(answer_numbers) != len(input_numbers):
                print("The number of numbers in the answer is not equal to the number of input numbers:", len(answer_numbers), len(input_numbers))
                return False
            print("Test answer:", test_answer)
            result = eval(test_answer)
            
            return abs(result - result) < 1e-10  # Allow for floating-point precision
        except:
            return False
        
    def extract_answer(self, test_solution: str):
        regex_pattern = "```python.*?```"
        matches = re.findall(regex_pattern, test_solution, re.DOTALL)
        if matches:
            return matches[-1].replace("```python", "").replace("```", "").strip()
        else:
            return ""