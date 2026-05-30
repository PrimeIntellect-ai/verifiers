from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
import re

import json

class CipherVerifier(Verifier):
    """
    Verifier for Cipher
    """
    def verify(self, data: Data, test_solution: str):
        """The scoring function for cipher.
        """
        try:
            answer = self.extract_answer(test_solution)
            answer = answer.split("```python")[-1].split("```")[0].strip()
            answer = eval(answer)
            gold_answer = data.answer
            gold_answer = [[gold_answer[2:-2]]]
            if answer == gold_answer:
                return 1.0
            else:
                return 0.0
        except Exception as e:
            return 0.0
    
    def extract_answer(self, test_solution: str):
        return test_solution