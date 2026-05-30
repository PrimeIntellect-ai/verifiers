from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
import re
import json
from typing import List, Tuple


class MinesweeperVerifier(Verifier):
    """
    Verifier for Minesweeper puzzle
    扫雷游戏验证器
    """
    def verify(self, data: Data, test_solution: str, **kwargs):
        try:
            # 从解答中提取地雷坐标
            predicted_mines = self.extract_answer(test_solution)
            
            # 从metadata中获取确定性地雷坐标
            expected_mines = data.metadata["current_mines"]
            
            # 验证提取的坐标是否正确
            if set(tuple(mine) for mine in predicted_mines) == set(tuple(mine) for mine in expected_mines):
                return True
            
            return False
            
        except Exception as e:
            # 如果验证过程中发生任何错误，返回False
            print(f"Verification error: {e}")
            return False
    
    def extract_answer(self, response: str) -> List[Tuple[int, int]]:
        """从模型的响应中提取地雷坐标
        Extract mine coordinates from the model's response"""
        patterns = [
            r'\[\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)(?:\s*,\s*\(\s*\d+\s*,\s*\d+\s*\))*\s*\]',  # [(0,1),(2,3)]
            r'\[\s*\[\s*(\d+)\s*,\s*(\d+)\s*\](?:\s*,\s*\[\s*\d+\s*,\s*\d+\s*\])*\s*\]',  # [[0,1],[2,3]]
            r'\(\s*(\d+)\s*,\s*(\d+)\s*\)(?:\s*,\s*\(\s*\d+\s*,\s*\d+\s*\))*',  # (0,1),(2,3)
        ]
        
        for pattern in patterns:
            coords = []
            for match in re.finditer(pattern, response):
                try:
                    # 提取所有坐标对
                    coord_pattern = r'(?:\(|\[)\s*(\d+)\s*,\s*(\d+)\s*(?:\)|\])'
                    for coord_match in re.finditer(coord_pattern, match.group(0)):
                        i, j = int(coord_match.group(1)), int(coord_match.group(2))
                        coords.append((i, j))
                except Exception:
                    continue
            
            if coords:
                return coords
        
        # 如果没有找到坐标，尝试查找可能是坐标的任何数字
        number_pairs = re.findall(r'(\d+)[^\d]+(\d+)', response)
        if number_pairs:
            return [(int(i), int(j)) for i, j in number_pairs]
        
        return []