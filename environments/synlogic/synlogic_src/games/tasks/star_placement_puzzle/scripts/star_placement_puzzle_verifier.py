from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
import re
import json
import ast

import re

class StarPlacementPuzzleVerifier(Verifier):
    """
    星星放置游戏验证器，用于验证模型提供的解答是否正确
    """
    def verify(self, data: Data, test_solution: str):
        """
        验证模型的回答是否符合星星放置游戏的规则

        @param data: 包含游戏信息的Data对象
        @param star_coords: 通过extract_answer提取的星星坐标字典 {区域: [(行,列), ...]}
        @return: 回答是否正确的布尔值
        """
        try:
            star_coords = self.extract_answer(test_solution)
            # 获取游戏元数据
            metadata = data.metadata
            n = metadata['n']
            k = metadata['k']
            region_grid = metadata['region_grid']
            
            print(f"验证: 游戏规模 {n}×{n}, 每行/列/区域星星数量: {k}")
            
            # 检查是否有有效的星星坐标
            if not star_coords:
                print("无法从回答中提取有效的星星坐标")
                return False
            
            # 创建一个表示星星位置的网格
            star_grid = [[0 for _ in range(n)] for _ in range(n)]
            for region, coords in star_coords.items():
                for coord in coords:
                    row, col = coord
                    if row < 0 or row >= n or col < 0 or col >= n:
                        print(f"无效坐标: ({row},{col}) - 超出网格范围")
                        return False
                    star_grid[row][col] = 1
            
            # 打印星星网格以便调试
            print("星星网格:")
            for row in star_grid:
                print(''.join(['* ' if cell == 1 else '. ' for cell in row]))
            
            # 1. 检查每行是否有k颗星星
            for i in range(n):
                stars_in_row = sum(star_grid[i])
                if stars_in_row != k:
                    print(f"行 {i+1} 有 {stars_in_row} 颗星星，应该有 {k} 颗")
                    return False
            
            # 2. 检查每列是否有k颗星星
            for j in range(n):
                stars_in_col = sum(star_grid[i][j] for i in range(n))
                if stars_in_col != k:
                    print(f"列 {j+1} 有 {stars_in_col} 颗星星，应该有 {k} 颗")
                    return False
            
            # 3. 检查每个区域是否有k颗星星
            regions = {}
            for i in range(n):
                for j in range(n):
                    region = region_grid[i][j]
                    if region not in regions:
                        regions[region] = []
                    regions[region].append((i, j))
            
            for region, cells in regions.items():
                stars_in_region = sum(star_grid[i][j] for i, j in cells)
                if stars_in_region != k:
                    print(f"区域 {region} 有 {stars_in_region} 颗星星，应该有 {k} 颗")
                    return False
            
            # 4. 检查星星是否互不相邻（水平、垂直、对角线）
            for i in range(n):
                for j in range(n):
                    if star_grid[i][j] == 1:
                        # 检查周围8个方向
                        for di in [-1, 0, 1]:
                            for dj in [-1, 0, 1]:
                                if di == 0 and dj == 0:
                                    continue  # 跳过自身
                                ni, nj = i + di, j + dj
                                if 0 <= ni < n and 0 <= nj < n and star_grid[ni][nj] == 1:
                                    print(f"星星在 ({i},{j}) 与星星在 ({ni},{nj}) 相邻")
                                    return False
            
            # 所有检查通过
            print("所有验证规则通过!")
            return True
            
        except Exception as e:
            print(f"验证过程出错: {e}")
            return False 
        
    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取星星坐标
        
        @param test_solution: 模型的完整回答
        @return: 提取的星星坐标字典 {区域: [(行,列), ...]}
        """
        try:
            # 从Python代码块中提取
            python_match = re.search(r'```python\s*\n(.*?)\n\s*```', test_solution, re.DOTALL)
            if not python_match:
                print("回答中没有找到```python代码块")
                return None
                
            code_content = python_match.group(1)
            
            # 尝试从Python代码中提取字典
            try:
                # 先尝试直接提取字典内容
                dict_match = re.search(r'\{[^{}]*\}', code_content, re.DOTALL)
                if dict_match:
                    dict_str = dict_match.group(0)
                    try:
                        # 将字符串转换为字典
                        coords_dict = ast.literal_eval(dict_str)
                        # 如果成功且是字典类型，继续处理
                        if isinstance(coords_dict, dict):
                            # 将坐标减1（因为用户输入的坐标是1-索引）
                            result = {}
                            for region, coords in coords_dict.items():
                                result[region] = [(row-1, col-1) for row, col in coords]
                            return result
                    except (ValueError, SyntaxError) as e:
                        print(f"解析字典字符串时出错: {e}")
                
                # 如果上面的方法失败，尝试解析变量赋值
                assign_match = re.search(r'(\w+)\s*=\s*(\{[^{}]*\})', code_content, re.DOTALL)
                if assign_match:
                    dict_str = assign_match.group(2)
                    try:
                        # 将字符串转换为字典
                        coords_dict = ast.literal_eval(dict_str)
                        # 如果成功且是字典类型，继续处理
                        if isinstance(coords_dict, dict):
                            # 将坐标减1（因为用户输入的坐标是1-索引）
                            result = {}
                            for region, coords in coords_dict.items():
                                result[region] = [(row-1, col-1) for row, col in coords]
                            return result
                    except (ValueError, SyntaxError) as e:
                        print(f"解析变量赋值字典时出错: {e}")
            except Exception as e:
                print(f"从Python代码中提取字典时出错: {e}")
            
            return None
            
        except Exception as e:
            print(f"提取星星坐标时出错: {e}")
            return None