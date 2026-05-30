from base.data import Data
from base.verifier import Verifier, THOUGHT_DELIMITER_START, THOUGHT_DELIMITER_END
import re
import json
import ast

    
class SkyscraperPuzzleVerifier(Verifier):
    """
    摩天楼游戏验证器，用于验证模型提供的解答是否正确
    """
    def verify(self, data: Data, test_solution: str):
        """
        验证模型的回答是否符合摩天楼游戏的规则

        @param data: 包含游戏信息的Data对象
        @param test_answer: 游戏类提取的网格数据
        @return: 回答是否正确的布尔值
        """
        try:
            # 获取游戏元数据
            metadata = data.metadata
            n = metadata['n']
            top = metadata['top']
            bottom = metadata['bottom']
            left = metadata['left']
            right = metadata['right']

            self.n = n
            test_answer = self.extract_answer(test_solution)
            
            print(f"验证: 游戏规模 {n}×{n}")
            print(f"上方提示: {top}")
            print(f"下方提示: {bottom}")
            print(f"左侧提示: {left}")
            print(f"右侧提示: {right}")
            
            # 使用提取好的网格数据
            grid = test_answer
            
            # 检查网格是否是字符串，如果是，说明提取失败
            if isinstance(grid, str):
                print("无法提取有效网格")
                return False
                
            print("提取的网格:")
            for row in grid:
                print(row)
            
            # 检查网格规模
            if len(grid) != n or any(len(row) != n for row in grid):
                print(f"网格规模不正确，应为 {n}×{n}")
                return False
            
            # 检查数字范围 (1 到 n)
            for i in range(n):
                for j in range(n):
                    if not isinstance(grid[i][j], int) or grid[i][j] < 1 or grid[i][j] > n:
                        print(f"位置 ({i+1},{j+1}) 的值 {grid[i][j]} 不在有效范围内 (1-{n})")
                        return False
            
            # 检查每行唯一性
            for i in range(n):
                if len(set(grid[i])) != n:
                    print(f"第 {i+1} 行包含重复数字")
                    return False
            
            # 检查每列唯一性
            for j in range(n):
                column = [grid[i][j] for i in range(n)]
                if len(set(column)) != n:
                    print(f"第 {j+1} 列包含重复数字")
                    return False
            
            # 检查从上方观察
            for j in range(n):
                visible_count = self._count_visible_skyscrapers([grid[i][j] for i in range(n)])
                if visible_count != top[j]:
                    print(f"从上方看第 {j+1} 列可见楼数为 {visible_count}，应为 {top[j]}")
                    return False
            
            # 检查从下方观察
            for j in range(n):
                visible_count = self._count_visible_skyscrapers([grid[i][j] for i in range(n-1, -1, -1)])
                if visible_count != bottom[j]:
                    print(f"从下方看第 {j+1} 列可见楼数为 {visible_count}，应为 {bottom[j]}")
                    return False
            
            # 检查从左侧观察
            for i in range(n):
                visible_count = self._count_visible_skyscrapers(grid[i])
                if visible_count != left[i]:
                    print(f"从左侧看第 {i+1} 行可见楼数为 {visible_count}，应为 {left[i]}")
                    return False
            
            # 检查从右侧观察
            for i in range(n):
                visible_count = self._count_visible_skyscrapers(grid[i][::-1])
                if visible_count != right[i]:
                    print(f"从右侧看第 {i+1} 行可见楼数为 {visible_count}，应为 {right[i]}")
                    return False
            
            # 所有检查通过
            print("所有验证规则通过!")
            return True
        
        except Exception as e:
            print(f"验证过程出错: {e}")
            return False
    
    def _count_visible_skyscrapers(self, heights):
        """
        计算从一个方向看过去能看到的摩天楼数量
        
        @param heights: 从观察方向依次排列的摩天楼高度列表
        @return: 可见的摩天楼数量
        """
        visible_count = 0
        max_height = 0
        
        for height in heights:
            if height > max_height:
                visible_count += 1
                max_height = height
        
        return visible_count 
    
    def extract_answer(self, test_solution: str):
        """
        从模型的回答中提取网格数据
        
        @param test_solution: 模型的完整回答
        @return: 提取的解答网格数据
        """
        try:
            n = self.n
            
            # 从 ```python 代码块中提取
            code_block_pattern = r"```python\s*\n([\s\S]*?)\n\s*```"
            code_blocks = re.findall(code_block_pattern, test_solution)
            
            if code_blocks:
                # 取第一个代码块（通常只有一个）
                code_block = code_blocks[0].strip()
                try:
                    # 直接解析代码块
                    grid = ast.literal_eval(code_block)
                    # 验证是否为有效的n×n网格
                    if (isinstance(grid, list) and 
                        len(grid) == n and 
                        all(isinstance(row, list) and len(row) == n for row in grid)):
                        return grid
                except Exception:
                    # 如果直接解析失败，尝试移除注释后再解析
                    code_without_comments = re.sub(r'#.*$', '', code_block, flags=re.MULTILINE)
                    try:
                        grid = ast.literal_eval(code_without_comments.strip())
                        if (isinstance(grid, list) and 
                            len(grid) == n and 
                            all(isinstance(row, list) and len(row) == n for row in grid)):
                            return grid
                    except Exception:
                        pass
            
            # 如果提取失败，返回原始答案
            return test_solution
        except Exception as e:
            print(f"提取网格时出错: {e}")
            return test_solution