import os
import sys

if os.path.dirname(os.path.abspath(__file__)) not in sys.path:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from games.tasks.game_of_24.scripts.game_of_24_verifier import GameOf24Verifier
from games.tasks.cryptarithm.scripts.cryptarithm_verifier import CryptarithmVerifier
from games.tasks.survo.scripts.survo_verifier import SurvoVerifier
from games.tasks.campsite.scripts.campsite_verifier import CampsiteVerifier
from games.tasks.skyscraper_puzzle.scripts.skyscraper_puzzle_verifier import SkyscraperPuzzleVerifier
from games.tasks.web_of_lies.scripts.web_of_lies_verifier import WebOfLiesVerifier
from games.tasks.goods_exchange.scripts.goods_exchange_verifier import GoodsExchangeVerifier
from games.tasks.sudoku.scripts.sudoku_verifier import SudokuVerifier
from corpus.misc.tasks.zebra_puzzle.scripts.zebra_puzzle_verifier import ZebraPuzzleVerifier
from corpus.misc.tasks.arc_agi.scripts.arc_agi_verifier import ArcAGIVerifier
from games.tasks.object_properties.scripts.object_properties_verifier import ObjectPropertiesVerifier
from games.tasks.object_counting.scripts.object_counting_verifier import ObjectCountingVerifier
from games.tasks.star_placement_puzzle.scripts.star_placement_puzzle_verifier import StarPlacementPuzzleVerifier
from games.tasks.arrow_maze.scripts.arrow_maze_verifier import ArrowMazeVerifier
from games.tasks.kukurasu.scripts.kukurasu_verifier import KukurasuVerifier
from games.tasks.number_wall.scripts.number_wall_verifier import NumberWallVerifier
from games.tasks.numbrix.scripts.numbrix_verifier import NumbrixVerifier
from games.tasks.norinori.scripts.norinori_verifier import NorinoriVerifier
from games.tasks.minesweeper.scripts.minesweeper_verifier import MinesweeperVerifier
from games.tasks.operation.scripts.operation_verifier import OperationVerifier
from games.tasks.word_sorting_mistake.scripts.word_sorting_mistake_verifier import WordSortingMistakeVerifier
from games.tasks.math_path.scripts.math_path_verifier import MathPathVerifier
from games.tasks.boolean_expressions.scripts.boolean_expressions_verifier import BooleanExpressionsVerifier
from games.tasks.space_reasoning.scripts.space_reasoning_verifier import SpaceReasoningVerifier
from games.tasks.space_reasoning_tree.scripts.space_reasoning_tree_verifier import SpaceReasoningTreeVerifier
from games.tasks.word_sorting.scripts.word_sorting_verifier import WordSortingVerifier
from games.tasks.cipher.scripts.cipher_verifier import CipherVerifier
from games.tasks.time_sequence.scripts.time_sequence_verifier import TimeSequenceVerifier
from games.tasks.wordscapes.scripts.wordscapes_verifier import WordscapesVerifier
from games.tasks.buggy_tables.scripts.game_of_buggy_tables_verifier import BuggyTableVerifier
from games.tasks.calcudoko.scripts.calcudoko_verifier import CalcudokoVerifier
from games.tasks.dyck_language.scripts.dyck_language_verifier import DyckLanguageVerifier
from games.tasks.dyck_language_errors.scripts.dyck_language_errors_verifier import DyckLanguageErrorsVerifier
from games.tasks.dyck_language_reasoning_errors.scripts.dyck_language_reasoning_errors_verifier import DyckLanguageReasoningErrorsVerifier
from games.tasks.futoshiki.scripts.futoshiki_verifier import FutoshikiVerifier

verifier_classes = {
    "arc_agi": ArcAGIVerifier,
    "arrow_maze": ArrowMazeVerifier,
    "boolean_expressions": BooleanExpressionsVerifier,
    "buggy_tables": BuggyTableVerifier,
    "calcudoko": CalcudokoVerifier,
    "campsite": CampsiteVerifier,
    "cipher": CipherVerifier,
    "cryptarithm": CryptarithmVerifier,
    "dyck_language": DyckLanguageVerifier,
    "dyck_language_errors": DyckLanguageErrorsVerifier,
    "dyck_language_reasoning_errors": DyckLanguageReasoningErrorsVerifier,
    "futoshiki": FutoshikiVerifier,
    "goods_exchange": GoodsExchangeVerifier,
    "kukurasu": KukurasuVerifier,
    "math_path": MathPathVerifier,
    "mathador": GameOf24Verifier,
    "minesweeper": MinesweeperVerifier,
    "norinori": NorinoriVerifier,
    "number_wall": NumberWallVerifier,
    "numbrix": NumbrixVerifier,
    "object_counting": ObjectCountingVerifier,
    "object_properties": ObjectPropertiesVerifier,
    "operation": OperationVerifier,
    "skyscraper_puzzle": SkyscraperPuzzleVerifier,
    "space_reasoning": SpaceReasoningVerifier,
    "space_reasoning_tree": SpaceReasoningTreeVerifier,
    "star_placement_puzzle": StarPlacementPuzzleVerifier,
    "sudoku": SudokuVerifier,
    "survo": SurvoVerifier,
    "time_sequence": TimeSequenceVerifier,
    "web_of_lies": WebOfLiesVerifier,
    "word_sorting": WordSortingVerifier,
    "word_sorting_mistake": WordSortingMistakeVerifier,
    "wordscapes": WordscapesVerifier,
    "zebra_puzzle": ZebraPuzzleVerifier,
}
