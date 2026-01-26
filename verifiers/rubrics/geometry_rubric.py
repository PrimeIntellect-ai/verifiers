"""
Geometry Rubric for verifiable geometric reasoning.

Provides four core verification primitives for Physical AI tasks:
- verify_no_overlap: Collision detection (bin packing, robot navigation)
- verify_contains: Containment check (nesting, geo-fencing)
- verify_connectivity: Path validation (maze solving, wire routing)
- calculate_iou: Intersection over Union (shape estimation)
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import TYPE_CHECKING

from shapely import wkt
from shapely.geometry import LineString, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.validation import make_valid

from verifiers.parsers.parser import Parser
from verifiers.rubrics.rubric import Rubric
from verifiers.types import Messages, RewardFunc

if TYPE_CHECKING:
    pass


class GeometryParser(Parser):
    """Parser for extracting geometry from model output.
    
    Supports:
    - WKT strings: POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))
    - Coordinate lists: [(0, 0), (1, 0), (1, 1), (0, 1)]
    - XML tags: <answer>POLYGON(...)</answer>
    - Code blocks: ```python ... ```
    """

    def __init__(
        self,
        geometry_tags: list[str] | None = None,
        extract_from_code: bool = True,
        variable_names: list[str] | None = None,
    ):
        """Initialize GeometryParser.
        
        Args:
            geometry_tags: XML tags to extract geometry from (default: ["answer", "geometry"])
            extract_from_code: Whether to try extracting from code blocks
            variable_names: Variable names to look for in code output (default: ["result", "path", "polygon"])
        """
        # Set instance vars first (needed by _extract_geometry)
        self.geometry_tags = geometry_tags or ["answer", "geometry", "solution"]
        self._extract_from_code = extract_from_code
        self.variable_names = variable_names or ["result", "path", "polygon", "shape", "coords"]
        # Pass extract method to parent
        super().__init__(extract_fn=self._extract_geometry)
    
    def _extract_geometry(self, text: str) -> str | None:
        """Extract geometry string from model output."""
        if not text:
            return None
        
        # Try XML tags first
        for tag in self.geometry_tags:
            pattern = rf"<{tag}>(.*?)</{tag}>"
            match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Try code blocks
        if self._extract_from_code:
            code_pattern = r"```(?:python)?\s*(.*?)```"
            matches = re.findall(code_pattern, text, re.DOTALL)
            for code in matches:
                # Look for variable assignments
                for var in self.variable_names:
                    var_pattern = rf"{var}\s*=\s*(.+?)(?:\n|$)"
                    var_match = re.search(var_pattern, code)
                    if var_match:
                        return var_match.group(1).strip()
        
        # Try to find WKT directly - use greedy matching for nested parentheses
        wkt_pattern = r"((?:POINT|LINESTRING|POLYGON|MULTIPOINT|MULTILINESTRING|MULTIPOLYGON)\s*\((?:[^()]*|\((?:[^()]*|\([^()]*\))*\))*\))"
        wkt_match = re.search(wkt_pattern, text, re.IGNORECASE)
        if wkt_match:
            return wkt_match.group(1).strip()
        
        # Try to find coordinate lists anywhere in text
        coord_pattern = r"\[\s*\(\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\)(?:\s*,\s*\(\s*-?\d+\.?\d*\s*,\s*-?\d+\.?\d*\s*\))*\s*\]"
        coord_match = re.search(coord_pattern, text)
        if coord_match:
            return coord_match.group(0).strip()
        
        return None


class GeometryRubric(Rubric):
    """Rubric for geometric verification using Shapely.
    
    Provides four core primitives for Physical AI tasks:
    - verify_no_overlap: Collision detection
    - verify_contains: Containment check  
    - verify_connectivity: Path validation
    - calculate_iou: Intersection over Union
    
    Example:
        rubric = vf.GeometryRubric()
        
        async def path_valid(geometry, completion, info) -> float:
            path = geometry.parse_geometry(completion)
            obstacles = info["obstacles"]
            return 1.0 if geometry.verify_no_overlap(path, obstacles) else 0.0
        
        rubric.add_reward_func(path_valid)
    """

    def __init__(
        self,
        funcs: list[RewardFunc] | None = None,
        weights: list[float] | None = None,
        parser: Parser | None = None,
        tolerance: float = 1e-9,
        auto_fix: bool = True,
    ):
        """Initialize GeometryRubric.
        
        Args:
            funcs: Additional reward functions
            weights: Weights for reward functions
            parser: Parser for extracting answers (default: GeometryParser)
            tolerance: Geometric tolerance for comparisons
            auto_fix: Whether to automatically fix invalid geometries
        """
        parser = parser or GeometryParser()
        super().__init__(funcs=funcs, weights=weights, parser=parser)
        self.tolerance = tolerance
        self.auto_fix = auto_fix
        self.logger = logging.getLogger(__name__)
        
        # Add self as class object so reward funcs can access methods
        self.add_class_object("geometry", self)

    def parse_geometry(
        self, 
        source: str | Messages | None,
        geometry_type: str | None = None,
    ) -> BaseGeometry | None:
        """Parse geometry from model output.
        
        Polymorphic parsing strategy:
        1. Try WKT first (most precise)
        2. Try JSON/coordinate list and infer type
        3. Return None if parsing fails
        
        Args:
            source: Raw text, messages, or extracted string
            geometry_type: Hint for type inference ("polygon", "linestring", "point")
                          Used when parsing coordinate lists
        
        Returns:
            Shapely geometry object or None if parsing fails
        """
        if source is None:
            return None
        
        # Extract text from messages if needed
        if isinstance(source, list):
            # Messages format - get last assistant content
            text = None
            for msg in reversed(source):
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    content = msg.get("content", "")
                    if isinstance(content, str):
                        text = content
                        break
            if text is None:
                return None
        else:
            text = source
        
        # Use parser to extract geometry string
        extracted = self.parser.extract_fn(text) if hasattr(self.parser, 'extract_fn') else text
        if extracted is None:
            extracted = text
        
        # Try WKT first
        geom = self._parse_wkt(extracted)
        if geom is not None:
            return self._maybe_fix(geom)
        
        # Try coordinate list
        geom = self._parse_coords(extracted, geometry_type)
        if geom is not None:
            return self._maybe_fix(geom)
        
        self.logger.debug(f"Failed to parse geometry from: {extracted[:100]}...")
        return None

    def _parse_wkt(self, text: str) -> BaseGeometry | None:
        """Parse WKT string."""
        try:
            # Clean up common issues
            cleaned = text.strip()
            return wkt.loads(cleaned)
        except Exception:
            return None

    def _parse_coords(
        self, 
        text: str, 
        geometry_type: str | None = None
    ) -> BaseGeometry | None:
        """Parse coordinate list and infer geometry type."""
        try:
            # Try to parse as Python literal
            coords = ast.literal_eval(text.strip())
        except Exception:
            try:
                # Try JSON
                coords = json.loads(text.strip())
            except Exception:
                return None
        
        if not isinstance(coords, list) or not coords:
            return None
        
        # Infer geometry type
        if geometry_type:
            gtype = geometry_type.lower()
        else:
            gtype = self._infer_geometry_type(coords)
        
        try:
            if gtype == "point":
                if isinstance(coords[0], (int, float)):
                    return Point(coords)
                return Point(coords[0])
            elif gtype == "linestring":
                return LineString(coords)
            elif gtype == "polygon":
                return Polygon(coords)
            else:
                # Default: polygon if closed, linestring otherwise
                if len(coords) >= 3 and coords[0] == coords[-1]:
                    return Polygon(coords)
                elif len(coords) >= 2:
                    return LineString(coords)
                else:
                    return Point(coords[0] if coords else (0, 0))
        except Exception as e:
            self.logger.debug(f"Failed to create geometry: {e}")
            return None

    def _infer_geometry_type(self, coords: list) -> str:
        """Infer geometry type from coordinate structure."""
        if not coords:
            return "point"
        
        # Single point: [x, y] or [(x, y)]
        if isinstance(coords[0], (int, float)):
            return "point"
        if len(coords) == 1:
            return "point"
        
        # Check if closed (first == last) -> polygon
        try:
            if len(coords) >= 4 and tuple(coords[0]) == tuple(coords[-1]):
                return "polygon"
        except (TypeError, IndexError):
            pass
        
        # Multiple points -> linestring
        return "linestring"

    def _maybe_fix(self, geom: BaseGeometry) -> BaseGeometry:
        """Fix invalid geometry if auto_fix is enabled."""
        if self.auto_fix and not geom.is_valid:
            try:
                return make_valid(geom)
            except Exception:
                pass
        return geom

    def verify_no_overlap(
        self,
        geometries: list[BaseGeometry],
        tolerance: float | None = None,
    ) -> bool:
        """Verify no two geometries overlap (collision detection).
        
        Use cases: Bin packing, PCB routing, robot arm navigation.
        
        Args:
            geometries: List of geometries to check pairwise
            tolerance: Area threshold for overlap (default: self.tolerance)
        
        Returns:
            True if no overlaps, False if any pair overlaps
        """
        tol = tolerance if tolerance is not None else self.tolerance
        
        for i in range(len(geometries)):
            for j in range(i + 1, len(geometries)):
                a, b = geometries[i], geometries[j]
                if a is None or b is None:
                    continue
                try:
                    intersection = a.intersection(b)
                    # Check area for polygon overlaps
                    if intersection.area > tol:
                        return False
                    # For LineStrings/Points (zero area geometries), check if they
                    # intersect, ignore boundary-only contact between polygons
                    if not intersection.is_empty and (a.area == 0 or b.area == 0):
                        return False
                except Exception as e:
                    self.logger.debug(f"Overlap check failed: {e}")
                    return False
        return True

    def verify_contains(
        self,
        container: BaseGeometry,
        items: BaseGeometry | list[BaseGeometry],
    ) -> bool:
        """Verify container fully contains all items.
        
        Use cases: Nesting, 3D printing slicing, geo-fencing.
        
        Args:
            container: The containing geometry
            items: Single geometry or list of geometries to check
        
        Returns:
            True if all items are fully contained, False otherwise
        """
        if container is None:
            return False
        
        if not isinstance(items, list):
            items = [items]
        
        for item in items:
            if item is None:
                continue
            try:
                if not container.contains(item):
                    return False
            except Exception as e:
                self.logger.debug(f"Containment check failed: {e}")
                return False
        return True

    def verify_connectivity(
        self,
        path: BaseGeometry,
        start: tuple[float, float] | Point | None = None,
        end: tuple[float, float] | Point | None = None,
        tolerance: float | None = None,
    ) -> bool:
        """Verify path is valid and connects start to end.
        
        Use cases: Maze solving, wire routing, motion planning.
        
        Checks:
        1. Path is a valid LineString
        2. Path is simple (no self-intersection)
        3. Path connects start and end points (if provided)
        
        Args:
            path: LineString geometry representing the path
            start: Start point (optional)
            end: End point (optional)
            tolerance: Distance tolerance for endpoint matching
        
        Returns:
            True if path is valid and connected, False otherwise
        """
        if path is None:
            return False
        
        tol = tolerance if tolerance is not None else self.tolerance * 1000  # Larger default for points
        
        # Must be a LineString
        if not isinstance(path, LineString):
            return False
        
        # Must be valid
        if not path.is_valid:
            return False
        
        # Must be simple (no self-intersection)
        if not path.is_simple:
            return False
        
        # Must not be empty (prevents IndexError and rejects degenerate paths)
        if path.is_empty:
            return False
        
        # Check start point
        if start is not None:
            start_pt = Point(start) if not isinstance(start, Point) else start
            path_start = Point(path.coords[0])
            if start_pt.distance(path_start) > tol:
                return False
        
        # Check end point
        if end is not None:
            end_pt = Point(end) if not isinstance(end, Point) else end
            path_end = Point(path.coords[-1])
            if end_pt.distance(path_end) > tol:
                return False
        
        return True

    def calculate_iou(
        self,
        predicted: BaseGeometry,
        ground_truth: BaseGeometry,
    ) -> float:
        """Calculate Intersection over Union (IoU).
        
        Use cases: Shape estimation, VLA model evaluation.
        
        Args:
            predicted: Predicted geometry
            ground_truth: Ground truth geometry
        
        Returns:
            IoU score between 0.0 and 1.0
        """
        if predicted is None or ground_truth is None:
            return 0.0
        
        try:
            intersection = predicted.intersection(ground_truth)
            union = predicted.union(ground_truth)
            
            if union.area == 0:
                return 0.0
            
            return intersection.area / union.area
        except Exception as e:
            self.logger.debug(f"IoU calculation failed: {e}")
            return 0.0
