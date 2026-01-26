"""
Tests for GeometryRubric - geometric verification primitives.
"""

from __future__ import annotations

import pytest
from shapely.geometry import LineString, Point, Polygon

from verifiers.rubrics.geometry_rubric import GeometryParser, GeometryRubric


class TestGeometryParser:
    """Tests for GeometryParser extraction."""

    def test_extract_wkt_from_text(self):
        """Parser extracts WKT from raw text."""
        parser = GeometryParser()
        text = "Here is my answer: POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"
        result = parser.extract_fn(text)
        assert result == "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"

    def test_extract_from_xml_tag(self):
        """Parser extracts from <answer> tags."""
        parser = GeometryParser()
        text = "The solution is <answer>POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))</answer>."
        result = parser.extract_fn(text)
        assert result == "POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))"

    def test_extract_from_geometry_tag(self):
        """Parser extracts from <geometry> tags."""
        parser = GeometryParser()
        text = "<geometry>[(0, 0), (1, 0), (1, 1)]</geometry>"
        result = parser.extract_fn(text)
        assert result == "[(0, 0), (1, 0), (1, 1)]"

    def test_extract_coordinate_list(self):
        """Parser extracts coordinate lists."""
        parser = GeometryParser()
        text = "The path is [(0, 0), (1, 0), (1, 1)]."
        result = parser.extract_fn(text)
        assert result == "[(0, 0), (1, 0), (1, 1)]"

    def test_extract_from_code_block(self):
        """Parser extracts from code blocks."""
        parser = GeometryParser()
        text = """
```python
x = 10
result = [(0, 0), (1, 0), (1, 1)]
```
"""
        result = parser.extract_fn(text)
        assert result == "[(0, 0), (1, 0), (1, 1)]"

    def test_returns_none_for_no_match(self):
        """Parser returns None when no geometry found."""
        parser = GeometryParser()
        text = "There is no geometry here."
        result = parser.extract_fn(text)
        assert result is None


class TestGeometryRubricParsing:
    """Tests for GeometryRubric.parse_geometry()."""

    def test_parse_wkt_polygon(self):
        """Parse WKT polygon."""
        rubric = GeometryRubric()
        geom = rubric.parse_geometry("POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))")
        assert isinstance(geom, Polygon)
        assert geom.is_valid
        assert geom.area == pytest.approx(1.0)

    def test_parse_wkt_linestring(self):
        """Parse WKT linestring."""
        rubric = GeometryRubric()
        geom = rubric.parse_geometry("LINESTRING(0 0, 1 0, 1 1)")
        assert isinstance(geom, LineString)
        assert geom.is_valid

    def test_parse_wkt_point(self):
        """Parse WKT point."""
        rubric = GeometryRubric()
        geom = rubric.parse_geometry("POINT(1 2)")
        assert isinstance(geom, Point)
        assert geom.x == 1
        assert geom.y == 2

    def test_parse_coord_list_polygon(self):
        """Parse coordinate list as polygon (closed)."""
        rubric = GeometryRubric()
        geom = rubric.parse_geometry("[(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)]")
        assert isinstance(geom, Polygon)

    def test_parse_coord_list_linestring(self):
        """Parse coordinate list as linestring (open)."""
        rubric = GeometryRubric()
        geom = rubric.parse_geometry("[(0, 0), (1, 0), (1, 1)]")
        assert isinstance(geom, LineString)

    def test_parse_with_geometry_type_hint(self):
        """Geometry type hint overrides inference."""
        rubric = GeometryRubric()
        geom = rubric.parse_geometry("[(0, 0), (1, 0), (1, 1)]", geometry_type="polygon")
        assert isinstance(geom, Polygon)

    def test_parse_from_messages(self):
        """Parse from chat messages format."""
        rubric = GeometryRubric()
        messages = [
            {"role": "user", "content": "Draw a square"},
            {"role": "assistant", "content": "<answer>POLYGON((0 0, 1 0, 1 1, 0 1, 0 0))</answer>"},
        ]
        geom = rubric.parse_geometry(messages)
        assert isinstance(geom, Polygon)

    def test_parse_invalid_returns_none(self):
        """Invalid input returns None."""
        rubric = GeometryRubric()
        assert rubric.parse_geometry("not a geometry") is None
        assert rubric.parse_geometry(None) is None
        assert rubric.parse_geometry("") is None


class TestVerifyNoOverlap:
    """Tests for verify_no_overlap (collision detection)."""

    def test_no_overlap_separate_polygons(self):
        """Non-overlapping polygons pass."""
        rubric = GeometryRubric()
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        assert rubric.verify_no_overlap([p1, p2]) is True

    def test_no_overlap_touching_polygons(self):
        """Edge-touching polygons pass (0 area intersection)."""
        rubric = GeometryRubric()
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(1, 0), (2, 0), (2, 1), (1, 1)])  # Shares edge
        assert rubric.verify_no_overlap([p1, p2]) is True

    def test_overlapping_polygons_fail(self):
        """Overlapping polygons fail."""
        rubric = GeometryRubric()
        p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])
        p2 = Polygon([(1, 1), (3, 1), (3, 3), (1, 3)])  # Overlaps
        assert rubric.verify_no_overlap([p1, p2]) is False

    def test_no_overlap_multiple_geometries(self):
        """Multiple pairwise checks work."""
        rubric = GeometryRubric()
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(2, 0), (3, 0), (3, 1), (2, 1)])
        p3 = Polygon([(4, 0), (5, 0), (5, 1), (4, 1)])
        assert rubric.verify_no_overlap([p1, p2, p3]) is True

    def test_empty_list_passes(self):
        """Empty list passes."""
        rubric = GeometryRubric()
        assert rubric.verify_no_overlap([]) is True

    def test_linestring_through_polygon_fails(self):
        """LineString passing through a polygon is detected as collision."""
        rubric = GeometryRubric()
        path = LineString([(0, 0), (10, 10)])
        obstacle = Polygon([(3, 3), (7, 3), (7, 7), (3, 7)])
        assert rubric.verify_no_overlap([path, obstacle]) is False

    def test_linestring_separate_from_polygon_passes(self):
        """LineString not touching polygon passes."""
        rubric = GeometryRubric()
        path = LineString([(0, 0), (1, 1)])
        obstacle = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        assert rubric.verify_no_overlap([path, obstacle]) is True

    def test_intersecting_linestrings_fail(self):
        """Two intersecting LineStrings are detected as collision."""
        rubric = GeometryRubric()
        line1 = LineString([(0, 0), (2, 2)])
        line2 = LineString([(0, 2), (2, 0)])
        assert rubric.verify_no_overlap([line1, line2]) is False


class TestVerifyContains:
    """Tests for verify_contains (containment check)."""

    def test_container_contains_item(self):
        """Container fully contains item."""
        rubric = GeometryRubric()
        container = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        item = Polygon([(1, 1), (2, 1), (2, 2), (1, 2)])
        assert rubric.verify_contains(container, item) is True

    def test_container_not_contains_item(self):
        """Container does not contain item (partial overlap)."""
        rubric = GeometryRubric()
        container = Polygon([(0, 0), (5, 0), (5, 5), (0, 5)])
        item = Polygon([(4, 4), (6, 4), (6, 6), (4, 6)])  # Sticks out
        assert rubric.verify_contains(container, item) is False

    def test_container_contains_multiple_items(self):
        """Container contains list of items."""
        rubric = GeometryRubric()
        container = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        items = [
            Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]),
            Polygon([(3, 3), (4, 3), (4, 4), (3, 4)]),
        ]
        assert rubric.verify_contains(container, items) is True

    def test_container_contains_point(self):
        """Container contains point."""
        rubric = GeometryRubric()
        container = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        point = Point(5, 5)
        assert rubric.verify_contains(container, point) is True

    def test_container_not_contains_outside_point(self):
        """Container does not contain outside point."""
        rubric = GeometryRubric()
        container = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)])
        point = Point(15, 15)
        assert rubric.verify_contains(container, point) is False


class TestVerifyConnectivity:
    """Tests for verify_connectivity (path validation)."""

    def test_valid_simple_path(self):
        """Valid simple path passes."""
        rubric = GeometryRubric()
        path = LineString([(0, 0), (1, 0), (1, 1)])
        assert rubric.verify_connectivity(path) is True

    def test_non_linestring_fails(self):
        """Non-LineString geometry fails."""
        rubric = GeometryRubric()
        polygon = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert rubric.verify_connectivity(polygon) is False

    def test_self_intersecting_path_fails(self):
        """Self-intersecting path fails."""
        rubric = GeometryRubric()
        # Figure-8 shape
        path = LineString([(0, 0), (2, 2), (2, 0), (0, 2)])
        assert rubric.verify_connectivity(path) is False

    def test_empty_path_fails(self):
        """Empty LineString fails instead of raising IndexError."""
        rubric = GeometryRubric()
        path = LineString()
        assert rubric.verify_connectivity(path) is False
        assert rubric.verify_connectivity(path, start=(0, 0), end=(1, 1)) is False

    def test_path_connects_start_end(self):
        """Path connecting start and end passes."""
        rubric = GeometryRubric()
        path = LineString([(0, 0), (1, 0), (2, 0), (2, 1)])
        assert rubric.verify_connectivity(path, start=(0, 0), end=(2, 1)) is True

    def test_path_wrong_start_fails(self):
        """Path with wrong start point fails."""
        rubric = GeometryRubric()
        path = LineString([(0, 0), (1, 0), (2, 0)])
        assert rubric.verify_connectivity(path, start=(5, 5), end=(2, 0)) is False

    def test_path_wrong_end_fails(self):
        """Path with wrong end point fails."""
        rubric = GeometryRubric()
        path = LineString([(0, 0), (1, 0), (2, 0)])
        assert rubric.verify_connectivity(path, start=(0, 0), end=(5, 5)) is False


class TestCalculateIoU:
    """Tests for calculate_iou."""

    def test_perfect_overlap(self):
        """Identical shapes have IoU = 1.0."""
        rubric = GeometryRubric()
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert rubric.calculate_iou(p1, p2) == pytest.approx(1.0)

    def test_no_overlap(self):
        """Non-overlapping shapes have IoU = 0.0."""
        rubric = GeometryRubric()
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        p2 = Polygon([(5, 5), (6, 5), (6, 6), (5, 6)])
        assert rubric.calculate_iou(p1, p2) == pytest.approx(0.0)

    def test_partial_overlap(self):
        """Partial overlap gives 0 < IoU < 1."""
        rubric = GeometryRubric()
        p1 = Polygon([(0, 0), (2, 0), (2, 2), (0, 2)])  # Area = 4
        p2 = Polygon([(1, 0), (3, 0), (3, 2), (1, 2)])  # Area = 4, overlaps 2
        # Intersection = 2, Union = 6, IoU = 2/6 = 1/3
        assert rubric.calculate_iou(p1, p2) == pytest.approx(1 / 3, rel=0.01)

    def test_none_geometry_returns_zero(self):
        """None geometry returns 0.0."""
        rubric = GeometryRubric()
        p1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        assert rubric.calculate_iou(None, p1) == 0.0
        assert rubric.calculate_iou(p1, None) == 0.0
        

class TestGeometryRubricIntegration:
    """Integration tests for GeometryRubric with reward functions."""

    @pytest.mark.asyncio
    async def test_reward_function_with_geometry(self):
        """Reward function can use geometry verification."""
        rubric = GeometryRubric()

        async def path_valid(geometry, completion, info) -> float:
            path = geometry.parse_geometry(completion)
            if path is None:
                return 0.0
            obstacles = [geometry.parse_geometry(o) for o in info["obstacles"]]
            return 1.0 if geometry.verify_no_overlap([path, *obstacles]) else 0.0

        rubric.add_reward_func(path_valid)

        # Test with valid path (no collision)
        messages = [{"role": "assistant", "content": "<answer>LINESTRING(0 0, 0 5)</answer>"}]
        info = {"obstacles": ["POLYGON((2 2, 3 2, 3 3, 2 3, 2 2))"]}

        # Create minimal state
        from verifiers.types import State

        state = State()
        state["completion"] = messages
        state["info"] = info

        # Note: Full scoring would require more setup; this tests the primitives
        path = rubric.parse_geometry(messages)
        obstacles = [rubric.parse_geometry(o) for o in info["obstacles"]]
        result = rubric.verify_no_overlap([path, *obstacles])
        assert result is True
