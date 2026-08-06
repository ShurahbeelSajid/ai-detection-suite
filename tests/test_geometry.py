import unittest

from ai_detection_suite.geometry import (
    Box,
    associate_face_attributes,
    intersection_over_union,
    scale_box_between,
)


class BoxTests(unittest.TestCase):
    def test_clip_is_immutable_and_handles_fully_outside_box(self):
        original = Box(-5, 4, 15, 30)
        clipped = original.clip(10, 20)
        self.assertEqual(original, Box(-5, 4, 15, 30))
        self.assertEqual(clipped, Box(0, 4, 10, 20))
        self.assertEqual(Box(20, 2, 30, 5).clip(10, 10), Box(10, 2, 10, 5))

    def test_scale_uses_independent_axes(self):
        self.assertEqual(
            Box(10, 20, 30, 40).scale(2.0, 0.5),
            Box(20, 10, 60, 20),
        )
        self.assertEqual(
            scale_box_between(
                Box(10, 10, 20, 20),
                source_width=160,
                source_height=120,
                target_width=960,
                target_height=540,
            ),
            Box(60, 45, 120, 90),
        )

    def test_iou_for_overlap_disjoint_and_empty_boxes(self):
        self.assertAlmostEqual(
            intersection_over_union(Box(0, 0, 10, 10), Box(5, 5, 15, 15)),
            1.0 / 7.0,
        )
        self.assertEqual(
            intersection_over_union(Box(0, 0, 1, 1), Box(2, 2, 3, 3)),
            0.0,
        )
        self.assertEqual(
            intersection_over_union(Box(1, 1, 1, 1), Box(1, 1, 1, 1)),
            0.0,
        )

    def test_invalid_geometry_is_rejected(self):
        with self.assertRaises(ValueError):
            Box(2, 0, 1, 1)
        with self.assertRaises(ValueError):
            Box(0, 0, 1, 1).scale(-1, 1)
        with self.assertRaises(ValueError):
            scale_box_between(
                Box(0, 0, 1, 1),
                source_width=0,
                source_height=1,
                target_width=1,
                target_height=1,
            )


class AssociationTests(unittest.TestCase):
    def test_attributes_are_matched_by_geometry_not_list_order(self):
        faces = [Box(0, 0, 10, 10), Box(30, 0, 40, 10), Box(60, 0, 70, 10)]
        attributes = [Box(31, 0, 41, 10), Box(1, 0, 11, 10), Box(90, 0, 95, 5)]

        association = associate_face_attributes(faces, attributes, min_iou=0.5)

        self.assertEqual(
            [(match.face_index, match.attribute_index) for match in association.matches],
            [(0, 1), (1, 0)],
        )
        self.assertEqual(association.unmatched_face_indices, (2,))
        self.assertEqual(association.unmatched_attribute_indices, (2,))

    def test_association_is_one_to_one(self):
        association = associate_face_attributes(
            [Box(0, 0, 10, 10), Box(1, 0, 11, 10)],
            [Box(0, 0, 10, 10)],
            min_iou=0.1,
        )
        self.assertEqual(len(association.matches), 1)
        self.assertEqual(association.matches[0].face_index, 0)
        self.assertEqual(association.unmatched_face_indices, (1,))


if __name__ == "__main__":
    unittest.main()
