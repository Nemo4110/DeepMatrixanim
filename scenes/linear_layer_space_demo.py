from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    ORANGE,
    PURPLE,
    RIGHT,
    UP,
    YELLOW,
    Arrow,
    Create,
    Dot,
    FadeIn,
    NumberPlane,
    Scene,
    Text,
    Transform,
    VGroup,
    Write,
)

from deepmatrixanim.linear_maps import (
    affine_transform,
    demo_bias_vector,
    demo_input_vector,
    demo_linear_matrix,
    demo_point_cloud,
)


def point_to_scene(point: np.ndarray) -> np.ndarray:
    return np.array([point[0], point[1], 0.0])


def vector_arrow(point: np.ndarray, color=BLUE) -> Arrow:
    return Arrow(start=np.zeros(3), end=point_to_scene(point), buff=0, color=color)


def dots_for(points: np.ndarray, color=YELLOW) -> VGroup:
    return VGroup(*[Dot(point_to_scene(point), radius=0.055, color=color) for point in points])


class LinearLayerAsSpaceTransform(Scene):
    def construct(self):
        matrix = demo_linear_matrix()
        bias = demo_bias_vector()
        x = demo_input_vector()

        wx = affine_transform(x, matrix, np.zeros(2))
        z = affine_transform(x, matrix, bias)
        points = demo_point_cloud()
        linear_points = affine_transform(points, matrix, np.zeros(2))
        affine_points = affine_transform(points, matrix, bias)

        title = Text("Linear layer as a space transform").scale(0.52).to_edge(UP)
        formula = Text("z = W x + b").scale(0.55).next_to(title, DOWN, buff=0.16)
        note = Text("reshape directions, then shift origin").scale(0.32).next_to(formula, DOWN, buff=0.12)

        plane = NumberPlane(
            x_range=[-3.5, 4.5, 1],
            y_range=[-3.0, 3.0, 1],
            x_length=8.0,
            y_length=5.2,
            background_line_style={
                "stroke_color": GREY_B,
                "stroke_width": 1,
                "stroke_opacity": 0.45,
            },
        ).shift(DOWN * 0.45)

        plane_center = plane.get_center()
        original_vector = vector_arrow(x, BLUE).shift(plane_center)
        linear_vector = vector_arrow(wx, GREEN).shift(plane_center)
        affine_vector = Arrow(
            start=point_to_scene(bias) + plane_center,
            end=point_to_scene(z) + plane_center,
            buff=0,
            color=PURPLE,
        )
        bias_arrow = vector_arrow(bias, ORANGE).shift(plane_center)

        original_dots = dots_for(points, YELLOW).shift(plane_center)
        linear_dots = dots_for(linear_points, GREEN).shift(plane_center)
        affine_dots = dots_for(affine_points, PURPLE).shift(plane_center)

        w_label = Text("W = [[1.2, 0.6], [-0.4, 1.1]]").scale(0.32)
        b_label = Text("b = [0.8, -0.35]").scale(0.32)
        VGroup(w_label, b_label).arrange(RIGHT, buff=0.55).to_edge(DOWN)

        self.play(Write(title), Write(formula), FadeIn(note))
        self.play(Create(plane), FadeIn(original_dots), Create(original_vector), FadeIn(w_label))
        self.wait(0.4)
        self.play(Transform(original_dots, linear_dots), Transform(original_vector, linear_vector), run_time=1.6)
        self.wait(0.4)
        self.play(Create(bias_arrow), FadeIn(b_label))
        self.play(Transform(original_dots, affine_dots), Transform(original_vector, affine_vector), run_time=1.3)
        self.wait(0.6)
