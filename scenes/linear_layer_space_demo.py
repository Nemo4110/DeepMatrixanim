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
    MathTex,
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


def point_to_scene(plane: NumberPlane, point: np.ndarray) -> np.ndarray:
    return plane.c2p(float(point[0]), float(point[1]))


def vector_arrow(plane: NumberPlane, point: np.ndarray, color=BLUE) -> Arrow:
    return Arrow(start=plane.c2p(0, 0), end=point_to_scene(plane, point), buff=0, color=color)


def dots_for(plane: NumberPlane, points: np.ndarray, color=YELLOW) -> VGroup:
    return VGroup(*[Dot(point_to_scene(plane, point), radius=0.055, color=color) for point in points])


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
        formula = MathTex(r"z = Wx + b").scale(0.55).next_to(title, DOWN, buff=0.16)
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

        original_vector = vector_arrow(plane, x, BLUE)
        linear_vector = vector_arrow(plane, wx, GREEN)
        affine_vector = Arrow(
            start=point_to_scene(plane, bias),
            end=point_to_scene(plane, z),
            buff=0,
            color=PURPLE,
        )
        bias_arrow = vector_arrow(plane, bias, ORANGE)

        original_dots = dots_for(plane, points, YELLOW)
        linear_dots = dots_for(plane, linear_points, GREEN)
        affine_dots = dots_for(plane, affine_points, PURPLE)

        w_label = MathTex(r"W = \begin{bmatrix}1.2 & 0.6 \\ -0.4 & 1.1\end{bmatrix}").scale(0.32)
        b_label = MathTex(r"b = \begin{bmatrix}0.8 \\ -0.35\end{bmatrix}").scale(0.32)
        VGroup(w_label, b_label).arrange(RIGHT, buff=0.55).to_edge(DOWN)

        self.play(Write(title), Write(formula), FadeIn(note))
        self.play(Create(plane), FadeIn(original_dots), Create(original_vector), FadeIn(w_label))
        self.wait(0.4)
        self.play(Transform(original_dots, linear_dots), Transform(original_vector, linear_vector), run_time=1.6)
        self.wait(0.4)
        self.play(Create(bias_arrow), FadeIn(b_label))
        self.play(Transform(original_dots, affine_dots), Transform(original_vector, affine_vector), run_time=1.3)
        self.wait(0.6)
