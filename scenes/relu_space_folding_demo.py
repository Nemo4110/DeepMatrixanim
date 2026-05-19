from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    ORANGE,
    RED,
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

from deepmatrixanim.linear_maps import demo_point_cloud, relu_transform


def point_to_scene(plane: NumberPlane, point: np.ndarray) -> np.ndarray:
    return plane.c2p(float(point[0]), float(point[1]))


def dots_for(plane: NumberPlane, points: np.ndarray, color=YELLOW) -> VGroup:
    return VGroup(*[Dot(point_to_scene(plane, point), radius=0.06, color=color) for point in points])


class ReLUAsSpaceFolding(Scene):
    def construct(self):
        points = demo_point_cloud()
        relu_points = relu_transform(points)

        title = Text("ReLU as space folding").scale(0.56).to_edge(UP)
        formula = MathTex(r"a = \mathrm{ReLU}(z) = \max(0, z)").scale(0.58).next_to(title, DOWN, buff=0.14)
        message = Text("negative coordinates clamp onto the axes").scale(0.32).next_to(formula, DOWN, buff=0.1)

        plane = NumberPlane(
            x_range=[-2.6, 3.2, 1],
            y_range=[-2.4, 2.8, 1],
            x_length=7.8,
            y_length=5.0,
            background_line_style={
                "stroke_color": GREY_B,
                "stroke_width": 1,
                "stroke_opacity": 0.45,
            },
        ).shift(DOWN * 0.5)

        original_dots = dots_for(plane, points, YELLOW)
        folded_dots = dots_for(plane, relu_points, GREEN)

        x_axis_arrow = Arrow(
            start=plane.c2p(-2.35, 0.0),
            end=plane.c2p(2.95, 0.0),
            buff=0,
            color=BLUE,
            stroke_width=4,
        )
        y_axis_arrow = Arrow(
            start=plane.c2p(0.0, -2.15),
            end=plane.c2p(0.0, 2.55),
            buff=0,
            color=BLUE,
            stroke_width=4,
        )

        clamp_x = MathTex(r"z_x < 0 \Rightarrow a_x = 0").set_color(RED).scale(0.43)
        clamp_y = MathTex(r"z_y < 0 \Rightarrow a_y = 0").set_color(ORANGE).scale(0.43)
        VGroup(clamp_y, clamp_x).arrange(DOWN, aligned_edge=LEFT, buff=0.14).to_corner(LEFT + DOWN)

        legend_original = VGroup(Dot(radius=0.055, color=YELLOW), MathTex(r"z").scale(0.42)).arrange(RIGHT, buff=0.12)
        legend_folded = VGroup(Dot(radius=0.055, color=GREEN), MathTex(r"a=\mathrm{ReLU}(z)").scale(0.42)).arrange(
            RIGHT,
            buff=0.12,
        )
        VGroup(legend_original, legend_folded).arrange(DOWN, aligned_edge=LEFT, buff=0.16).to_corner(RIGHT + DOWN)

        final = Text("coordinate-wise: fold left/bottom half-planes to zero").scale(0.34).to_edge(DOWN)

        self.play(Write(title), Write(formula), FadeIn(message))
        self.play(Create(plane), Create(x_axis_arrow), Create(y_axis_arrow), FadeIn(original_dots), FadeIn(legend_original))
        self.wait(0.4)
        self.play(FadeIn(clamp_x), FadeIn(clamp_y))
        self.play(Transform(original_dots, folded_dots), FadeIn(legend_folded), run_time=1.8)
        self.wait(0.4)
        self.play(Write(final))
        self.wait(0.8)
