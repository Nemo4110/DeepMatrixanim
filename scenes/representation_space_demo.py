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
    Arrow,
    Circumscribe,
    Create,
    DashedLine,
    Dot,
    FadeIn,
    MathTex,
    NumberPlane,
    Rectangle,
    Scene,
    Text,
    Transform,
    TransformFromCopy,
    VGroup,
    Write,
)

from deepmatrixanim.decision_boundaries import (
    demo_decision_boundary_points,
    hidden_activation,
    hidden_preactivation,
    representation_space_transform,
)


def point_to_scene(plane: NumberPlane, point: np.ndarray) -> np.ndarray:
    return plane.c2p(float(point[0]), float(point[1]))


def labeled_dots(plane: NumberPlane, points: np.ndarray, labels: np.ndarray) -> VGroup:
    dots = []
    for point, label in zip(points, labels):
        color = BLUE if int(label) == 1 else ORANGE
        dots.append(Dot(point_to_scene(plane, point), radius=0.065, color=color))
    return VGroup(*dots)


def hidden_score_bars(values: np.ndarray, labels: np.ndarray, relu_stage: bool = False) -> VGroup:
    group = VGroup()
    max_abs = max(1.0, float(np.max(np.abs(values))))
    row_gap = 0.34
    col_gap = 0.38
    max_bar_height = 0.25

    for row_index, (row, label) in enumerate(zip(values, labels)):
        y = (len(values) - 1) * row_gap / 2 - row_index * row_gap
        class_dot = Dot(
            point=np.array([-0.98, y, 0.0]),
            radius=0.04,
            color=BLUE if int(label) == 1 else ORANGE,
        )
        group.add(class_dot)
        for col_index, value in enumerate(row):
            x = (col_index - 1.5) * col_gap
            baseline = Rectangle(width=0.18, height=0.012, color=GREY_B, fill_opacity=0.55, stroke_width=0)
            baseline.move_to(np.array([x, y, 0.0]))
            group.add(baseline)

            magnitude = abs(float(value))
            height = max(0.018, magnitude / max_abs * max_bar_height)
            color = GREEN if value > 0 else RED
            opacity = 0.82 if value > 0 else 0.36
            if relu_stage and value <= 0:
                color = GREY_B
                opacity = 0.35
                height = 0.018

            bar = Rectangle(width=0.105, height=height, color=color, fill_color=color, fill_opacity=opacity, stroke_width=0)
            offset = height / 2 if value > 0 else -height / 2
            if relu_stage and value <= 0:
                offset = 0.0
            bar.move_to(np.array([x, y + offset, 0.0]))
            group.add(bar)

    return group


class RepresentationSpaceDecisionBoundary(Scene):
    def construct(self):
        points, labels = demo_decision_boundary_points()
        preactivation = hidden_preactivation(points)
        activation = hidden_activation(points)
        represented = representation_space_transform(points)

        title = Text("Representation space makes XOR linearly separable").scale(0.46).to_edge(UP)
        formula = MathTex(
            r"x",
            r"\xrightarrow{W_1}",
            r"h",
            r"\xrightarrow{\mathrm{ReLU}}",
            r"a",
            r"\xrightarrow{W_2}",
            r"r",
        ).scale(0.55).next_to(title, DOWN, buff=0.14)

        input_plane = NumberPlane(
            x_range=[-2.2, 2.2, 1],
            y_range=[-2.0, 2.0, 1],
            x_length=3.75,
            y_length=3.5,
            background_line_style={
                "stroke_color": GREY_B,
                "stroke_width": 1,
                "stroke_opacity": 0.42,
            },
        ).shift(LEFT * 4.4 + DOWN * 0.55)
        representation_plane = NumberPlane(
            x_range=[0.0, 3.0, 1],
            y_range=[0.0, 3.0, 1],
            x_length=3.55,
            y_length=3.5,
            background_line_style={
                "stroke_color": GREY_B,
                "stroke_width": 1,
                "stroke_opacity": 0.42,
            },
        ).shift(RIGHT * 4.25 + DOWN * 0.55)

        input_label = Text("input space").scale(0.3).next_to(input_plane, UP, buff=0.12)
        hidden_label = Text("hidden scores").scale(0.3).move_to(UP * 1.55)
        relu_label = Text("ReLU clamps negatives to zero").scale(0.26).next_to(hidden_label, DOWN, buff=0.08)
        representation_label = Text("learned representation").scale(0.3).next_to(
            representation_plane,
            UP,
            buff=0.12,
        )

        input_dots = labeled_dots(input_plane, points, labels)
        preactivation_bars = hidden_score_bars(preactivation, labels).shift(DOWN * 0.62)
        activation_bars = hidden_score_bars(activation, labels, relu_stage=True).shift(DOWN * 0.62)
        represented_dots = labeled_dots(representation_plane, represented, labels)

        w1_arrow = Arrow(
            input_plane.get_right() + RIGHT * 0.22,
            LEFT * 1.25 + DOWN * 0.55,
            buff=0.12,
            color=GREEN,
        )
        w1_label = MathTex(r"W_1x").scale(0.35).next_to(w1_arrow, UP, buff=0.08)
        w2_arrow = Arrow(
            RIGHT * 1.25 + DOWN * 0.55,
            representation_plane.get_left() + LEFT * 0.22,
            buff=0.12,
            color=GREEN,
        )
        w2_label = MathTex(r"W_2a").scale(0.35).next_to(w2_arrow, UP, buff=0.08)

        split_line = DashedLine(
            representation_plane.c2p(0.0, 0.0),
            representation_plane.c2p(2.85, 2.85),
            color=GREEN,
            dash_length=0.12,
        )
        split_label = MathTex(r"r_1 = r_2").scale(0.36).next_to(split_line, RIGHT, buff=0.1)
        positive_side = MathTex(r"r_1 > r_2").set_color(BLUE).scale(0.34)
        positive_side.move_to(representation_plane.c2p(2.15, 0.65))
        negative_side = MathTex(r"r_2 > r_1").set_color(ORANGE).scale(0.34)
        negative_side.move_to(representation_plane.c2p(0.75, 2.2))

        legend_positive = VGroup(Dot(radius=0.055, color=BLUE), MathTex(r"x_1x_2>0").scale(0.35)).arrange(
            RIGHT,
            buff=0.12,
        )
        legend_negative = VGroup(Dot(radius=0.055, color=ORANGE), MathTex(r"x_1x_2<0").scale(0.35)).arrange(
            RIGHT,
            buff=0.12,
        )
        VGroup(legend_positive, legend_negative).arrange(DOWN, aligned_edge=LEFT, buff=0.14).to_edge(DOWN)

        self.play(Write(title), Write(formula))
        self.play(Create(input_plane), Write(input_label), FadeIn(input_dots), FadeIn(legend_positive), FadeIn(legend_negative))
        self.wait(0.35)
        self.play(Circumscribe(formula[1], color=GREEN), Create(w1_arrow), Write(w1_label))
        self.play(Write(hidden_label), FadeIn(preactivation_bars), run_time=1.2)
        self.wait(0.25)
        self.play(Circumscribe(formula[3], color=GREEN), Write(relu_label))
        self.play(Transform(preactivation_bars, activation_bars), run_time=1.4)
        self.wait(0.25)
        self.play(Circumscribe(formula[5], color=GREEN), Create(w2_arrow), Write(w2_label))
        self.play(Create(representation_plane), Write(representation_label))
        self.play(TransformFromCopy(preactivation_bars, represented_dots), run_time=1.6)
        self.wait(0.35)
        self.play(Create(split_line), Write(split_label), FadeIn(positive_side), FadeIn(negative_side))
        self.wait(0.8)
