from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    LEFT,
    ORANGE,
    RED,
    RIGHT,
    UP,
    VGroup,
    Arrow,
    Circumscribe,
    Create,
    FadeIn,
    MathTex,
    Rectangle,
    Scene,
    SurroundingRectangle,
    Tex,
    Write,
)

from deepmatrixanim.mlp import demo_forward_pass


def matrix_tex(values: np.ndarray, decimals: int = 2) -> MathTex:
    rows = []
    for row in values:
        cells = []
        for value in row:
            if float(value).is_integer():
                cells.append(str(int(value)))
            else:
                cells.append(f"{value:.{decimals}f}".rstrip("0").rstrip("."))
        rows.append(" & ".join(cells))
    body = r" \\ ".join(rows)
    return MathTex(r"\begin{bmatrix}" + body + r"\end{bmatrix}")


def labeled_matrix(label: str, values: np.ndarray, color) -> VGroup:
    label_mob = MathTex(label).set_color(color)
    matrix_mob = matrix_tex(values)
    group = VGroup(label_mob, matrix_mob).arrange(DOWN, buff=0.15)
    box = SurroundingRectangle(group, color=color, buff=0.18)
    return VGroup(group, box)


def vector_row_center(labeled: VGroup, row_index: int, row_count: int):
    matrix_mob = labeled[0][1]
    offset = matrix_mob.height * (0.5 - (row_index + 0.5) / row_count)
    return matrix_mob.get_center() + UP * offset


class MLPForwardDemo(Scene):
    def construct(self):
        values = demo_forward_pass()

        title = Tex("DeepMatrixanim: two-layer MLP forward pass")
        title.scale(0.82)

        formula = MathTex(
            r"x",
            r"\rightarrow",
            r"z_1 = W_1x + b_1",
            r"\rightarrow",
            r"a_1 = \mathrm{ReLU}(z_1)",
            r"\rightarrow",
            r"y = W_2a_1 + b_2",
        ).scale(0.58)
        formula.next_to(title, DOWN, buff=0.35)

        x = labeled_matrix("x", values["x"], BLUE)
        w1 = labeled_matrix("W_1", values["w1"], ORANGE)
        b1 = labeled_matrix("b_1", values["b1"], ORANGE)
        z1 = labeled_matrix("z_1", values["z1"], GREEN)
        a1 = labeled_matrix("a_1", values["a1"], GREEN)
        w2 = labeled_matrix("W_2", values["w2"], ORANGE)
        b2 = labeled_matrix("b_2", values["b2"], ORANGE)
        y = labeled_matrix("y", values["y"], BLUE)

        x.scale(0.62)
        w1.scale(0.46)
        b1.scale(0.53)
        z1.scale(0.53)
        a1.scale(0.53)
        w2.scale(0.49)
        b2.scale(0.58)
        y.scale(0.58)

        first_layer = VGroup(w1, x, b1, z1).arrange(RIGHT, buff=0.35)
        first_layer.next_to(formula, DOWN, buff=0.32).shift(LEFT * 0.2)
        plus_1 = MathTex("+").scale(0.8).move_to((x.get_right() + b1.get_left()) / 2)
        arrow_1 = Arrow(b1.get_right(), z1.get_left(), buff=0.12, color=GREEN)
        first_label = MathTex(r"z_1 = W_1x + b_1").scale(0.5)
        first_label.next_to(first_layer, DOWN, buff=0.12)

        relu_panel = VGroup(z1.copy(), a1).arrange(RIGHT, buff=0.55)
        relu_panel.next_to(first_label, DOWN, buff=0.45)
        relu_arrow = Arrow(relu_panel[0].get_right(), relu_panel[1].get_left(), buff=0.12, color=GREEN)
        relu_label = MathTex(r"\mathrm{ReLU}(-2.5) = 0").set_color(RED).scale(0.48)
        relu_label.next_to(relu_panel, UP, buff=0.08)
        negative_box = Rectangle(width=0.52, height=0.3, color=RED)
        negative_box.move_to(vector_row_center(relu_panel[0], row_index=1, row_count=4))
        zero_box = Rectangle(width=0.52, height=0.3, color=RED)
        zero_box.move_to(vector_row_center(relu_panel[1], row_index=1, row_count=4))

        output_layer = VGroup(w2, a1.copy(), b2, y).arrange(RIGHT, buff=0.28)
        output_layer.next_to(relu_panel, DOWN, buff=0.34)
        plus_2 = MathTex("+").scale(0.8).move_to((output_layer[1].get_right() + b2.get_left()) / 2)
        arrow_2 = Arrow(b2.get_right(), y.get_left(), buff=0.12, color=BLUE)
        output_label = MathTex(r"y = W_2a_1 + b_2").scale(0.5)
        output_label.next_to(output_layer, DOWN, buff=0.12)

        full_layout = VGroup(
            title,
            formula,
            first_layer,
            plus_1,
            arrow_1,
            first_label,
            relu_panel,
            relu_arrow,
            relu_label,
            negative_box,
            zero_box,
            output_layer,
            plus_2,
            arrow_2,
            output_label,
        )
        max_width = 12.4
        max_height = 6.25
        full_layout.scale(min(max_width / full_layout.width, max_height / full_layout.height, 1.0))
        full_layout.move_to(0)

        self.play(Write(title), Write(formula))
        self.wait(0.4)
        self.play(FadeIn(w1), FadeIn(x), FadeIn(b1), Write(plus_1))
        self.play(Create(arrow_1), FadeIn(z1), Write(first_label))
        self.play(Circumscribe(formula[2], color=GREEN))
        self.wait(0.5)
        self.play(
            FadeIn(relu_panel[0]),
            Create(relu_arrow),
            FadeIn(relu_panel[1]),
            Write(relu_label),
        )
        self.play(Create(negative_box), Create(zero_box))
        self.play(Circumscribe(formula[4], color=RED))
        self.wait(0.5)
        self.play(FadeIn(w2), FadeIn(output_layer[1]), FadeIn(b2), Write(plus_2))
        self.play(Create(arrow_2), FadeIn(y), Write(output_label))
        self.play(Circumscribe(formula[6], color=BLUE))
        self.wait(1.0)
