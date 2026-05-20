from __future__ import annotations

import numpy as np
from manim import (
    BLUE,
    DOWN,
    GREEN,
    GREY_B,
    LEFT,
    ORANGE,
    PURPLE,
    RIGHT,
    UP,
    Arrow,
    Create,
    Dot,
    FadeIn,
    FadeOut,
    MathTex,
    NumberPlane,
    Scene,
    Text,
    Transform,
    TransformFromCopy,
    VGroup,
    Write,
)

from deepmatrixanim.iris_training import build_iris_training_result


CLASS_COLORS = (BLUE, ORANGE, GREEN)


def normalize_projection(points: np.ndarray, reference: np.ndarray) -> np.ndarray:
    center = reference.mean(axis=0)
    spread = np.max(np.abs(reference - center))
    if spread == 0.0:
        return np.zeros_like(points)
    return (points - center) / spread * 2.05


def point_to_scene(plane: NumberPlane, point: np.ndarray) -> np.ndarray:
    return plane.c2p(float(point[0]), float(point[1]))


def dots_for(plane: NumberPlane, points: np.ndarray, labels: np.ndarray, radius: float = 0.026) -> VGroup:
    dots = []
    for point, label in zip(points, labels):
        dots.append(Dot(point_to_scene(plane, point), radius=radius, color=CLASS_COLORS[int(label)]))
    return VGroup(*dots)


def plot_panel(title: str, subtitle: str, center: np.ndarray, width: float = 3.3, height: float = 2.85) -> VGroup:
    plane = NumberPlane(
        x_range=[-2.4, 2.4, 1],
        y_range=[-2.4, 2.4, 1],
        x_length=width,
        y_length=height,
        background_line_style={
            "stroke_color": GREY_B,
            "stroke_width": 1,
            "stroke_opacity": 0.35,
        },
    ).move_to(center)
    title_mob = Text(title).scale(0.25).next_to(plane, UP, buff=0.1)
    subtitle_mob = Text(subtitle).scale(0.18).next_to(title_mob, DOWN, buff=0.05)
    return VGroup(plane, title_mob, subtitle_mob)


def class_legend(target_names: tuple[str, ...]) -> VGroup:
    entries = []
    for index, name in enumerate(target_names):
        entries.append(
            VGroup(Dot(radius=0.045, color=CLASS_COLORS[index]), Text(name).scale(0.2)).arrange(RIGHT, buff=0.1)
        )
    return VGroup(*entries).arrange(RIGHT, buff=0.45)


class IrisMLPBeforeAfterTraining(Scene):
    def construct(self):
        result = build_iris_training_result()
        labels = result.y

        input_points = normalize_projection(result.input_projection, result.input_projection)
        hidden_before = normalize_projection(
            result.hidden_before_projection,
            np.vstack([result.hidden_before_projection, result.hidden_after_projection]),
        )
        hidden_after = normalize_projection(
            result.hidden_after_projection,
            np.vstack([result.hidden_before_projection, result.hidden_after_projection]),
        )
        logits_before = normalize_projection(
            result.logits_before_projection,
            np.vstack([result.logits_before_projection, result.logits_after_projection]),
        )
        logits_after = normalize_projection(
            result.logits_after_projection,
            np.vstack([result.logits_before_projection, result.logits_after_projection]),
        )

        title = Text("Training changes how Iris moves through vector spaces").scale(0.42).to_edge(UP)
        formula = MathTex(
            r"x",
            r"\rightarrow",
            r"W_1x+b_1",
            r"\rightarrow",
            r"\mathrm{ReLU}",
            r"\rightarrow",
            r"W_2a+b_2",
            r"\rightarrow",
            r"\mathrm{logits}",
        ).scale(0.46).next_to(title, DOWN, buff=0.12)
        legend = class_legend(result.target_names).to_edge(DOWN)

        input_panel = plot_panel("input", "PCA of standardized 4D Iris", LEFT * 4.1 + DOWN * 0.25)
        hidden_panel = plot_panel("hidden", "shared PCA of 8D activations", DOWN * 0.25)
        logits_panel = plot_panel("logits", "fixed 3-class contrast projection", RIGHT * 4.1 + DOWN * 0.25)

        input_dots = dots_for(input_panel[0], input_points, labels)
        hidden_before_dots = dots_for(hidden_panel[0], hidden_before, labels)
        hidden_after_dots = dots_for(hidden_panel[0], hidden_after, labels)
        logits_before_dots = dots_for(logits_panel[0], logits_before, labels)
        logits_after_dots = dots_for(logits_panel[0], logits_after, labels)

        arrow_1 = Arrow(input_panel[0].get_right() + RIGHT * 0.12, hidden_panel[0].get_left() + LEFT * 0.12, buff=0.1, color=PURPLE)
        arrow_2 = Arrow(hidden_panel[0].get_right() + RIGHT * 0.12, logits_panel[0].get_left() + LEFT * 0.12, buff=0.1, color=PURPLE)

        mode_label = Text("random weights").scale(0.32).set_color(ORANGE).next_to(formula, DOWN, buff=0.16)
        trained_label = Text("trained weights").scale(0.32).set_color(GREEN).next_to(formula, DOWN, buff=0.16)

        before_metrics = Text(
            f"before acc {result.initial_accuracy:.2f}  loss {result.initial_loss:.2f}",
        ).scale(0.23).next_to(logits_panel[0], DOWN, buff=0.12)
        after_metrics = Text(
            f"after acc {result.trained_accuracy:.2f}  loss {result.trained_loss:.2f}",
        ).scale(0.23).next_to(logits_panel[0], DOWN, buff=0.12)

        self.play(Write(title), Write(formula), FadeIn(legend))
        self.play(Create(input_panel), FadeIn(input_dots))
        self.wait(0.25)

        self.play(Write(mode_label), Create(arrow_1), Create(hidden_panel))
        self.play(TransformFromCopy(input_dots, hidden_before_dots), run_time=1.3)
        self.play(Create(arrow_2), Create(logits_panel))
        self.play(TransformFromCopy(hidden_before_dots, logits_before_dots), FadeIn(before_metrics), run_time=1.3)
        self.wait(0.45)

        self.play(Transform(mode_label, trained_label), FadeOut(before_metrics))
        self.play(Transform(hidden_before_dots, hidden_after_dots), run_time=1.2)
        self.play(Transform(logits_before_dots, logits_after_dots), FadeIn(after_metrics), run_time=1.2)
        self.wait(0.45)

        final_before_panel = plot_panel("before training", "random logit geometry", LEFT * 2.4 + DOWN * 0.25, width=3.7)
        final_after_panel = plot_panel("after training", "learned class separation", RIGHT * 2.4 + DOWN * 0.25, width=3.7)
        final_before_dots = dots_for(final_before_panel[0], logits_before, labels, radius=0.028)
        final_after_dots = dots_for(final_after_panel[0], logits_after, labels, radius=0.028)
        final_before_metrics = Text(f"acc {result.initial_accuracy:.2f}").scale(0.25).next_to(
            final_before_panel[0],
            DOWN,
            buff=0.12,
        )
        final_after_metrics = Text(f"acc {result.trained_accuracy:.2f}").scale(0.25).next_to(
            final_after_panel[0],
            DOWN,
            buff=0.12,
        )
        rule = MathTex(r"\mathrm{predict}\ \arg\max(\mathrm{logits})").scale(0.42).next_to(formula, DOWN, buff=0.16)
        closing = Text("Learned matrices reshape 4D measurements into class-separable logits.").scale(0.29).to_edge(DOWN)

        self.play(
            FadeOut(input_panel),
            FadeOut(input_dots),
            FadeOut(hidden_panel),
            FadeOut(hidden_before_dots),
            FadeOut(logits_panel),
            FadeOut(logits_before_dots),
            FadeOut(arrow_1),
            FadeOut(arrow_2),
            FadeOut(mode_label),
            FadeOut(after_metrics),
            FadeOut(legend),
        )
        self.play(Transform(formula, rule))
        self.play(
            Create(final_before_panel),
            FadeIn(final_before_dots),
            FadeIn(final_before_metrics),
            Create(final_after_panel),
            FadeIn(final_after_dots),
            FadeIn(final_after_metrics),
        )
        self.play(Write(closing))
        self.wait(1.0)
