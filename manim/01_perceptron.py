import manim as m
from nn import NeuralNetworkMobject


class WParamIn(m.Scene):
    def construct(self):
        text = m.Text('w').scale(3)
        self.play(m.FadeIn(text))


class WParam(m.Scene):
    def construct(self):
        text = m.Text('w').scale(3)
        self.add(text)


class WParamOut(m.Scene):
    def construct(self):
        text = m.Text('w').scale(3)
        self.play(m.FadeOut(text))


class WParamCostChange(m.Scene):
    def cost_w(self, w):
        return -5 * w ** 4 + 5 * w ** 6 + w ** 2 + 5

    def construct(self):
        w = 1.0
        cost = self.cost_w(w)
        w_var = m.Variable(w, m.Text("w"), num_decimal_places=3)
        cost_var = m.Variable(cost, m.MathTex("{cost}_{w}"), num_decimal_places=3)
        m.Group(w_var, cost_var).arrange(m.DOWN)
        self.play(m.Write(w_var), m.Write(cost_var))

        self.wait(2)

        w = 0.5
        cost = self.cost_w(w)
        w_tracker = w_var.tracker
        cost_tracker = cost_var.tracker
        self.play(
            w_tracker.animate.set_value(w),
            cost_tracker.animate.set_value(cost)
        )

        self.wait(3)

        w = 0.8
        cost = self.cost_w(w)
        w_tracker = w_var.tracker
        cost_tracker = cost_var.tracker
        self.play(
            w_tracker.animate.set_value(w),
            cost_tracker.animate.set_value(cost),
            run_time=1
        )

        self.wait(0.5)

        w = 0.7
        cost = self.cost_w(w)
        w_tracker = w_var.tracker
        cost_tracker = cost_var.tracker
        self.play(
            w_tracker.animate.set_value(w),
            cost_tracker.animate.set_value(cost),
            run_time=1
        )

        self.wait(0.5)

        w = 0.6
        cost = self.cost_w(w)
        w_tracker = w_var.tracker
        cost_tracker = cost_var.tracker
        self.play(
            w_tracker.animate.set_value(w),
            cost_tracker.animate.set_value(cost),
            run_time=1
        )

        self.wait(0.5)


class CostPlot(m.Scene):
    def cost_w(self, w):
        return -5 * w ** 4 + 5 * w ** 6 + w ** 2 + 5

    def construct(self):
        ax = m.Axes(x_range=(0, 2), y_range=(4, 7), x_length=8, y_length=5)

        y_label = ax.get_y_axis_label(
            m.MathTex("{cost}_{w}").scale(0.65).rotate(90 * m.DEGREES),
            edge=m.LEFT,
            direction=m.LEFT,
            buff=0.3,
        )
        x_label = ax.get_x_axis_label(
            m.Text("w").scale(0.65),
            edge=m.RIGHT,
            direction=m.DOWN,
            buff=0.3,
        )

        self.play(m.Create(ax))
        self.wait(1.5)

        points = [
            ax.coords_to_point(w, self.cost_w(w))
            for w in [1.0, 0.5, 0.8, 0.7, 0.6]
        ]
        dots = m.VGroup(*[m.Dot(p) for p in points])
        self.play(m.Write(dots))

        self.wait(3.5)

        self.play(m.Write(x_label))

        self.wait(1)

        self.play(m.Write(y_label))

        self.wait(2)

        curve = ax.plot(lambda x: self.cost_w(x), color=m.RED, x_range=(0, 1.1))
        self.play(m.Create(curve))

        self.wait(5)

        min_w = 0.7376
        min_cost = self.cost_w(min_w)
        label = ax.get_graph_label(
            graph=curve,
            label=m.MathTex(r"min_{w}"),
            x_val=min_w,
            dot=True,
            direction=m.DR,
            dot_config={"color": m.BLUE},
            color=m.BLUE,
        )
        self.play(m.Create(label))

        self.wait(5)


class DerivativeMinFunction(m.Scene):
    def construct(self):
        f = m.MathTex(r"f(w) = -5w^4 + 5w^6 + w^2 + 5")
        min_f = m.MathTex(r"min_{f} = ?").next_to(f, m.DOWN)

        self.play(m.Write(f))
        self.play(m.Write(min_f))
        self.wait(2)

        min_f_0 = m.MathTex(r"min_{f} = w, \text{where} f'(w) = 0").next_to(f, m.DOWN)

        self.play(m.TransformMatchingShapes(min_f, min_f_0, path_arc=0))

        self.wait(5)


class MinGradientDescent(m.Scene):
    def cost_w(self, w):
        return -5 * w ** 4 + 5 * w ** 6 + w ** 2 + 5

    def construct(self):
        ax = m.Axes(x_range=(0, 2), y_range=(4, 7), x_length=8, y_length=5)
        y_label = ax.get_y_axis_label(
            m.MathTex("{cost}_{w}").scale(0.65).rotate(90 * m.DEGREES),
            edge=m.LEFT,
            direction=m.LEFT,
            buff=0.3,
        )
        x_label = ax.get_x_axis_label(
            m.Text("w").scale(0.65),
            edge=m.RIGHT,
            direction=m.DOWN,
            buff=0.3,
        )

        self.play(m.Create(ax), m.Write(x_label), m.Write(y_label))
        self.wait(1.5)

        curve = ax.plot(lambda x: self.cost_w(x), color=m.RED, x_range=(0, 1.1))
        self.play(m.Create(curve))

        self.wait(4)

        random_w = 1.0
        cost = self.cost_w(random_w)
        x_var = m.Variable(random_w, m.Text("w"), num_decimal_places=4)
        cost_var = m.Variable(cost, m.MathTex("{cost}_{w}"), num_decimal_places=4)
        m.Group(x_var, cost_var).arrange(m.DOWN).next_to(curve, m.RIGHT)
        x = x_var.tracker
        y = cost_var.tracker

        dot = m.always_redraw(lambda: m.Dot().move_to(ax.coords_to_point(x.get_value(), self.cost_w(x.get_value()))))
        self.play(m.Write(dot), m.Write(x_var), m.Write(cost_var))

        self.wait(7)

        slopes = m.always_redraw(
            lambda: ax.get_secant_slope_group(
                x=x.get_value() - 0.0005,
                graph=curve,
                dx=0.001,
                dx_line_color=m.YELLOW,
                dy_line_color=m.GREEN,
                secant_line_length=4,
                secant_line_color=m.BLUE,
            )
        )

        self.play(m.Create(slopes))

        self.play(x.animate.set_value(0.5), y.animate.set_value(self.cost_w(0.5)), run_time=4)

        self.play(x.animate.set_value(0.7376), y.animate.set_value(self.cost_w(0.7376)), run_time=2)

        self.wait(7)


class MultiLocalMinima(m.Scene):
    def cost_w(self, w):
        return m.np.cos(3 * m.np.pi * w) / w

    def construct(self):
        ax = m.Axes(x_range=(0.1, 1.1), y_range=(-2, 2), x_length=8, y_length=5).scale(0.8)
        y_label = ax.get_y_axis_label(
            m.MathTex("{cost}_{w}").scale(0.65).rotate(90 * m.DEGREES),
            edge=m.LEFT,
            direction=m.LEFT,
            buff=0.3,
        )
        x_label = ax.get_x_axis_label(
            m.Text("w").scale(0.65),
            edge=m.RIGHT,
            direction=m.DOWN,
            buff=0.3,
        )

        self.play(m.Create(ax), m.Write(x_label), m.Write(y_label))
        self.wait(1)

        curve = ax.plot(lambda x: self.cost_w(x), color=m.RED, x_range=(0.1, 1.1))
        self.play(m.Create(curve))

        self.wait(1)

        xs = [m.ValueTracker(a) for a in [0.2, 0.4, 0.6, 0.8, 1.0]]
        dots = m.always_redraw(lambda: m.VGroup(*[m.Dot().move_to(ax.coords_to_point(x.get_value(), self.cost_w(x.get_value()))) for x in xs]))

        self.play(m.Write(dots))

        self.wait(1)

        min_xs = [0.296, 0.296, 0.296, 0.988, 0.988]
        self.play(*[x.animate.set_value(min_x) for x, min_x in zip(xs, min_xs)], run_time=2)

        self.wait(5)


class Cost2DParameters(m.ThreeDScene):
    def construct(self):
        resolution_fa = 16
        self.set_camera_orientation(phi=75 * m.DEGREES, theta=-60 * m.DEGREES)
        axes = m.ThreeDAxes(x_range=(-3, 3, 1), y_range=(-3, 3, 1), z_range=(-5, 5, 1))

        def param_trig(u, v):
            x = u
            y = v
            z = 2 * m.np.sin(x) + 2 * m.np.cos(y)
            return z

        trig_plane = axes.plot_surface(
            param_trig,
            resolution=(resolution_fa, resolution_fa),
            u_range=(-3, 3),
            v_range=(-3, 3),
            colorscale=[m.BLUE, m.GREEN, m.YELLOW, m.ORANGE, m.RED],
        )

        self.play(m.Create(axes), m.Write(trig_plane))

        self.wait(5)


class MultiDimensionalIntuition(m.Scene):
    def construct(self):
        text = m.MathTex(r"w, b").scale(1)
        var_dim = m.Variable(2, m.Text("dim"), num_decimal_places=0).next_to(text, m.DOWN)
        self.play(m.Write(text), m.Write(var_dim))
        self.wait(3)

        text_more = m.MathTex(r"w_1, w_2, \ldots, w_n, b").scale(1)
        var_dim_tracker = var_dim.tracker

        self.play(
            var_dim_tracker.animate.set_value(128*128*3+1),
            m.TransformMatchingShapes(text, text_more)
        )
        self.wait(5)


class Perceptron(m.Scene):
    def construct(self):
        nn = NeuralNetworkMobject([128*128*3, 1])
        nn.label_inputs("x")
        nn.label_outputs("y")
        nn.label_outputs_text(["is\ cat?"])

        nn.scale(0.75).shift(3.5*m.LEFT)
        self.play(m.Write(nn))
        self.wait(1)

        texts = ["yes", "no", "no", "yes", "yes"]
        text_mobjects = [m.Text(text, color=m.YELLOW).scale(0.5) for text in texts]
        text_group = m.VGroup(m.Text("Predictions").scale(0.5), *text_mobjects).arrange(m.DOWN, buff=0.5).next_to(nn, m.RIGHT)
        self.play(m.Write(text_group))

        ground_truths = ["yes", "no", "yes", "yes", "no"]
        ground_truth_mobjects = [m.Text(text, color=m.RED if pred != text else m.GREEN).scale(0.5) for (pred, text) in zip(texts, ground_truths)]
        ground_truth_group = m.VGroup(m.Text("Ground Truth").scale(0.5), *ground_truth_mobjects).arrange(m.DOWN, buff=0.5).next_to(text_group, m.RIGHT)
        self.play(m.Write(ground_truth_group))
        self.wait(1)

        cost_value = sum([1 if pred != text else 0 for (pred, text) in zip(texts, ground_truths)]) / len(texts)
        cost = m.Variable(cost_value, m.MathTex("{cost}_{w}"), num_decimal_places=2).next_to(text_group, m.DOWN)
        self.play(m.Write(cost))
        self.wait(5)


class GradientFormula(m.Scene):
    def construct(self):
        parameters = m.MathTex(r"w_1, w_2, \ldots, w_n, b").scale(1.5)
        self.play(m.Write(parameters))

        self.wait(2)

        cost_derivatives = m.MathTex(r"\frac{\partial C}{\partial w_1}, \frac{\partial C}{\partial w_2}, \ldots, \frac{\partial C}{\partial w_n}, \frac{\partial C}{\partial b}").scale(1.5)
        self.play(m.TransformMatchingShapes(parameters, cost_derivatives))

        self.wait(2)

        cost_vector = m.MathTex(r"\nabla C = \begin{pmatrix} \cfrac{\partial C}{\partial w_1} \vspace{.4em} \\ \cfrac{\partial C}{\partial w_2} \vspace{.4em} \\ \vdots \vspace{.4em} \\ \cfrac{\partial C}{\partial w_n} \vspace{.4em} \\ \cfrac{\partial C}{\partial b} \end{pmatrix}").scale(1.0)
        self.play(m.TransformMatchingShapes(cost_derivatives, cost_vector))

        self.wait(7)

        params_minus_cost_vector = m.MathTex(r"m' = m - \nabla C = \begin{pmatrix} w_1 \vspace{.4em} \\ w_2 \vspace{.4em} \\ \vdots \vspace{.4em} \\ w_n \vspace{.4em} \\ b \end{pmatrix} - \begin{pmatrix} \cfrac{\partial C}{\partial w_1} \vspace{.4em} \\ \cfrac{\partial C}{\partial w_2} \vspace{.4em} \\ \vdots \vspace{.4em} \\ \cfrac{\partial C}{\partial w_n} \vspace{.4em} \\ \cfrac{\partial C}{\partial b} \end{pmatrix}").scale(1.0)
        self.play(m.TransformMatchingShapes(cost_vector, params_minus_cost_vector))

        self.wait(6)


class GradientDescentText(m.Scene):
    def construct(self):
        text = m.Text("Gradient Descent").scale(2)
        self.play(m.Write(text))
        self.wait(5)


class MathText(m.Scene):
    def construct(self):
        text = m.Text("Math").scale(2)
        self.play(m.Write(text))
        self.wait(5)


class DerivativeOfCostWithRespectToWeightsText(m.Scene):
    def construct(self):
        text = m.MathTex(r"\frac{\partial C}{\partial w}").scale(2)
        self.play(m.Write(text))
        self.wait(7)


class ChainRuleExample(m.Scene):
    def construct(self):
        self.wait(7)

        initial = m.MathTex(r"\cfrac{a}{b} \times \cfrac{b}{c} \times \cfrac{c}{d}").scale(2)
        self.play(m.Write(initial), run_time=4)
        self.wait(1)

        equals = m.MathTex(r"\cfrac{a}{b} \times \cfrac{b}{c} \times \cfrac{c}{d} = \cfrac{a}{d}").scale(2)
        self.play(m.TransformMatchingShapes(initial, equals))
        self.wait(12)


class DerivativeOfCostWithRespectToWeights(m.Scene):
    def construct(self):
        initial = m.MathTex(r"\frac{\partial C}{\partial w}").scale(2)
        self.play(m.Write(initial))
        self.wait(4)

        initial_2 = m.MathTex(r"\frac{\partial C}{\partial something} \times \frac{\partial something}{\partial w}").scale(2)
        self.play(m.TransformMatchingShapes(initial, initial_2))
        self.wait(5)


class NNFormulas(m.Scene):
    def construct(self):
        c_formula = m.MathTex(r"C = (a - \hat{y})^2").scale(1.5).move_to(m.ORIGIN + m.UP)
        self.play(m.Write(c_formula))

        self.wait(1)

        z_formula = m.MathTex(r"z = w \cdot x + b").scale(1.5).next_to(c_formula, m.DOWN)
        self.play(m.Write(z_formula))

        self.wait(1)

        a_formula = m.MathTex(r"a = \sigma(z)").scale(1.5).next_to(z_formula, m.DOWN)
        self.play(m.Write(a_formula))

        self.wait(7)


class DerivativeOfCostWithRespectToWeightsUseA(m.Scene):
    def construct(self):
        initial_2 = m.MathTex(r"\frac{\partial C}{\partial a} \times \frac{\partial a}{\partial w}").scale(2)
        self.play(m.Write(initial_2))
        self.wait(9)

        initial_3 = m.MathTex(r"\frac{\partial C}{\partial a} \times \frac{\partial a}{\partial z} \times \frac{\partial z}{\partial w}").scale(2)
        self.play(m.TransformMatchingShapes(initial_2, initial_3))
        self.wait(7)


class SigmoidDerivative(m.Scene):
    def construct(self):
        sigmoid = m.MathTex(r"a = \sigma(z) = \frac{1}{1 + e^{-z}}").scale(1.5).move_to(m.ORIGIN + m.UP)
        self.play(m.Write(sigmoid))

        self.wait(2)

        sigmoid_prime = m.MathTex(r"\frac{\partial a}{\partial z} = \sigma'(z) = \sigma(z) \cdot (1 - \sigma(z))").scale(1.5).next_to(sigmoid, m.DOWN)
        self.play(m.Write(sigmoid_prime))

        self.wait(10)


class DZDWDerivative(m.Scene):
    def construct(self):
        z_formula = m.MathTex(r"z = w \cdot x + b").scale(1.5).move_to(m.ORIGIN + m.UP)
        self.play(m.Write(z_formula))

        self.wait(2)

        dz_dw = m.MathTex(r"\frac{\partial z}{\partial w} = x").scale(1.5).next_to(z_formula, m.DOWN)
        self.play(m.Write(dz_dw))

        self.wait(10)


class DerivativesAllWToB(m.Scene):
    def construct(self):
        c_derivative = m.MathTex(r"\frac{\partial C}{\partial a} = 2(a - \hat{y})").scale(1.0).move_to(m.ORIGIN + m.UP * 3)
        self.play(m.Write(c_derivative))

        a_derivative = m.MathTex(r"\frac{\partial a}{\partial z} = \sigma(z) \cdot (1 - \sigma(z))").scale(1.0).next_to(c_derivative, m.DOWN)
        self.play(m.Write(a_derivative))

        z_derivative = m.MathTex(r"\frac{\partial z}{\partial w} = x, \frac{\partial z}{\partial b} = 1").scale(1.0).next_to(a_derivative, m.DOWN)
        self.play(m.Write(z_derivative))

        c_w = m.MathTex(r"\frac{\partial C}{\partial w} = 2(a - \hat{y}) \cdot \sigma(z) \cdot (1 - \sigma(z)) \cdot x").scale(1.0).next_to(z_derivative, m.DOWN)
        self.play(m.Write(c_w))
        self.wait(10)


class MinGradientDescentLearning(m.Scene):
    def cost_w(self, w):
        return -5 * w ** 4 + 5 * w ** 6 + w ** 2 + 5

    def construct(self):
        ax = m.Axes(x_range=(0, 2), y_range=(4, 7), x_length=8, y_length=5)
        y_label = ax.get_y_axis_label(
            m.MathTex("{cost}_{w}").scale(0.65).rotate(90 * m.DEGREES),
            edge=m.LEFT,
            direction=m.LEFT,
            buff=0.3,
        )
        x_label = ax.get_x_axis_label(
            m.Text("w").scale(0.65),
            edge=m.RIGHT,
            direction=m.DOWN,
            buff=0.3,
        )
        curve = ax.plot(lambda x: self.cost_w(x), color=m.RED, x_range=(0, 1.1))

        self.add(ax, x_label, y_label, curve)

        random_w = 1.0
        cost = self.cost_w(random_w)
        x_var = m.Variable(random_w, m.Text("w"), num_decimal_places=4)
        cost_var = m.Variable(cost, m.MathTex("{cost}_{w}"), num_decimal_places=4)
        m.Group(x_var, cost_var).arrange(m.DOWN).next_to(curve, m.RIGHT)
        x = x_var.tracker
        y = cost_var.tracker

        dot = m.always_redraw(lambda: m.Dot().move_to(ax.coords_to_point(x.get_value(), self.cost_w(x.get_value()))))
        self.add(dot, x_var, cost_var)

        slopes = m.always_redraw(
            lambda: ax.get_secant_slope_group(
                x=x.get_value() - 0.0005,
                graph=curve,
                dx=0.001,
                dx_line_color=m.YELLOW,
                dy_line_color=m.GREEN,
                secant_line_length=4,
                secant_line_color=m.BLUE,
            )
        )

        self.add(slopes)

        self.play(x.animate.set_value(0.7376), y.animate.set_value(self.cost_w(0.7376)), run_time=2)

        self.wait(7)
