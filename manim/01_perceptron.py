import manim as m


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
