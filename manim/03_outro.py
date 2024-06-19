import manim as m


class BulletedListNN(m.Scene):
    def construct(self):
        blist = m.BulletedList("Linear Layers", "Sigmoid", "Backpropagation", height=2, width=2).scale(2.0)
        blist.set_color_by_tex("Linear Layers", m.RED)
        blist.set_color_by_tex("Sigmoid", m.GREEN)
        blist.set_color_by_tex("Backpropagation", m.BLUE)
        self.play(m.Write(blist))
        self.wait(5)
