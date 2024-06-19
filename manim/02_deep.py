import manim as m
from nn import NeuralNetworkMobject


class PerceptronDeeper(m.Scene):
    def construct(self):
        nn = NeuralNetworkMobject([128*128*3, 1])
        nn.label_inputs("x")
        nn.label_outputs("y")
        nn.label_outputs_text(["is\ cat?"])

        nn.scale(0.75).shift(1.5*m.LEFT)
        self.play(m.Write(nn))
        self.wait(4)

        nn2 = NeuralNetworkMobject([128*128*3, 4, 1])
        nn2.label_inputs("x")
        nn2.label_outputs("y")
        nn2.label_outputs_text(["is\ cat?"])

        nn2.scale(0.75).shift(1.0*m.LEFT)
        self.play(m.TransformMatchingShapes(nn, nn2))
        self.wait(8)

        self.wait(60)


class DerivativesAllCW(m.Scene):
    def construct(self):
        c_w = m.MathTex(r"\frac{\partial C}{\partial w'}").scale(2.0)
        self.play(m.Write(c_w))
        self.wait(10)

        c_w2 = m.MathTex(r"\frac{\partial C}{\partial w'} = \frac{\partial C}{\partial a'} \times \frac{\partial a'}{\partial z'} \times \frac{\partial z'}{\partial w'}").scale(2.0)
        self.play(m.TransformMatchingShapes(c_w, c_w2))
        self.wait(10)


class DerivativeAPrime(m.Scene):
    def construct(self):
        c_w = m.MathTex(r"\frac{\partial C}{\partial a'}").scale(2.0)
        self.play(m.Write(c_w))
        self.wait(30)

        initial_3 = m.MathTex(r"\frac{\partial C}{\partial a'} = \frac{\partial C}{\partial a} \times \frac{\partial a}{\partial z} \times \frac{\partial z}{\partial a'}").scale(2)
        self.play(m.TransformMatchingShapes(c_w, initial_3))
        self.wait(20)


class DerivativesAllCW2(m.Scene):
    def construct(self):
        c_w2 = m.MathTex(r"\frac{\partial C}{\partial w'} = \frac{\partial C}{\partial a'} \times \frac{\partial a'}{\partial z'} \times \frac{\partial z'}{\partial w'}").scale(2.0)
        self.play(m.Write(c_w2))
        self.wait(5)

        c_w3 = m.MathTex(r"\frac{\partial C}{\partial w'} = \frac{\partial C}{\partial a} \times \frac{\partial a}{\partial z} \times \frac{\partial z}{\partial a'} \times \frac{\partial a'}{\partial z'} \times \frac{\partial z'}{\partial w'}").scale(1.5)
        self.play(m.TransformMatchingShapes(c_w2, c_w3))
        self.wait(30)
