

class Network(object):
    """ A network represents a collection of layers """

    def __init__(self):
        self.layers = []
        self.criterion = None
        self.inp = None
        self.inps = []
        self.output  = None

    def forward(self, inp, targets):
        self.inps = [inp]
        for l in self.layers:
            inp = l.forward(inp)
            self.inps.append(inp)
        inp = self.criterion.forward(inp, targets)
        self.output = inp
        return inp

    def backward(self, inp, targets):
        grad_output = self.criterion.backward(self.inps[-1], targets)

        for i,l in enumerate(self.layers[::-1]):
            grad_output = l.backward(self.inps[len(self.layers) -1 - i], grad_output)

        return grad_output

    def zero_grad_parameters(self):
        for l in self.layers:
            l.zero_grad_parameters()

    def update_parameters(self, learning_rate=1.0):
        for l in self.layers:
            l.update_parameters(learning_rate)

    def train(self, X, targets, learning_rate=1e-3):
        output = self.forward(X, targets)

        self.zero_grad_parameters()
        grad_input = self.backward( X, targets )

        self.update_parameters(learning_rate)
        return output.sum()

    def training(self):
        for l in self.layers:
            l.training()

    def evaluate(self):
        for l in self.layers:
            l.evaluate()





