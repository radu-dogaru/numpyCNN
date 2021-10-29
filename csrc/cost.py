import cupy as cp

epsilon = 1e-20


class CostFunction:
    def f(self, a_last, y):
        raise NotImplementedError

    def grad(self, a_last, y):
        raise NotImplementedError


class SigmoidCrossEntropy(CostFunction):
    def f(self, a_last, y):
        batch_size = y.shape[0]
        # It would be better to have the logits and use this instead
        # max(logits, 0) - logits * y + log(1 + exp(-abs(logits)))
        a_last = cp.clip(a_last, epsilon, 1.0 - epsilon)
        cost = -1 / batch_size * (y * cp.log(a_last) + (1 - y) * cp.log(1 - a_last)).sum()
        return cost

    def grad(self, a_last, y):
        a_last = cp.clip(a_last, epsilon, 1.0 - epsilon)
        return - (cp.divide(y, a_last) - cp.divide(1 - y, 1 - a_last))


class SoftmaxCrossEntropy(CostFunction):
    def f(self, a_last, y):
        batch_size = y.shape[0]
        cost = -1 / batch_size * (y * cp.log(cp.clip(a_last, epsilon, 1.0))).sum()
        return cost

    def grad(self, a_last, y):
        return - cp.divide(y, cp.clip(a_last, epsilon, 1.0))


softmax_cross_entropy = SoftmaxCrossEntropy()
sigmoid_cross_entropy = SigmoidCrossEntropy()
