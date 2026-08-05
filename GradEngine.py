
class Tensor:
    def __init__(self,data, prev=None):
        self.data = data
        self.prev = [] if prev is None else prev
        self._back = lambda:None
        self.grad = 0.0 

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self,other):
        out = Tensor(self.data + other.data, [self,other])

        def _back():
            self.grad  += out.grad * 1.0 
            other.grad += out.grad * 1.0 
        out._back = _back
        return out

    def __mul__(self,other):
        out = Tensor(self.data * other.data,[self,other])

        def _back():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._back = _back
        return out

    def _reLU(self):
        x = self.data
        out = Tensor(max(0.0, x), [self])

        def _back():
            self.grad += (1 if x > 0 else 0) * out.grad
        out._back = _back
        return out


#inputs 
x1 = Tensor(2.0)
x2 = Tensor(0.0)


#weights
w1 = Tensor(-3.0)
w2 = Tensor(1.0)

#bias
b= Tensor(6.7)


x1w1 = x1 * w1 
x2w2 = x2 * w2

x1w1x2w2 = x1w1 + x2w2

n = x1w1x2w2 + b 

o = n._reLU()

print(n)
print(o)