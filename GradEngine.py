
class Tensor:
    def __init__(self,data, prev=None):
        self.data = data
        self.prev = [] if prev is None else prev
        self.back = lambda:None
        self.grad = 0.0 

    def __repr__(self):
        return f"Tensor(data={self.data}, grad={self.grad})"

    def __add__(self,other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, [self,other])

        def _back():
            self.grad  += out.grad * 1.0 
            other.grad += out.grad * 1.0 
        out.back = _back
        return out

    def __mul__(self,other):
        other = other if isinstance(other,Tensor) else Tensor(other)
        
        out = Tensor(self.data * other.data,[self,other])

        def _back():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out.back = _back
        return out

    def _reLU(self):
        x = self.data
        out = Tensor(max(0.0, x), [self])

        def _back():
            self.grad += (1 if x > 0 else 0) * out.grad
        out.back = _back
        return out

    def backward(self):
        topo = []
        visited  =  set()

        def topo_sort(value):
            if value not in visited:
                visited.add(value)
                for child in value.prev:
                    topo_sort(child)
                topo.append(value)
        topo_sort(self)
        self.grad = 1.0

        for node in reversed(topo):
            node.back()

    def __rmul__(self,other):
        return self * other

    def __radd__(self,other):
        return self + other
        
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

o.backward()


print(x1)
print(x2)
print(w1)
print(w2)