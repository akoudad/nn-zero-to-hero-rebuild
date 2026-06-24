class Value : 
    """stores a single scalar value and its gradient"""
    # Store scalar values and compute gradients automatically.
    # It doesn’t know what a neuron or neural network is.

    def __init__ ( self, data, _ children=(), _op=''):
        self.data = data
        self.grad = 0 
        #internal variables used for autograd graph construction
        self._backward = lambda: None
        self._prev= set(_children)
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def __add__(self, other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        # d(out)/d(self) =1 , d(out)/d(other) = 1
        # chain rule: push ouot's grad straight through to both inputs 

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward 

        return out 
    
    def __mul__(self,other): 
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self,other), '*')

        def _backward():
            # d(out)/d(self) = other.data, d(out)/d(other) = self.data
            # example : x = self , y = other , out = x * y
            # dout/dx = y 
            # each input's local derivative is just "the other side"
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out 
    def __pow__(self,other):
        # only numeric exponents supported - keeps the derivative simple 
        # (power rule), no need to handle Value ** Value here 
        assert insintance(other , (int,float)), "only supportung int/float powers for now"
        out = Value(self.data**other,(self,), f'**{other}')

        def _backward():
            #standard power rule: d(X^n)/dx = n* X^(n-1)
            self.grad += (other * self.data**(other-1))* out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        # activation function : passes positive values through unchanged, 
        # zeroes out negative values 
        out = Value(0 if self.data < 0 else self.data, (self,), 'ReLU')

        def _backward():
            #derivative is 1 where output was positive , 0 otherwise 
            #(out.data > 0) evaluates to True/False, acts as 1/0 here 
            self.grad += (out.data>0) * out.grad
        out._backward = _backward
        return out 
    
    #-----------------------------------------------------------------
    #block 3 - the orchestrator 
    #this is the only method that walks the whole graph
    
    def backward(self): 

        #topological sort : order nodes so every node comes after all the nodes
        #that depends on it (output first once reversed)
        # this guarantees a node's grad is fully accumlated before
        # we use it to push gradient further back to its own inputs 

        topo=[]
        visited = set()
        def build_topo(v):
            if v not in visited: 
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v) #append after children 
        build_topo(self)

        #seed : the outputs's effect on itself is always 1 (dL/dL = 1)
        # this is the starting signal that gets multiplied through 
        # every local derivative on the way back to the inputs 
        self.grad = 1 
        for v in reversed(topo): 
            v._backward()

        #------------------------------------------------------------------
        # block 4 
    def __neg__(self):          # -self        == self * -1
        return self * -1

    def __radd__(self, other):  # other + self == self + other (commutative)
        return self + other

    def __sub__(self, other):   # self - other == self + (-other)
        return self + (-other)

    def __rsub__(self, other):  # other - self == other + (-self)
        return other + (-self)

    def __rmul__(self, other):  # other * self == self * other (commutative)
        return self * other

    def __truediv__(self, other):   # self / other == self * other^-1
        return self * other**-1

    def __rtruediv__(self, other):  # other / self == other * self^-1
        return other * self**-1 


     def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"