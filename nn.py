
#building neural nets out of value objects from engine.py 
# neuron -> layer -> mlp ,  same pattern nested three times 

import random 
from engine import Value

class Module: 
    def zero_grad(self): 
        # reset every param's grad to 0 before a new backward passs
        # (grads accumulate with +=, so must clear before each step)
        for p in self.parameters():
            p.grad=0 

    def parameters(self):
        return[] # overriden by each subclass below
    
    #neuron : one weighted sum + optional activation 

class Neuron(Module): 
        def __init__(self, nin , nonlin=True): 
            # one random weight per input 
            self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
            self.b = Value(0) #bias starts  at 0 
            self.nonlin = nonlin # apply ReLU ? false = raw output 
        
        def __call__(self,x):
            #w.x + b 
            act = sum((wi*xi for wi,xi in zip(self.w,x)), self.b)
            return act.relu() if self.nonlin else act 
        
        def parameters(self): 
            return self.w + [self.b]
        
        def __repr__(self):
            return f"{'ReLU' if self.nonlin else 'Liear'}Neuron({len(self.w)})"
        
        # LAYER : serveral neurons , same inputs ,parallel output 

class Layer(Module): 
            def __init__(self, nin, nout, **kwargs):
                self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

            def __call__ (self,x):
                out = [n(x) for n in self.neurons]
                return out[0] if len(out) == 1 else out 
                #unwrap if single neuron 0.5 instead of   [0.5]
            
            def parameters(self):
                return[p for n in self.neurons for p in n.parameters()]
            
            def __repr__(self):
                return f"Layer of[{', '.join(str(n) for n in self.neurons)}]"
            
            #MLP : stack of layers, output of one feeds into the next 

class MLP(Module):
                def __init__(self,nin,nouts):
                    sz=[nin] + nouts
                    # nonlin=False on the last layer only ->no activation on final output
                    self.layers = [Layer(sz[i], sz[i+1], nonlin=i !=len(nouts)-1) for i in range(len(nouts))]
                
                def __call__(self,x): 
                    for layer in self.layers:
                        x=layer(x)
                    return x 
                
                def parameters(self):
                    return[p for layer in self.layers for p in layer.parameters()]

                def __repr__(self):
                    return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
                


n = MLP(3,[4,4,1])
print(n)
print(n([2.0,3.0,-1.0]))