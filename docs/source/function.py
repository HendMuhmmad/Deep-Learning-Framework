class Function: 
    """
    Abstract model of a differentiable function.
    """
    def __init__(self, *args, **kwargs): 
        """
        :Description:
        initializing cache for intermediate results(gradients/adjoints),intermediate results are used with backprop
        """
        self.cache = {}
        self.grad = {} 
        
    def __call__(self, *args, **kwargs): 

        # calculating output
        output = self.forward(*args, **kwargs)
        # calculating and caching local gradients
        self.grad = self.local_grad(*args, **kwargs)

        return output

    def forward(self, *args, **kwargs):
        """
        :Description:Forward pass of the function. Calculates the output value (label) and the
        gradient at the input as well.
        """
        pass

    def backward(self, *args, **kwargs):
        """
        :Description:Backward pass. Computes the local gradient at the input value
        after forward pass.
        """
        pass

    def local_grad(self, *args, **kwargs):
        """
        :Description:Calculates the local gradients of the function at the given input.

        :Returns:dictionary of local gradients
        """
        pass