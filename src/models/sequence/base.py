from torch import nn

class SequenceModule(nn.Module):
    """ Abstract sequence model class. All layers that the backbones use must adhere to this

    A sequence model is a layer that transforms an input of shape
    (n_batch, l_sequence, d_input) to (n_batch, l_sequence, d_output)

    Additionally, it returns a "state" which can be any additional information
    For example, RNN and SSM layers may return their hidden state,
    while some types of transformer layers (e.g. Transformer-XL) may want to pass through state as well

    - default_state receives a batch_shape with device and returns an initial state
    - step simulates a single step of the sequence (e.g. one unroll for an RNN). It receives a state and single input (n_batch, d_input) and returns a state and output (n_batch, d_output)
    - forward is a sequence-to-sequence transformation that receives an optional state
    """

    # def __init__(self, transposed=False, *args, **kwargs):
    #     """ model should support regular (B, L, H) and transposed (B, H, L) axes ordering """
    #     self.transposed = transposed

    @property
    def d_output(self):
        return self._d_output
    @d_output.setter
    def d_output(self, d):
        self._d_output = d

    @property
    def state_to_tensor(self):
        """ Returns a function mapping a state to a single tensor, in case one wants to use the hidden state instead of the output for final prediction """
        return lambda _: None

    @property
    def d_state(self):
        """ Returns dimension of output of self.state_to_tensor """
        return None

    @property
    def transposed(self):
        return self._transposed
    @transposed.setter
    def transposed(self, x):
        self._transposed = x


    def default_state(self, *batch_shape, device=None): # TODO device shouldn't be needed; models should store their own initial state at initialization
        return None

    def step(self, x, state=None, *args, **kwargs):
        return x, state

    def forward(self, x, state=None, *args, **kwargs):
        return x, state

def Transpose(module):
    """ Wrap a SequenceModule class to transpose the forward pass """
    # TODO maybe possible with functools.wraps
    class WrappedModule(module):
        def __init__(self, *args, transposed=False, **kwargs):
            super().__init__(*args, **kwargs)
            self.transposed = transposed

        def forward(self, x, *args, **kwargs):
            if self.transposed: x = x.transpose(-1, -2)
            x, state = super().forward(x)
            if self.transposed: x = x.transpose(-1,-2)
            return x, state
    # https://stackoverflow.com/questions/5352781/how-to-set-class-names-dynamically
    WrappedModule.__name__ = module.__name__
    return WrappedModule

class SequenceIdentity(SequenceModule):
    def __init__(self, d_model, dropout=0.0):
        super().__init__()
        self.d_output = d_model

    def forward(self, x, state=None, *args, **kwargs):
        return x, state
SequenceIdentity = Transpose(SequenceIdentity)
