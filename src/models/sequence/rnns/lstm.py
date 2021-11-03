""" Wrapper around nn.LSTM to make it compatible with our RNN interface """

from torch import nn
from models.sequence import SequenceModule
from einops import rearrange
import src.models.nn.utils as U

class TorchLSTM(nn.LSTM, SequenceModule):
    """ Wrapper around nn.LSTM to make it compatible with our RNN interface """

    def __init__(self, d_model, d_hidden, n_layers=1, **kwargs):
        # Rename input_size, hidden_size to d_input, d_model
        # Set batch_first as default as per this codebase's convention
        self.d_model = d_model
        self.n_layers = n_layers
        super().__init__(d_model, d_hidden, num_layers=n_layers, batch_first=True, **kwargs)

    # def forward(self, inputs, *args, **kwargs):
    #     output, (h_n, c_n) = super().forward(inputs)
    #     return output, (h_n, c_n)

    def step(self, x, state):
        raise NotImplementedError("Needs to be implemented.")

    def default_state(self, *args, **kwargs):
        # TODO unimplemented
        """
        Snippet from nn.LSTM source
        # https://pytorch.org/docs/stable/_modules/torch/nn/modules/rnn.html#LSTM
        num_directions = 2 if self.bidirectional else 1
        real_hidden_size = self.proj_size if self.proj_size > 0 else self.hidden_size
        h_zeros = torch.zeros(self.num_layers * num_directions,
                              max_batch_size, real_hidden_size,
                              dtype=input.dtype, device=input.device)
        c_zeros = torch.zeros(self.num_layers * num_directions,
                              max_batch_size, self.hidden_size,
                              dtype=input.dtype, device=input.device)
        hx = (h_zeros, c_zeros)
        """
        raise NotImplementedError("Needs to be implemented.")

    @property
    def d_state(self):
        return self.n_layers * self.d_model

    @property
    def d_output(self):
        return self.d_model

    @property
    def state_to_tensor(self):
        if self.n_layers == 1:
            return lambda state: state[0]
        else:
            return lambda state: rearrange(state[0], 'd b h -> b (d h)')

# Handle 'transposed' argument and absorb extra args in forward
TorchLSTM = U.Transpose(U.TupleModule(TorchLSTM))
