if __name__ == '__main__':
    import sys
    import pathlib
    p = pathlib.Path().absolute()
    print("Adding path: ", p)
    sys.path.append(str(p))

import torch
from src.models.sequence.modules.s4block import S4Block

def test_state(random_init=False, **kwargs):
    # B = 1
    # H = 64
    # N = 64
    # L = 1024
    B = 2
    H = 3
    N = 4
    L = 8
    s4 = S4Block(H, d_state=N, l_max=L, **kwargs)
    s4.to(device)
    s4.eval()
    # for module in s4.modules():
    #     if hasattr(module, '_setup_step'): module._setup_step()
    s4.setup_step()

    u = torch.ones(B, H, L).to(device)
    initial_state = s4.default_state(B)
    if random_init:
        if initial_state.size(-1) == N:
            initial_state = initial_state[..., :N//2]
            initial_state = torch.randn_like(initial_state)
            initial_state = torch.cat([initial_state, initial_state.conj()], dim=-1)
        else:
            initial_state = torch.randn_like(initial_state)

    state = initial_state.clone()
    y, final_state = s4(u, state=state)
    print("output:\n", y, y.shape)
    print("final state:\n", final_state, final_state.shape)

    # Use Stepping
    # for module in s4.modules():
    #     if hasattr(module, '_setup_step'): module._setup_step()
    s4.setup_step()
    state = initial_state.clone()
    ys = []
    for u_ in torch.unbind(u, dim=-1):
        y_, state = s4.step(u_, state=state)
        ys.append(y_)
    ys = torch.stack(ys, dim=-1)
    print("step outputs:\n", ys)
    print("step final state:\n", state)
    
    # return

    # Use Chunking

    chunks = 4
    state = initial_state.clone()
    ys = []
    for u_ in u.chunk(chunks, dim=-1):
        y_, state = s4(u_, state=state)
        ys.append(y_)
    ys = torch.cat(ys, dim=-1)
    print("chunk outputs:\n", ys)
    print("chunk final state:\n", state)
    # print("chunk output error:")
    # utils.compare_outputs(y, ys)
    # print("chunk final state error:")
    # utils.compare_outputs(final_state, state)


if __name__ == '__main__':
    # from benchmark import utils
    torch.manual_seed(42)

    device = 'cpu' # 'cuda'
    device = torch.device(device)

    # test_state(random_init=True, mode='dense', init='legt', rank=2, channels=2)
    # test_state(random_init=True, mode='dplr', init='legt', rank=2, channels=2)
    # test_state(random_init=False, mode='diag', init='legs', rank=1)
    test_state(random_init=False, mode='diag', init='legs', rank=1, disc='zoh', channels=3, transposed=True)
