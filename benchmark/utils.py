""" Useful functions for writing test code. """

import torch
import torch.utils.benchmark as B

def benchmark(fn, *inputs, T=1, memory=False, mode='forward', desc='', **kwinputs):
    if memory:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    if mode == 'forward': stmt = 'fn(*inputs, **kwinputs)'
    elif mode == 'combined': stmt = 'fn(*inputs, **kwinputs).sum().backward(retain_graph=True)'
    else: raise NotImplementedError

    print(desc, '--', mode)
    t0 = B.Timer(
            stmt=stmt,
            globals={'fn': fn, 'inputs': inputs, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    print(t0.timeit(T))

    if memory:
        print(f'{desc} max memory allocated: ', torch.cuda.max_memory_allocated() / 2**20)
        torch.cuda.empty_cache()

def benchmark_forward(T, fn, *inputs, desc='', verbose=True, **kwinputs):
    """ Use Pytorch Benchmark on the forward pass of an arbitrary function. """
    if verbose:
        print(desc, '- Forward pass')
    t = B.Timer(
            stmt='fn(*inputs, **kwinputs)',
            globals={'fn': fn, 'inputs': inputs, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(T)
    if verbose:
        print(m)
    return t, m


def benchmark_backward(T, fn, *inputs, grad=None, desc='', verbose=True, **kwinputs):
    """ Use Pytorch Benchmark on the backward pass of an arbitrary function. """
    if verbose:
        print(desc, '- Backward pass')
    y = fn(*inputs, **kwinputs)
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError('Grad shape does not match output shape')
    t = B.Timer(
            stmt='y.backward(grad, retain_graph=True)',
            globals={'y': y, 'grad': grad},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(T)
    if verbose:
        print(m)
    return t, m


def benchmark_combined(T, fn, *inputs, grad=None, desc='', verbose=True, **kwinputs):
    """ Use Pytorch Benchmark on the forward+backward pass of an arbitrary function. """
    if verbose:
        print(desc, 'Forward + Backward pass')
    y = fn(*inputs, **kwinputs)
    if grad is None:
        grad = torch.randn_like(y)
    else:
        if grad.shape != y.shape:
            raise RuntimeError('Grad shape does not match output shape')
    del y
    t = B.Timer(
            stmt='fn(*inputs, **kwinputs).backward(grad, retain_graph=True)',
            globals={'fn': fn, 'inputs': inputs, 'grad': grad, 'kwinputs': kwinputs},
            num_threads=torch.get_num_threads(),
            )
    m = t.timeit(T)
    if verbose:
        print(m)
    return t, m


def benchmark_all(T, fn, *inputs, grad=None, desc='', verbose=True, **kwinputs):
    """ Use Pytorch Benchmark on the forward+backward pass of an arbitrary function. """
    return (
        benchmark_forward(T, fn, *inputs, desc=desc, verbose=verbose, **kwinputs),
        benchmark_backward(T, fn, *inputs, grad=grad, desc=desc, verbose=verbose, **kwinputs),
        benchmark_combined(T, fn, *inputs, grad=grad, desc=desc, verbose=verbose, **kwinputs),
    )


def pytorch_profiler(T, fn, *inputs):
    """ Wrap benchmark functions in Pytorch profiler to see CUDA information. """

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
                ],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
            ) as p:
            # benchmark_forward(T, fn, *inputs)
            fn(*inputs)

    print(p.key_averages().table(
        sort_by="self_cuda_time_total", row_limit=-1))

def convert_data(*tensors, device='cuda'):
    tensors = tuple(t.to(device) for t in tensors)
    for t in tensors:
        if t.is_leaf: t.requires_grad = True
        t.retain_grad()
    return tensors

def log_backward(output, *inputs):
    """ Perform backward pass of output and print gradients of input tensors. """

    print(f"{output=}")
    output.sum().backward(retain_graph=True)
    print("Gradients:")
    for t in inputs:
        print(t.grad)
        t.grad.zero_()

def benchmark_memory(fn, *inputs, desc='', **kwinputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn(*inputs, **kwinputs)
    mem = torch.cuda.max_memory_allocated()
    print(f'{desc} memory allocated: ', torch.cuda.memory_allocated() / 2**20)
    print(f'{desc} max memory allocated: ', torch.cuda.max_memory_allocated() / 2**20)
    print(f'{desc} memory reserved: ', torch.cuda.memory_reserved() / 2**20)
    print(f'{desc} max memory reserved: ', torch.cuda.max_memory_reserved() / 2**20)
    torch.cuda.empty_cache()
    return mem

def benchmark_memory_combined(fn, *inputs, desc='', **kwinputs):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn(*inputs, **kwinputs).sum().backward(retain_graph=True)
    print(f'{desc} memory allocated: ', torch.cuda.memory_allocated() / 2**20)
    print(f'{desc} max memory allocated: ', torch.cuda.max_memory_allocated() / 2**20)
    print(f'{desc} memory reserved: ', torch.cuda.memory_reserved() / 2**20)
    print(f'{desc} max memory reserved: ', torch.cuda.max_memory_reserved() / 2**20)
    torch.cuda.empty_cache()
    # return mem

def compare_outputs(*outputs, full=False, relative=True):
    out = outputs[0]
    print("output shape", out.shape)
    if full:
        print("output", out)
    print("max, mean output entry:", torch.max(torch.abs(out)), torch.mean(torch.abs(out)))

    for output in outputs[1:]:
        if output.shape != out.shape:
            print("SHAPE MISMATCH", output.shape)
        if relative:
            err = torch.abs(output-out)
        else:
            err = torch.abs((output-out)/out)
        if full: print("full err", err)
        print(f"max, mean sq {'rel' if relative else ''} err:", torch.max(err), torch.mean(err**2))
