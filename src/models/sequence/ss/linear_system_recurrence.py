""" Earlier version of LSSL module that uses pure recurrence (with variable step sizes) """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda")


class LinearSystem(nn.Module):
    def __init__(
        self,
        N,
        transition,
        C,
        D,
    ):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        C: (..., M, N)
        D: (..., M)
        """
        super().__init__()
        self.N = N
        self.transition = transition

        self.C = C
        self.D = D

    def forward(self, dt, u, x_=None):
        """
        u : (length, ...)
        x : (..., N)
        Returns
        y : (length, ..., M)
        """

        if x_ is None:
            x_ = u.new_zeros(u.shape[1:] + (self.N,))
        ys = []
        for dt_, u_ in zip(dt, u):
            x_ = self.transition.bilinear(dt_, x_, u_)  # (..., N)
            y = (self.C @ x_.unsqueeze(-1)).squeeze(
                -1
            )  # TODO can use sum instead of matmul if M = 1
            ys.append(y)
        y = torch.stack(ys, dim=0)
        v = u.unsqueeze(-1) * self.D  # (L, ..., M)
        y = y + v  # (L, ..., M)
        return y, x_

    def adjoint_input(self, dy, dt):
        """Computes adjoint to the input u

        dy: (L, ..., M)
        dt: (L, ...)
        """

        # Compute dx_
        dx_ = torch.sum(dy[-1].unsqueeze(-1) * self.C, dim=-2)  # (..., N)

        dyC = (self.C.transpose(-1, -2) @ dy.unsqueeze(-1)).squeeze(
            -1
        )  # C^T dy (L, ..., N)
        dyD = torch.sum(dy * self.D, dim=-1)  # D^T dy (L, ...)
        du = []
        for dt_, dyC_ in zip(dt.flip(0), dyC.flip(0)):
            dx_ = self.transition.inverse_mult(dx_, dt_ / 2, transpose=True)  # (..., N)
            du_ = torch.sum(self.transition.B * dx_, dim=-1)  # (...)
            du_ = dt_ * du_  # (...)
            dx_ = (
                self.transition.forward_mult(dx_, dt_ / 2, transpose=True) + dyC_
            )  # (..., N)
            du.append(du_)
        du = torch.stack(du, dim=0)  # (L, ...)
        du = du.flip(0)
        du = du + dyD
        return du

    def adjoint_projection(self, dy, dt, u):
        """Computes adjoint to the projection parameters C, D

        dy: (L, ..., M)
        u: (L, ...)
        dt: (L, ...)
        """

        dC = torch.zeros_like(self.C)
        x_ = u.new_zeros(u.shape[1:] + (self.N,))
        for dt_, u_, dy_ in zip(dt, u, dy):
            x_ = self.transition.bilinear(dt_, x_, u_)  # (..., N)
            dC_ = dy_.unsqueeze(-1) * x_.unsqueeze(-2)  # (..., M, N)
            dC += dC_.view((-1,) + self.C.shape).sum(dim=0)  # (M, N)
        dD = dy * u.unsqueeze(-1)  # (L, ..., M)
        dD = dD.view((-1,) + self.D.shape).sum(dim=0)  # (M,)
        return dC, dD


class LinearSystemStepsize(nn.Module):
    def __init__(
        self,
        N,
        transition,
        C,
        D,
    ):
        """
        N: the order of the HiPPO projection
        dt: discretization step size - should be roughly inverse to the length of the sequence
        """
        super().__init__()
        self.N = N
        self.transition = transition

        self.C = C
        self.D = D

    def forward(self, dt, u, x=None):
        """
        u : (length, ...)
        x : (..., N)
        Returns
        y : (length, ..., M)
        """

        v = u.unsqueeze(-1) * self.D  # (L, ..., M)

        if x is None:
            x = u.new_zeros(u.shape[1:] + (self.N,))
        ys = []
        for dt_, u_ in zip(dt, u):
            x = self.transition.bilinear(dt_, x, u_)  # (..., N)
            y = (self.C @ x.unsqueeze(-1)).squeeze(
                -1
            )  # TODO can use sum instead of matmul if M = 1
            ys.append(y)
        y = torch.stack(ys, dim=0)
        y = y + v  # (L, ..., M)
        return y, x

    def adjoint(self, dy, x_, dt, u):
        """
        gradient:
        dy: (L, ..., M)

        state:
        # dx_: (..., N)
        x: (..., N)

        cached arguments:
        dt: (L, ...)
        u: (L, ...)
        """

        dx_ = torch.sum(dy[-1].unsqueeze(-1) * self.C, dim=-2)  # (..., N)

        dyC = (self.C.transpose(-1, -2) @ dy.unsqueeze(-1)).squeeze(
            -1
        )  # C^T dy (L, ..., N)
        dyD = torch.sum(dy * self.D, dim=-1)  # D^T dy (L, ...)

        dC = torch.zeros_like(self.C)
        dD = torch.zeros_like(self.D)
        du = []
        ddt = []
        for dt_, dyC_, u_, dy_ in zip(dt.flip(0), dyC.flip(0), u.flip(0), dy.flip(0)):
            # dy_: (..., M)
            # x_: (..., N)
            # u_, dt_: (...)
            dC_ = dy_.unsqueeze(-1) * x_.unsqueeze(-2)  # (..., M, N)
            dC += dC_.view((-1,) + self.C.shape).sum(dim=0)  # (M, N)
            dD_ = dy_ * u_.unsqueeze(-1)  # (..., M)
            dD += dD_.view((-1,) + self.D.shape).sum(dim=0)  # (M,)

            dx_ = self.transition.inverse_mult(dx_, dt_ / 2, transpose=True)  # (..., N)

            # Compute du
            du_ = torch.sum(self.transition.B * dx_, dim=-1)  # (...)
            du_ = dt_ * du_  # (...)
            du.append(du_)

            x_prev = self.transition.bilinear(-dt_, x_, u_)  # (..., N)
            ddt_ = self.transition.quadratic(dx_, 0.5 * (x_prev + x_))  # (...)
            ddt_ = ddt_ + torch.sum(self.transition.B * dx_, dim=-1) * u_
            ddt.append(ddt_)  # (...)
            x_ = x_prev

            dx_ = (
                self.transition.forward_mult(dx_, dt_ / 2, transpose=True) + dyC_
            )  # (..., N)

        du = torch.stack(du, dim=0).flip(0)  # (L, ...)
        du = du + dyD

        ddt = torch.stack(ddt, dim=0).flip(0)  # (L, ...)

        # Sanity check
        # print(f"{x_=}") # should be 0 (initial state)

        return du, ddt, dC, dD


class LinearSystemFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, u, C, D, transition):
        """
        dt : (L, ...)
        u : (L, ...)
        C : (M, N)
        D : (M,)
        transition: Transition objective implementing forward_mult, inverse_mult, bilinear, quadratic

        Returns:
        y : (L, ..., M)
        """
        ctx.transition = transition
        ctx.save_for_backward(dt, u, C, D)

        with torch.no_grad():
            if x is None:
                x = u.new_zeros(u.shape[1:] + (transition.N,))
            ys = []
            for dt_, u_ in zip(dt, u):
                # breakpoint()
                x = transition.bilinear(dt_, x, u_)  # (..., N)
                y = (C @ x.unsqueeze(-1)).squeeze(
                    -1
                )  # TODO can use sum instead of matmul if M = 1
                ys.append(y)
            y = torch.stack(ys, dim=0)
            # breakpoint()
            v = u.unsqueeze(-1) * D  # (L, ..., M)
            y = y + v  # (L, ..., M)
        return y

    @staticmethod
    def backward(ctx, dy):
        """Computes adjoint to the input u

        dy: (L, ..., M)
        """
        dt, u, C, D = ctx.saved_tensors
        transition = ctx.transition

        with torch.no_grad():

            # Compute dx_
            dx_ = torch.sum(dy[-1].unsqueeze(-1) * C, dim=-2)  # (..., N)

            # Compute du
            dyC = (C.transpose(-1, -2) @ dy.unsqueeze(-1)).squeeze(
                -1
            )  # C^T dy (L, ..., N)
            dyD = torch.sum(dy * D, dim=-1)  # D^T dy (L, ...)
            du = []
            for dt_, dyC_ in zip(dt.flip(0), dyC.flip(0)):
                dx_ = transition.inverse_mult(dx_, dt_ / 2, transpose=True)  # (..., N)
                du_ = torch.sum(transition.B * dx_, dim=-1)  # (...)
                du_ = dt_ * du_  # (...)
                dx_ = (
                    transition.forward_mult(dx_, dt_ / 2, transpose=True) + dyC_
                )  # (..., N)
                du.append(du_)
            du = torch.stack(du, dim=0)  # (L, ...)
            du = du.flip(0)
            du = du + dyD

            # Compute dC, dD
            dC = torch.zeros_like(C)
            x_ = u.new_zeros(u.shape[1:] + (transition.N,))
            for dt_, u_, dy_ in zip(dt, u, dy):
                x_ = transition.bilinear(dt_, x_, u_)  # (..., N)
                dC_ = dy_.unsqueeze(-1) * x_.unsqueeze(-2)  # (..., M, N)
                dC += dC_.view((-1,) + C.shape).sum(dim=0)  # (M, N)
            dD = dy * u.unsqueeze(-1)  # (L, ..., M)
            dD = dD.view((-1,) + D.shape).sum(dim=0)  # (M,)

        if not ctx.needs_input_grad[0]:
            dx_ = None
        return dx_, None, du, dC, dD, None


linearsystem = LinearSystemFunction.apply


class LinearSystemStepsizeFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dt, u, C, D, transition):
        """
        dt : (L, ...)
        u : (L, ...)
        C : (M, N)
        D : (M,)
        transition: Transition objective implementing forward_mult, inverse_mult, bilinear, quadratic

        Returns:
        y : (L, ..., M)
        """
        ctx.transition = transition
        # ctx.save_for_backward(dt, u, C, D)

        v = u.unsqueeze(-1) * D  # (L, ..., M)

        if x is None:
            x = u.new_zeros(u.shape[1:] + (transition.N,))
        ys = []
        for dt_, u_ in zip(dt, u):
            x = transition.bilinear(dt_, x, u_)  # (..., N)
            y = (C @ x.unsqueeze(-1)).squeeze(
                -1
            )  # TODO can use sum instead of matmul if M = 1
            ys.append(y)
        y = torch.stack(ys, dim=0)
        y = y + v  # (L, ..., M)

        ctx.save_for_backward(dt, u, C, D, x)
        return y

    @staticmethod
    def backward(ctx, dy):
        """
        gradient:
        dy: (L, ..., M)

        state:
        # dx_: (..., N)
        x: (..., N)

        cached arguments:
        dt: (L, ...)
        u: (L, ...)
        """

        # dt, u, C, D = ctx.saved_tensors
        dt, u, C, D, x_ = ctx.saved_tensors
        transition = ctx.transition

        # Compute dx_
        dx_ = torch.sum(dy[-1].unsqueeze(-1) * C, dim=-2)  # (..., N)

        dyC = (C.transpose(-1, -2) @ dy.unsqueeze(-1)).squeeze(-1)  # C^T dy (L, ..., N)
        dyD = torch.sum(dy * D, dim=-1)  # D^T dy (L, ...)

        dC = torch.zeros_like(C)
        dD = torch.zeros_like(D)
        du = []
        ddt = []
        for dt_, dyC_, u_, dy_ in zip(dt.flip(0), dyC.flip(0), u.flip(0), dy.flip(0)):
            # dy_: (..., M)
            # x_: (..., N)
            # u_, dt_: (...)
            dC_ = dy_.unsqueeze(-1) * x_.unsqueeze(-2)  # (..., M, N)
            dC += dC_.view((-1,) + C.shape).sum(dim=0)  # (M, N)
            dD_ = dy_ * u_.unsqueeze(-1)  # (..., M)
            dD += dD_.view((-1,) + D.shape).sum(dim=0)  # (M,)

            dx_ = transition.inverse_mult(dx_, dt_ / 2, transpose=True)  # (..., N)

            # Compute du
            du_ = torch.sum(transition.B * dx_, dim=-1)  # (...)
            du_ = dt_ * du_  # (...)
            du.append(du_)

            x_prev = transition.bilinear(-dt_, x_, u_)  # (..., N)
            ddt_ = transition.quadratic(dx_, 0.5 * (x_prev + x_))  # (...)
            ddt_ = ddt_ + torch.sum(transition.B * dx_, dim=-1) * u_
            ddt.append(ddt_)  # (...)
            x_ = x_prev

            dx_ = (
                transition.forward_mult(dx_, dt_ / 2, transpose=True) + dyC_
            )  # (..., N)

        du = torch.stack(du, dim=0).flip(0)  # (L, ...)
        du = du + dyD

        ddt = torch.stack(ddt, dim=0).flip(0)  # (L, ...)

        # Sanity check
        # print(f"{x_=}") # should be 0 (initial state)

        if not ctx.needs_input_grad[0]:
            dx_ = None
        return dx_, ddt, du, dC, dD, None


linearsystemstepsize = LinearSystemStepsizeFunction.apply
