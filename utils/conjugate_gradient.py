import torch

class ConjugateGradient:

    def __init__(self, tolerance : float, max_iterations : int):
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        pass

    def solve(self, A : torch.Tensor, b : torch.Tensor, xk : torch.Tensor)-> torch.Tensor:
        r_prev  = b - A @ xk 

        if torch.linalg.vector_norm(r_prev) < self.tolerance:
            return xk
        else:
            p = r_prev

            for i in range(self.max_iterations):
                AkP = A @ p
                alpha = torch.inner(r_prev,r_prev) / torch.inner(p,AkP)
                xk = xk + alpha*p
                r_next = r_prev - alpha*AkP
                if torch.linalg.vector_norm(r_next) < self.tolerance:
                    break

                bk = torch.inner(r_next,r_next) / torch.inner(r_prev, r_prev)
                p = r_next + bk*p
                r_prev = r_next

        return xk