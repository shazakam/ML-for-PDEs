import torch

class ConjugateGradient:
    """Iterative solver for symmetric positive-definite (SPD) linear systems.

    Implements the standard Conjugate Gradient (CG) algorithm, which finds
    the solution to   A @ x = b   by minimising the A-norm of the error over
    a sequence of Krylov subspaces.  Convergence is guaranteed in at most n
    iterations for an n×n SPD matrix in exact arithmetic.

    .. note::
        All tensors passed to :meth: solve  must share the same dtype.
        For large or ill-conditioned systems,   torch.float64   is strongly
        recommended to avoid precision loss.
    """

    def __init__(self, tolerance : float, max_iterations : int):
        """Initialise the solver with stopping criteria.

        :param tolerance: Absolute L2-norm threshold on the residual
            ||b - A @ x||.  The solver stops as soon as the residual
            drops below this value.
        :type tolerance: float
        :param max_iterations: Maximum number of CG iterations before the
            solver returns the best solution found so far.
        :type max_iterations: int
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations
        pass

    def solve(self, A : torch.Tensor, b : torch.Tensor, xk : torch.Tensor)-> torch.Tensor:
        """Solve the SPD linear system A @ x = b.

        Runs the Conjugate Gradient iteration starting from the initial guess
          xk  , terminating early when the residual norm falls below
          self.tolerance   or   self.max_iterations   steps are reached.

        :param A: Square symmetric positive-definite coefficient matrix of
            shape   (n, n)  .
        :type A: torch.Tensor
        :param b: Right-hand side vector of shape   (n,)  .
        :type b: torch.Tensor
        :param xk: Initial guess vector of shape   (n,)  .  Pass
              torch.zeros(n)   for a cold start.
        :type xk: torch.Tensor
        :returns: Approximate solution vector of shape   (n,)  .
        :rtype: torch.Tensor
        """
        r_prev  = b - A @ xk 

        if torch.linalg.vector_norm(r_prev) < self.tolerance: # type: ignore
            return xk
        else:
            p = r_prev

            for _ in range(self.max_iterations):
                AkP = A @ p
                alpha = torch.inner(r_prev,r_prev) / torch.inner(p,AkP)
                xk = xk + alpha*p
                r_next = r_prev - alpha*AkP
                if torch.linalg.vector_norm(r_next) < self.tolerance: # type: ignore
                    break

                bk = torch.inner(r_next,r_next) / torch.inner(r_prev, r_prev)
                p = r_next + bk*p
                r_prev = r_next

        return xk