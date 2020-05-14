from functools import reduce
import numpy as np
import cvxpy as cp
from sage.all import *

class SageCVXProblem:
    def __init__(self, list_vars, equality_constraints, inequality_constraints,
                 psd_matrices, objective, debug=False):
        """
        Construct an CVXPY SDP problem from Sage variables.
        Args:
          list_vars: list of decision variables
          equality_constraints/inequality_constraints: list of linear functions f_i in 
            the decision variables `list_vars`. Used to impose the constraint f_i == 0 or f_i >= 0
          psd_matrices: list of matrices Q_i. Used to impose the constraint Q_i psd.
          objective: linear function in the decision variables used as objective function to maximize.
        """
        
        if debug:
            print('SDP with {} variables, involving {} matrices of sizes:'\
                  .format(len(list_vars), len(psd_matrices)))
            print(list(map(lambda M: M.dimensions()[0], psd_matrices)))
            
        #self.prob = picos.Problem()
        self.list_vars = list_vars
        self.sage_equality_constraints = equality_constraints
        self.sage_inequality_constraints = inequality_constraints
        self.sage_psd_matrices = psd_matrices
        self.sage_objective = objective

        self._sage_to_cvx = {}
        self._cvx_equality_constraints = []
        self._cvx_inequality_constraints = []
        self._cvx_psd_constraints = []
        self._cvx_objective = []

        # Sage -> Cvx conversion
        self._build_cvx_vars()
        self._build_equality_constraints()
        self._build_inequality_constraints()
        self._build_psd_matrices()
        self._build_objective()
        self._build_problem()


    def _get_cvx_var_from_sage_var(self, sage_var):
        return self._sage_to_cvx.get(str(sage_var), None)

    
    def _get_cvx_expr_from_sage_expr(self, sage_expr):
        sage_expr = SR(sage_expr)
        op = sage_expr.operands()
        # if sage_expr contains an operation
        if len(op) > 0:
            return sage_expr.operator()(*map(self._get_cvx_expr_from_sage_expr, op))
        # otherwise, `sage_expr` is either a variable or a constant (i.e., a flot)
        else:
            cvx_var = self._get_cvx_var_from_sage_var(sage_expr)
            return cvx_var if cvx_var else float(sage_expr)
        
    def _build_cvx_vars(self):
        for v in self.list_vars:
            if not(self._get_cvx_var_from_sage_var(v)):
                self._sage_to_cvx[str(v)] = cp.Variable(name=str(v))
            
    def _build_equality_constraints(self):
        cvx_lhs = map(self._get_cvx_expr_from_sage_expr, self.sage_equality_constraints)

        for eq in cvx_lhs:
            self._cvx_equality_constraints.append(eq == float(0))
            
            
    def _build_inequality_constraints(self):
        cvx_lhs = map(self._get_cvx_expr_from_sage_expr, self.sage_inequality_constraints)

        for eq in cvx_lhs:
            self._cvx_inequality_constraints.append(eq >= float(0))            
                     
                
    def _build_psd_matrices(self):
        for A in self.sage_psd_matrices:
            cvx_A = [[self._get_cvx_expr_from_sage_expr(Aij)
                      for Aij in Ai]
                     for Ai in A]
            if len(cvx_A) == 1:
                self._cvx_psd_constraints.append(cvx_A[0][0] >= float(0))
                continue

            # build a cvx matrix from cvx_A
            cvx_A = [cp.hstack(Ai) for Ai in cvx_A]
            cvx_A = cp.vstack(cvx_A)

            self._cvx_psd_constraints.append(cvx_A >> float(0))

        
    def _build_objective(self):
        self._cvx_objective = cp.Maximize(self._get_cvx_expr_from_sage_expr(self.sage_objective))
        

    def _build_problem(self):
        constraints = self._cvx_psd_constraints + self._cvx_equality_constraints + self._cvx_inequality_constraints
        self.prob = cp.Problem(self._cvx_objective, constraints)


    def solve(self, **solver_params):
        """
        Solve SDP.
        Returns a tuple `(st, dict_solution)`, where `st` is the status returned by the solver, 
          and dict_solution is a dictionary mapping decision variables to their value.
        """
        objective_value = self.prob.solve(**solver_params)
        return objective_value,  {k: float(v.value) for k, v in self._sage_to_cvx.items()}

    

def symb_matrix(n, name='Q'):
    """Construct an `n` x `n` symmetric symbolic matrix named `name`."""
    template_var_names = name+"_%d%d"
    order = lambda idx: (max(idx), min(idx))
    var_names = map(lambda idx: template_var_names % order(idx), 
                    cartesian_product([range(1, n+1),]*2))
    var_names = list(var_names)
    var_names = np.matrix(var_names).reshape(n, n)
    create_sym_entry = np.vectorize(lambda name: polygen(QQ, name))
    Q = create_sym_entry(var_names)
    Q = matrix(SR, Q)
    return Q
