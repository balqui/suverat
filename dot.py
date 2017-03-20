""" 
Author: Jose L Balcazar, ORCID (feb 2017)
License: MIT 

2D SVM toy Suverat, auxiliary functions related to dot product:
- dot product itself
- check for margin violation
"""

from math import log, sqrt

from colors import white

def dot(p, q):
    "2D dot product"
    return p[0]*q[0] + p[1]*q[1]

def violates_margin(p, nv, q, lab):
    """ whether point q with label lab = +/- 1 violates the margin wrt hplane
        through p with normal vector nv (nv not necessarily normalized)
    """
    norm_nv = dot(nv,nv)
    if lab > 0:
        "q should be in the positive side and not violate margin"
        return dot((q[0]-p[0], q[1]-p[1]), nv) < 0.5*norm_nv - 0.001
    else:
        return dot((q[0]-p[0], q[1]-p[1]), nv) > -0.5*norm_nv + 0.001
