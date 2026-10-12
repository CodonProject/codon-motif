import torch

_ORIGINAL = torch.linalg.solve_triangular


def row_forward_inverse(matrix):
    '''Textbook forward substitution, one row at a time (C sequential steps).'''
    size = matrix.shape[-1]
    eye = torch.eye(size, dtype=matrix.dtype, device=matrix.device)
    strict = matrix - eye
    inverse = eye.expand_as(matrix).clone()
    for i in range(size):
        inverse[..., i, :] = eye[i] - (strict[..., i:i + 1, :] @ inverse)[..., 0, :]
    return inverse


def solve_patch(a, b, *, upper=False, left=True, unitriangular=False, out=None):
    '''`torch.linalg.solve_triangular`, by forward substitution when it applies.'''
    if upper or not unitriangular or a.shape[-1] != a.shape[-2] or b.shape[-1] != a.shape[-1]:
        return _ORIGINAL(a, b, upper=upper, left=left, unitriangular=unitriangular, out=out)
    return row_forward_inverse(a) @ b


def install_solve_patch():
    '''Route `torch.linalg.solve_triangular` through `solve_patch`.'''
    torch.linalg.solve_triangular = solve_patch


def remove_solve_patch():
    '''Restore the library implementation.'''
    torch.linalg.solve_triangular = _ORIGINAL
