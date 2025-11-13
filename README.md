# Measuring Numerical Errors with Hyb Error

Hyb Error `|x-y| / (1+|y|)` is a smoothed relative error that approaches absolute error as `|y|`  approaches 0. It is a pragmatic metric to measure numerical errors of floating-point numbers.

## Maximum difference (MaxDiff) between two arrays

You can use the following function to measure the maximum difference (MaxDiff) between two arrays.
```
def get_max_diff(A: torch.Tensor, B: torch.Tensor):
    A = A.double()
    B = B.double()
    diff = (A-B).abs() / (1+B.abs())
    return diff.max()
```

This MaxDiff has the property that if `eps = get_max_diff(A, B)`, then `torch.allclose(A, B, eps, eps) == True`.

## Mean difference (MeanDiff) between two arrays

You can use the following function to measure the mean difference (MeanDiff) between two arrays.

```
def get_mean_diff(A: torch.Tensor, B: torch.Tensor):
    A = A.double()
    B = B.double()
    diff = (A-B).abs() / (1+B.abs())
    return diff.mean()
```

## Citation

```
@misc{xie_hyb_2024,
    title = {Hyb {Error}: {A} {Hybrid} {Metric} {Combining} {Absolute} and {Relative} {Errors}},
    url = {https://doi.org/10.48550/arXiv.2403.07492},
    doi = {10.48550/ARXIV.2403.07492},
    author = {Xie, Peichen},
    year = {2024},
    note = {arXiv: 2403.07492},
}
```
