# Measuring Numerical Errors with Hyb Error

Hyb Error `|x-y| / (1+|y|)` is a smoothed relative error that approaches absolute error as `|y|`  approaches 0. It is a pragmatic metric to measure numerical errors of floating-point numbers. You can use the following function to get the Hyb Error between two arrays.

```python
def get_hyb_error(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    A = A.double()
    B = B.double()
    return (A-B).abs() / (1+B.abs())
```


## Maximum difference (MaxDiff) between two arrays

You can use the following function to measure the maximum difference (MaxDiff) between two arrays.

```python
def get_max_diff(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return get_hyb_error(A, B).max()
```

This MaxDiff has the property that

```python
max_diff = get_max_diff(A, B)                                # A and B are close within the absolute and relative tolerance of max_diff
assert torch.allclose(A, B, rtol=max_diff, atol=max_diff)    # always True
delta = max_diff - 1e-6                                      # but not within the absolute and relative tolerance of delta for any delta < max_diff
assert torch.allclose(A, B, rtol=delta, atol=delta) == False # always False
```

Therefore, the MaxDiff represents the **threshold of allclose tolerance**.

You can also check the values in A and B that cause the MaxDiff:

```python
idx = get_hyb_error(A, B).argmax()
print(idx, A[idx], B[idx])
```

## Mean difference (MeanDiff) between two arrays

You can use the following function to measure the mean difference (MeanDiff) between two arrays.

```python
def get_mean_diff(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    return get_hyb_error.mean()
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
