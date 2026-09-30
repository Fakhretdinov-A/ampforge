# The submitted files

`library.fasta` and `top.fasta` here are what goes to Kaggle. They are tracked so their
hashes can be checked without a regeneration, which takes about 70 minutes.

```
library.fasta  5b505097dde766a72e954f767c241c85a86ed35c0497465d73bca1cf9e1ed43e
top.fasta      3093b43c161ce4bca70e675405b464937af125762e3c281585fe2879e2767448
```

`uv run generate` writes the same two files into `generate/`, which is not tracked.

## previous/

The earlier submission candidate, kept because it was fully verified and could have been
filed as it stood. It is also preserved at the `submission-2026-10-01` tag and on the
`submission-baseline` branch.

```
library.fasta  3dafddd63e103de6e7380969ba1d2d5cf22a08e62a75eb30258ec0e77938dbab
top.fasta      50411199e766961a3a590b0122b7a0bc34373a21eabcc96774524893e70e9121
```

Three changes separate the two, each adopted on a measurement recorded in
`EXPERIMENTS.md`: the composite weights moved from 0.45/0.35/0.20 to 0.60/0.30/0.10, the
selection blend from 15:5 to 12:8, and the sampling temperature from 1.00 to 1.10.

On the libraries the two are close to a wash: the newer one is better on the three
distance metrics and on authenticity, the older one on the four coverage metrics and on
conformity, every difference a few percent against an eightfold lead over the official
baseline.

The shortlist is what decided it. The newer one raises the predicted Gram-positive success
rate from 0.810 to 0.877 and its Gram-positive MIC from 4.41 to 2.87 µM, improves the
overall success rate from 0.903 to 0.925, is more novel (maximum identity 0.615 against
0.643) and less self-similar (0.524 against 0.619), at the cost of two points of safety
window out of 119.
