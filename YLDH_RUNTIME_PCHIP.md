# Runtime PCHIP contract

All table entries, coordinates after `log`, weights, slopes, intermediate
axis results, and returned values are `float32`. Round after every one-axis
evaluation. The accepted validation evaluates axes in this order:

1. pitch cosine `xi`;
2. `log(u)`;
3. `log(R)`.

For an axis `x[0:n]` and a query `q`, reject `q` outside the closed axis.
Choose `k` with `x[k] <= q <= x[k+1]` and
`t=float32((q-x[k])/(x[k+1]-x[k]))`. A value within
`2*epsilon_float32*max(1,abs(x[i]))` of a node is snapped to that node.
If `t=0` or `t=1`, return the exact node and use no neighbor on that axis.
Otherwise use indices
`max(0,k-1):min(n,k+3)`: four nodes internally and three at an end.

For one PCHIP line, define

```text
h[i]     = x[i+1] - x[i]
delta[i] = (y[i+1] - y[i]) / h[i]
```

At an interior node `i`, set `d[i]=0` if either adjacent slope is zero or
their signs differ. Otherwise,

```text
w1   = 2*h[i] + h[i-1]
w2   = h[i] + 2*h[i-1]
d[i] = (w1+w2) / (w1/delta[i-1] + w2/delta[i])
```

At the lower end,

```text
d[0] = ((2*h[0]+h[1])*delta[0] - h[0]*delta[1]) / (h[0]+h[1])
```

Set `d[0]=0` if its sign differs from `delta[0]`. If
`sign(delta[0]) != sign(delta[1])` and
`abs(d[0]) > 3*abs(delta[0])`, set `d[0]=3*delta[0]`.
Use the mirrored formula and tests at the upper end.

Evaluate the interval with the float32 Hermite form

```text
y(q) =
  ( 2*t^3 - 3*t^2 + 1)*y[k]
+ (   t^3 - 2*t^2 + t)*h[k]*d[k]
+ (-2*t^3 + 3*t^2    )*y[k+1]
+ (   t^3 -   t^2    )*h[k]*d[k+1]
```

Apply that one-dimensional operation to every required pitch line, store
those results in a float32 array, apply it to speed, round again, then apply
it to resonance and round the returned scalar.

Apply PCHIP independently to both drift entries and to `L11`, `L21`, and
`L22`. Reconstruct the covariance rate only as `L*transpose(L)`, which
preserves positive semidefiniteness. Trilinearly interpolate the two support
fractions for diagnostics.

Before PCHIP, inspect the strict eight corners of the queried cell for each
mode. If the corners used by the query contain both unsupported and supported
entries, reject the lookup. On an exact node or face, inspect only the corners
with nonzero trilinear weight. PCHIP neighbor nodes determine monotone slopes
only and do not relax this eight-corner rejection.

After lookup, one external wave-power amplitude `A >= 0` scales drift by `A`
and the lower factor by `sqrt(A)`.
