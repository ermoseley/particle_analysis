# Stage 9 equilibrium charging reference

`dust_charge_equilibrium.py` is the float64 offline reference and generator
for the compact float32 Stage 9 runtime closure.

## Cell equilibrium

The cell variable is one mean potential for the current single HD23 astrodust
material. Its dark limit is an H+/effective-heavy-ion OML balance:

`nH+ = xHII nH`

`nheavy = ne - nH+`

`Rion = [nH+ sqrt(me/mH) + nheavy sqrt(me/mC)] / ne`.

Both ion channels are singly charged. Carbon mass is only an effective
heavy-ion mass; the residual channel is not literally C+ in every phase.
Inputs are rejected if `xHII` lies outside `[0,1]` or if `xHII nH > ne`
beyond float32 roundoff.

The dimensionless dark potential `nu = e phi/(kT)` solves the appropriate
attractive/repulsive OML branch, giving

`Zdark = nu aref kT/e^2`.

For external shielded radiation, define the Ibáñez-Mejía et al. (2019)
charging parameter

`psi = Gext sqrt(T) / ne`

and the published illuminated 1000 Å silicate centroid

`ZIM19 = b + k psi^alpha`.

The practical closure is the smooth asymptotic blend

`w = exp(-psi/hZ)`

`<Zref> = w Zdark + (1-w) ZIM19`.

Thus zero external FUV returns the dark OML solution and strong illumination
approaches the published IM19 fit. This blend is empirical: it is not a
published DS87/WD01 current-balance solution, and it does not insert an
unresolved CR-induced FUV floor.

IM19 Table 1 calibrates silicate fits only through 0.1 micron. Applying the
0.1-micron fit with constant-potential scaling over the production
0.0575--0.92 micron span, especially above 0.1 micron, is an explicit
extrapolation. It is a provisional HD23 astrodust proxy, not published HD23
charging accuracy.

## Particle quantities and Gaussian-cgs code units

For `aref = 0.1 micron`,

`phi = e <Zref>/aref`

`<Z(a)> = (a/aref) <Zref>`

`q/m = 3 phi/(4 pi rho_gr a^2)`.

The last expression is physical Gaussian-cgs `q/m` in esu/g. The Lorentz
kernel does not consume it directly. Since

`dv_phys/dt_phys = (q/mc) v_phys cross B_phys`

and RAMSES uses

`B_phys = v_unit sqrt(4 pi rho_unit) B_code`,

the dimensionless code coefficient is

`charge_parameter = (q/m) L_unit sqrt(4 pi rho_unit)/c`.

The regression `Z(1 micron)=100`, `rho_gr=2 g cm^-3`,
`L_unit=3.0857e18 cm`, and
`rho_unit=1.50492957435e-20 g cm^-3` gives
`charge_parameter=4851.240862014687` at 0.23 micron.

## Charge distribution and continuous grain sizes

The normalized integer distribution is used only offline. The float32 runtime
table stores `<Z^2>` and `<Z^2 ln|Z|>` at twelve logarithmic radius knots.
Stage 6/7 particle radii are continuous, so runtime lookup brackets the actual
particle radius and interpolates the positive moments in log radius; the
twelve values are not particle families.

For the YLD04 Coulomb logarithm,

`ln A = ln[3(kT)^(3/2)/(2 e^3 sqrt(pi ni))]`

and

`sum f_Z Z^2 ln(A/|Z|) = <Z^2> ln A - <Z^2 ln|Z|>`.

This remains nonzero at `<Z>=0` when the distribution variance is nonzero.
There is no charge floor and no persistent per-particle charge.

The 257-point transformed axis covers
`-512 <= <Zref> <= 8192`; the audited closure envelope reaches
`-418.6 <= <Zref> <= 5794.1`. Out-of-range charge or radius lookups are
rejected, not clamped. At continuous off-knot radii, the float32 lookup agrees
with direct float64 values to 1.31% in `<Z^2>` and 1.26% in the physical
Coulomb moment. The auxiliary `<Z^2 ln|Z|>` scaled error reaches 5.88% near
integer-charge structure; the combined drag moment is the acceptance
quantity.

## YLD04 Table 1 anchors

The validator carries the exact physical entries from YLD04 Table 1:

| Phase | T (K) | nH (cm^-3) | ne (cm^-3) | GUV | B (microgauss) |
|---|---:|---:|---:|---:|---:|
| CNM | 100 | 30 | 0.03 | 1 | 6 |
| WNM | 6000 | 0.3 | 0.03 | 1 | 5.8 |
| WIM | 8000 | 0.1 | 0.0991 | 1 | 3.35 |
| MC | 25 | 300 | 0.03 | 0.1 | 11 |
| DC1 | 10 | 1e4 | 0.01 | 0.01 | 80 |
| DC2 | 10 | 1e4 | 0.001 | 0.001 | 80 |

YLD04 does not provide `xHII` in that table. The audit assigns the WNM/WIM
electrons to H+ and neutral/cloud electrons to the effective heavy-ion proxy;
this extra composition assumption is stated in the generated JSON.

## Charging-time audit

The audit uses the YLD04 definition

`tau_Z = sigma_Z^2 / sum f_Z Jtot(Z)`.

The fitted distribution and classical electron collection rate set the scale;
detailed balance supplies an equal probability-weighted reverse current, so
the proxy is `tau_Z = sigma_Z^2/(2 sum f_Z Je)`. It is compared against

`min(dt, 1/|Omega|, tau_drag,total)`,

where inverse Epstein and distribution-averaged subsonic Coulomb times add.
The Coulomb rate separately sums `nH+ sqrt(mH)` and
`nheavy sqrt(mC)`, so shielded gas with `xHII=0` but `ne>0` does not
artificially lose Coulomb drag.

The checked baseline has
`max[tau_Z/min(...)]=0.011992`. HLS12 independently argues that charging is
fast relative to Larmor dynamics for grains above about `2e-7 cm`; the
production minimum is `5.75e-6 cm`, so the literature supports equilibrium
charging throughout this larger-grain range.

Draine--Sutin polarization corrections can matter at low reduced
temperature and are omitted here, as are a full WD01 current balance and
supersonic Coulomb roll-off. The timescale result is an order-of-magnitude
separation audit. It must be repeated using actual Stage 8 `T`, `ne`,
`xHII`, shielded radiation, `B`, and `dt` after the dependency merge.

Generate and validate:

```bash
python dust_charge_equilibrium.py
python validate_dust_charge_equilibrium.py
```

For post-merge states, use
`python dust_charge_equilibrium.py --environment-json states.json`.

## Literature

- Draine & Sutin (1987), ApJ 320, 803
- Weingartner & Draine (2001), ApJS 134, 263
- Lazarian & Yan (2002), ApJ 566, L105
- Yan, Lazarian & Draine (2004), ApJ 616, 895
- Hoang, Lazarian & Schlickeiser (2012), ApJ 745, 164
- Ibáñez-Mejía et al. (2019), MNRAS 485, 1220
