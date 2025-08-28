# Square root & bounded generator

We use spectral calculus for finite-dimensional PSD matrices:
- `sqrt_psd_matrix(A)` implements the functional calculus for `sqrt{A}`.
- `bounded_generator(H, α)` implements `H (I + H^2/α^2)^(-1/2)`.

**Tests validate:**
1. `sqrt(A)` is Hermitian PSD and squares back to `A` (up to numerical tolerance).
2. `H_α → H` strongly on vectors (`‖(H_α−H)x‖→0`).
3. Hölder-1/2 behavior near gap closures (numerical proxy), Lipschitz-like for gapped regime.
