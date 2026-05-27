# Method

This is a **continuous-resolution diffusion model** because one sample is a set of Gaussian primitive parameters, and the renderer maps the same parameter set to arbitrary `(H,W)`.

Diffusion is performed in Gaussian parameter tensor space `B x N x D` (not pixel space):
- `D=8`: `[cx_raw, cy_raw, sx_raw, sy_raw, alpha_raw, r_raw, g_raw, b_raw]`.
- Constrained params: `center=sigmoid`, `scale` mapped to `[scale_min,scale_max]`, `opacity=sigmoid`, `color=tanh`.

Renderer:
`G_i(u)=exp(-0.5*(((x-cx_i)^2/sx_i^2)+((y-cy_i)^2/sy_i^2)))`
`I(u)=sum_i alpha_i*G_i(u)*c_i / (sum_i alpha_i*G_i(u)+eps)`.

Two-stage training:
1) image -> Gaussian parameters: train encoder + renderer by reconstruction.
2) diffusion over normalized Gaussian parameters from encoder outputs.

Inverse problem:
`y = A(R(z)) + eta`, posterior `p(z|y) ∝ p(y|z)p(z)`.
We implement a stable MAP-like approximation:
`min_z ||A(R(z))-y||^2 + lambda_prior*E_prior(z)` where prior energy is a simple latent norm proxy.

Need renderer-only baseline to show gains are from learned prior, not just renderer fitting.

Limitations: fixed `N`, ordered primitives approximation, encoder quality bottleneck, Gaussian limits for very fine textures.
