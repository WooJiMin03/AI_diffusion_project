# MNIST DDPM with Classifier-Free Guidance — Report

All figures referenced below are in `outputs/figures/` and are also embedded inline in
`cfg_completed.ipynb`. Every number in this report comes from the executed notebook or from
the auxiliary measurement script described in §5.2; nothing is estimated or carried over from
expectation.

## 1. Objective

Implement a DDPM on MNIST in the epsilon-estimation formulation and compare

- unconditional generation, and
- conditional generation with classifier-free guidance (CFG), where the guidance scale
  `gamma` controls how strongly the digit label is enforced,

then run a gamma ablation and two exploration experiments (nearby noisy states, hole filling).

## 2. Implementation

Four functions were completed in `cfg_completed.ipynb`. The provided scheduler, U-Net, EMA and
visualization helpers were left unchanged.

### `ddpm_loss_epsilon(sched, net, x0, y)`

1. `bsz = x0.shape[0]` (no hard-coded batch size)
2. `t = sched.sample_timesteps(bsz)` — uniform over `{0..T-1}`, dtype `long`, shape `(B,)`
3. `eps = torch.randn_like(x0)`
4. `x_t = sched.q_sample(x0, t, eps)` — the provided forward diffusion
   `x_t = sqrt(alpha_bar_t) x0 + sqrt(1-alpha_bar_t) eps`
5. `eps_hat = net(x_t, t, y)`
6. return `F.mse_loss(eps_hat, eps)` — a scalar tensor; no `.item()`, no backward inside

### `ddpm_loss_cfg(sched, net, x0, y, null_id)`

Same steps 1–4, then **one** `(t, eps, x_t)` triple is shared by both branches:

```python
eps_hat_cond   = net(x_t, t, y)
y_null         = torch.full_like(y, null_id)
eps_hat_uncond = net(x_t, t, y_null)
loss = F.mse_loss(eps_hat_cond, eps) + F.mse_loss(eps_hat_uncond, eps)
```

Sharing the noisy sample matters: the two terms must be two views of the *same* denoising
problem (with and without the label). Only then is `eps_c - eps_u` at sampling time the
contribution of the label rather than the difference between two unrelated noisy images.

The notebook contains a test for this: with a fixed seed, `ddpm_loss_cfg` returns
2.0017 while `ddpm_loss_epsilon` returns 1.0009 on the same input — the conditional term of
the CFG loss reproduces the epsilon loss exactly on the same `(t, eps, x_t)`, and the total is
the sum of two comparable MSEs.

### `sample_unconditional(n, steps)`

Ancestral DDPM sampling from `x_T ~ N(0, I)` using the null label at every step:

```python
eps_hat = net(x, t, y_null)
mu, var = sched.posterior_mean_variance(x, eps_hat, t)
x = mu + sqrt(var) * z   # ti > 0
x = mu                   # ti == 0
```

returning `x.clamp(-1, 1)`.

### `sample_conditional(label, gamma, n, steps)`

Identical loop, but two forward passes on the same `x` per step and the CFG combination
specified by the notebook:

```python
eps_hat = (1.0 + gamma) * eps_c - gamma * eps_u
```

### Meaning of gamma

Rewriting the weighting as

```
eps_hat = eps_u + (1 + gamma) * (eps_c - eps_u)
```

shows the update is an **extrapolation** from the unconditional prediction along the label
direction `eps_c - eps_u`:

- `gamma = -1` → `eps_hat = eps_u`: purely unconditional; the label is ignored entirely.
- `gamma = 0` → `eps_hat = eps_c`: plain conditional model, no guidance.
- `gamma > 0` → the label direction is amplified, trading diversity for class fidelity.

Because this is extrapolation rather than interpolation, large `gamma` pushes the trajectory
outside the region the model was trained on, which is the source of the saturation artifacts
quantified in §5.

### Deviations from the starter notebook (all minimal)

| Change | Reason |
| --- | --- |
| device selection extended with an MPS branch | run on Apple Silicon (M5) |
| `num_workers=2` → `0` | see note below |
| training loop records epoch-average loss, wall time, and periodic checkpoints | reproducibility and crash recovery |
| `save_grid()` / `show_compare()` helpers added | write figures to `outputs/figures/`; the provided `show_grid()` is untouched and still used |
| conditional example title fixed | the starter called `label=2, gamma=-1` but titled the figure `label=5, gamma=1.0` |
| `denoise_from_xt()` helper added | the explorations need a reverse process starting from an intermediate `x_t` |
| `SMOKE` env flag | enables a fast 1-epoch end-to-end debug pass; defaults to the full configuration, and all reported results use the full configuration |

**Note on `num_workers`.** Worker processes died during development, but on re-examination that
was an artifact of our test harness rather than of the notebook — it reproduces only when the
loader is built inside an `exec`-ed namespace, not in a real Jupyter kernel, where
`num_workers=2` works. We kept `num_workers=0` because it costs nothing measurable (200 batches
in 0.25 s, far from the bottleneck) and removes a source of nondeterminism. The original
justification ("macOS worker crash") was wrong and is corrected here.

## 3. Training setup

| Item | Value |
| --- | --- |
| Device | `mps` (Apple M5, 10-core GPU) |
| Diffusion steps `T` | 200 (linear betas 1e-4 → 0.02) |
| Model | provided `SmallUNet`, `ch=64`, null class enabled (`null_id = 10`) |
| Loss | `ddpm_loss_cfg` (`USE_CFG_LOSS = True`) |
| Epochs | 30 |
| Batch size | 64 (`drop_last=True`, 937 iterations/epoch, 28,110 steps total) |
| Optimizer | AdamW, lr 1e-4 |
| EMA | decay 0.999; EMA weights copied into the model before all sampling |
| Seeds | global seed 0; each gamma / exploration call re-seeds immediately before sampling |
| Final epoch-average loss | **0.0841** |
| Total training time | **7954.6 s (132.6 min)** |

Epoch-average loss: 0.1871 (e1) → 0.1068 (e2) → 0.0989 (e3) → 0.0943 (e5) → 0.0888 (e10) →
0.0869 (e15) → 0.0854 (e20) → 0.0848 (e25) → 0.0841 (e30).

*Observed:* the curve drops steeply for the first three epochs and then flattens, ending at
0.0841 without approaching zero.

*Interpretation (not separately tested):* because `t` is drawn uniformly per sample, each batch
mixes trivial (small `t`) and near-impossible (large `t`) denoising problems, so the loss would
not be expected to approach zero even for a well-fit model. Consistent with this, the visible
improvement after roughly epoch 5 appears in sample quality — we verified at epoch 10 that
conditional samples were already clean digits — rather than in the loss value. We did not run
the per-`t` loss breakdown that would test this directly.

### A property of this schedule that explains much of §6 and §7

With `T = 200` and `beta_end = 0.02`, the forward process never fully destroys the image:

| t | `alpha_bar_t` | signal `sqrt(alpha_bar_t)` | noise `sqrt(1-alpha_bar_t)` |
| --- | --- | --- | --- |
| 0 | 0.9999 | 0.9999 | 0.0100 |
| 20 | 0.9771 | 0.9885 | 0.1512 |
| 80 | 0.7168 | 0.8466 | 0.5322 |
| 150 | 0.3155 | 0.5617 | 0.8273 |
| 199 | 0.1322 | **0.3636** | 0.9316 |

Even at the last timestep the original image still contributes ~36 % of the signal amplitude.
This schedule is much shorter than the `T = 1000` typically used for DDPM, so *every*
timestep retains more of the source than one might expect. Both explorations below are
dominated by this fact.

## 4. Unconditional generation

`outputs/figures/unconditional.png` (64 samples, seed 0).

The grid contains clearly formed, well-separated digits with smooth continuous strokes and
clean black backgrounds, in a variety of stroke weights and slants. Outputs are finite and
within `[-1, 1]`, as asserted in the notebook.

Scoring the same 64 samples with the auxiliary classifier of §5.2 gives:

```
0:14  1:1  2:6  3:14  4:4  5:4  6:3  7:9  8:8  9:1
```

**All ten classes are represented in this batch**, mean classifier confidence is 0.885, and no
sample falls below 0.5 confidence. Note that confidence is the classifier's certainty, not a
measure of image quality or correctness — it says the shape resembles a digit class, nothing
more. The distribution is uneven: "0" and "3" together account for 28 of 64 samples while "1"
and "9" appear once each.

We do not read this as an established class imbalance of the model. At n = 64 the counts are
noisy, and a single batch cannot estimate the model's true marginal over digits. Visual
inspection suggests that thin single-stroke shapes may be harder — samples nearest to a "1"
tend to acquire extra curvature — but this sample is too small to establish that. The
distribution should be treated as indicative only.

## 5. Conditional generation and gamma ablation

`outputs/figures/gamma_label2_{-1,0,1,2,3,4,5}.png`, all with `label = 2`, 64 samples,
`torch.manual_seed(0)` called immediately before each `sample_conditional()` call so that every
gamma starts from the same initial noise.

### 5.1 Direct visual observations

- **γ = −1** — identical to the unconditional grid. This was verified numerically, not by eye:
  regenerating both with the same seed and comparing tensors gives `torch.equal(...) == True`,
  a maximum absolute difference of `0.000e+00`, and 100.0000 % identical pixels. At `γ = −1`
  the weighting collapses to `eps_hat = eps_u` exactly (`0·eps_c + 1·eps_u` in IEEE
  arithmetic), and both samplers consume the RNG identically, so bitwise equality is the
  correct expectation — and it holds. This is the strongest available confirmation that the
  CFG weighting is implemented correctly.
- **γ = 0** — a mixture. Some clear 2s, but also 5s, 0s, 9s, 3s and shapes that are not digits
  at all. The plain conditional prediction alone is a weak steering signal.
- **γ = 1** — most samples are 2s; a handful of non-2s remain.
- **γ = 2** — nearly all samples are 2s.
- **γ = 3** — almost uniformly 2s, strokes clean and well formed. This is the best-looking grid.
- **γ = 4, 5** — still overwhelmingly 2s, but strokes visibly thicken and some samples begin to
  blob, losing the thin inner curvature that makes a 2 look handwritten.

Comparing the same grid position across gammas shows the mechanism directly: a sample that is
a "0" at γ = −1 progressively deforms into a "2" as γ increases — the same initial noise is
being steered further along the label direction.

### 5.2 Quantitative measurement (auxiliary)

To avoid eyeballing 7 × 64 images, a small CNN classifier was trained on MNIST
(98.49 % test accuracy) and applied to each gamma's 64 samples. **This classifier is an
analysis tool only — it is not part of the assignment deliverable and does not touch the
diffusion model.** Script: `outputs/gamma_metrics.txt`.

| γ | fraction classified as "2" | mean confidence in "2" | diversity (mean pairwise L2) | saturated pixels |
| --- | --- | --- | --- | --- |
| −1 | 0.094 | 0.103 | 23.43 | 0.745 |
| 0 | 0.719 | 0.688 | 23.11 | 0.740 |
| 1 | 0.844 | 0.832 | 22.74 | 0.744 |
| 2 | 0.953 | 0.941 | 22.46 | 0.751 |
| 3 | **0.984** | 0.973 | 22.33 | 0.752 |
| 4 | **0.984** | 0.977 | 22.25 | 0.752 |
| 5 | 0.969 | 0.963 | 22.27 | 0.751 |

Reading of the table:

- **γ = −1 gives 0.094**, essentially the natural frequency of the digit 2 in MNIST (5958 of
  60000 training images, 0.099). The label is genuinely ignored, exactly as the algebra
  predicts and as the bitwise check above confirms.
- **Class fidelity rises steeply, peaks, and then declines.** The large gains are from
  −1 → 0 → 1 → 2; the maximum is at γ = 3–4 (0.984); at γ = 5 it falls back to 0.969.
  Guidance is not "more is better" — past a point, pushing further along `eps_c - eps_u`
  degrades the very class identity it is meant to enforce, because the trajectory is being
  extrapolated further outside the training distribution.
- **Diversity falls monotonically** through γ = 4 (23.43 → 22.25), with a negligible uptick at
  γ = 5 (22.27) that we do not read as meaningful. The cost is modest but consistently one-way.
- **Saturation** is lowest at γ = 0 (0.740), rises to a plateau of ≈ 0.751–0.752 for γ ≥ 2, and
  does not increase further. Note γ = −1 sits at 0.745, above γ = 0, so this metric is not a
  clean monotone function of gamma; the visual thickening of strokes at high gamma is more
  evident in the images than in this aggregate number, which is dominated by the large black
  background common to all samples.

### 5.3 Interpretation

Guidance strength trades **class fidelity against diversity and naturalness**, and the trade is
not monotone in gamma. Among the settings tested here — this model, this checkpoint, label "2",
64 samples per gamma, one classifier — γ = 2–3 gave the best observed trade-off: 95–98 % of
samples carry the target class while strokes still look like handwriting. Below that the label
is under-enforced (γ = 0 reaches 0.719).

Two caveats on the upper end. First, **γ = 3 and γ = 4 are tied on measured fidelity** (both
0.984); we prefer γ = 3 only because diversity is marginally higher there (22.33 vs 22.25), and
because stroke quality looked better — the latter is a *visual* judgement, not a measurement.
Second, the drop at γ = 5 (0.969) comes from a single batch of 64 and we ran no significance
test, so it is best read as **diminishing returns and mild degradation in the tested batch**
rather than a demonstrated decline. What the data supports is the existence of a guidance
ceiling in this experiment; the useful range may shift with a different model, schedule, label
or diversity requirement.

## 6. Exploration 1 — nearby noisy states

**Question.** Generate a "7", noisify it to timestep `t`, perturb slightly, denoise back to
`t = 0`. Do we get a similar image? Does the answer depend on `t` or on
conditional/unconditional sampling?

**Setup.** 4 source 7s generated with CFG (`label = 7, gamma = 3`), `perturb_scale = 0.1`
added on top of `x_t`, `t ∈ {20, 80, 150}` (project-chosen; the assignment does not fix them),
reverse process via `denoise_from_xt()`, which reuses exactly the same ancestral update and CFG
weighting as the samplers. Figures: `outputs/figures/nearby_state_t{20,80,150}.png`, each row
showing `source | noisy | perturbed | unconditional result | conditional result`.

**Mean absolute error against the source (n = 4 per condition):**

| t | unconditional | conditional (label 7) |
| --- | --- | --- |
| 20 | 0.0210 | 0.0212 |
| 80 | 0.0499 | 0.0474 |
| 150 | 0.1000 | 0.0911 |

These are the values printed by the executed notebook. The extension in
[§6.1](#pushing-t-to-the-end-of-the-schedule) repeats t = 150 at n = 16 and reports slightly
different numbers (0.0973 / 0.0889) from that larger sample; the two are kept separate
throughout and never combined.

**Observations.**

- **t = 20** — near-perfect reconstruction. Not just the digit class but the specific
  handwriting is preserved: slant, stroke thickness, the exact shape of the upper bar. MAE
  ≈ 0.02. Conditional and unconditional are visually indistinguishable.
- **t = 80** — still recognisably the *same* 7 in every case. Strokes shift slightly and tend
  to get marginally cleaner and more canonical, but identity is retained. MAE roughly doubles
  to ≈ 0.048.
- **t = 150** — **still 7s, and still recognisably the same 7s.** This contradicts the naive
  expectation that a high timestep would erase instance identity. The crossed European-style 7
  in row 4 survives at t = 150 under *both* unconditional and conditional denoising, which is
  a strong result: the horizontal bar is an instance-level idiosyncrasy, not a class feature,
  and no label could have restored it. MAE ≈ 0.09–0.10 — five times the t = 20 value but still
  small in absolute terms.

**Why.** The answer is the schedule property in §3: at t = 150 the source still contributes
`sqrt(alpha_bar_150) = 0.56` of the signal amplitude. The noise has not destroyed the image; it
has blurred it. The reverse process is therefore anchored by real remaining structure rather
than free to invent, so the trajectory returns to the neighbourhood it started in.

A note on wording: the assignment calls this "sampling nearby latent codes", but what we
perturb is not a learned latent representation — it is the noisy image state `x_t`, which lives
in the same 28×28 pixel space as the image. The accurate statement of what we observed is
therefore narrower than "the latent space is smooth": **for the tested states, small
perturbations of `x_t` produced small changes in `x_0` after reverse diffusion, i.e. the
reverse process was locally stable around those states.** This is specific to the tested
schedule, perturbation scale (0.1) and timesteps. Under a longer schedule that destroys more
source information — `T = 1000`, where `alpha_bar_T ≈ 0` — we would expect weaker instance
retention, but that was not tested here.

**Conditional vs unconditional.** The difference is small and only becomes measurable at
t = 150, where conditioning gives a slightly lower MAE (0.0911 vs 0.1000, n = 4). This is
consistent with what a label can and cannot do: it constrains *which digit*, not *which
instance*. Since
the class was never in danger at these noise levels — the source structure already determined
it — the label had little work to do. Its small advantage at t = 150 is the point at which
class ambiguity first starts to appear.

### Pushing t to the end of the schedule

The assignment asks explicitly whether the answer changes with `t`. Our chosen grid
`{20, 80, 150}` turned out to stop just short of where it does, so we extended the same
procedure to **t = 190 and t = 199** (the last timestep the schedule defines), raised the
sample count to **n = 16**, and added the measurement that answers the question directly —
*is the result still a 7?* — using the classifier of §5.2. All 16 sources classify as 7.

| t | signal `sqrt(alpha_bar_t)` | MAE uncond / cond | still classified "7" uncond / cond | label gain |
| --- | --- | --- | --- | --- |
| 80 | 0.847 | 0.0501 / 0.0461 | 1.000 / 1.000 | +0.000 |
| 150 | 0.562 | 0.0973 / 0.0889 | 0.938 / 1.000 | +0.062 |
| 190 | 0.397 | 0.1511 / 0.1408 | 0.812 / 1.000 | **+0.188** |
| 199 | 0.364 | 0.1702 / 0.1386 | 0.750 / 1.000 | **+0.250** |

This resolves the question cleanly, and in the direction the earlier grid was too coarse to
show:

- **Unconditional denoising does eventually lose the class.** One of 16 drifts to a 0 at
  t = 150; by t = 190 three have become a 2, a 3 and an 8; by t = 199 four have drifted. The
  identity that survived so robustly at t ≤ 150 is not indestructible — the earlier grid simply
  never reached noise levels high enough to threaten it.
- **Conditional denoising retained the target class in all 16 samples at every timestep
  tested.** This is the job a label is able to do — constrain class — and within this sample it
  did so without exception. At n = 16 that is consistent with a strong effect but does not
  establish a 100 % rate.
- **The label's value grows monotonically with `t`** (+0.000 → +0.062 → +0.188 → +0.250),
  and the MAE gap widens in step (largest at t = 199: 0.1702 vs 0.1386).

So the answer to "does it change with `t`?" is yes, but the crossover sits above t = 150. Below
it the residual signal fixes the class by itself and the label adds little; above it the signal
degrades enough that conditioning substantially reduces class drift. The label does not become
the *only* thing holding the class — unconditional sampling still keeps it in 75 % of samples at
t = 199 — but its contribution grows as residual source information weakens. That 75 % is itself
a consequence of this schedule retaining 36 % signal amplitude at the final timestep; under
`T = 1000` we would expect it to fall further, though we did not test that.

As a consistency check, the n = 16 values at t = 150 (0.0973 / 0.0889) closely reproduce the
n = 4 values reported above (0.1000 / 0.0911).

*(t = 190 and t = 199 are an extension beyond the three timesteps used for the main figures;
the figures in `outputs/figures/` cover t ∈ {20, 80, 150}.)*

## 7. Exploration 2 — hole filling

**Question.** Generate an "8", mask part of it, noisify to `t`, denoise back. Can the model fill
the hole? Does it depend on `t` or on conditional/unconditional sampling?

**Setup.** 4 source 8s generated with CFG (`label = 8, gamma = 3`); the **top 10 of 28 rows**
are set to `-1.0` (background in `[-1,1]` space); the masked image is noisified with
`sched.q_sample()` and denoised from `t ∈ {20, 80, 150}`. Figures:
`outputs/figures/hole_filling_t{20,80,150}.png`, rows showing
`source | masked | noisy | unconditional result | conditional result`.

Note this is deliberately **not** an inpainting algorithm: known pixels are *not* re-imposed at
each reverse step. We only mask, noisify and denoise, and observe the model's tendency — the
assignment explicitly scopes the experiment this way.

**Ink fraction inside the masked band** — a *weak proxy* for how much ink returned to the erased
region, not a hole-filling score. It counts bright pixels without regard to shape, so scattered
stray strokes score well and a loop straddling the rows 9–13 boundary scores poorly; it measures
neither structural similarity to the original nor loop closure. (Fraction of pixels above 0 in
the top 10 rows, n = 4 per condition; the unmasked source measures 0.179.)

| t | unconditional | conditional (label 8) |
| --- | --- | --- |
| 20 | 0.000 | 0.000 |
| 80 | 0.010 | 0.012 |
| 150 | 0.076 | 0.062 |

**Observations.**

- **t = 20 — no filling at all.** Both variants reproduce the truncated digit faithfully and
  simply clean up the noise. The model treats "8 with its top cut off" as a perfectly valid
  image to denoise. Ink recovered: exactly 0.000.
- **t = 80 — essentially still no filling.** A few isolated specks appear near the top edge
  (0.010 / 0.012 vs 0.179 in the source), but no structure. The digits remain truncated.
- **t = 150 — partial recovery in both variants.** This is where something actually happens.
  Ink returns above the mask line under both conditional and unconditional denoising, and the
  **preserved lower half also changes**: the model is not filling a hole, it is redrawing the
  whole digit. Visually the conditional results look more like closed 8s — rows 2 and 4 regrow
  an upper loop where the unconditional results end as open Y/X figures — but see the
  measurement below before treating that as established.

### Why neither metric settles the conditional/unconditional question

The ink metric ranks unconditional (0.076) above conditional (0.062) at t = 150, which is the
opposite of the visual impression. We probed this rather than assuming, measuring ink row by
row and asking the classifier of §5.2 what each result is:

| rows | source 8 | masked | uncond | cond (8) |
| --- | --- | --- | --- | --- |
| 0–9 (counted by the metric) | 0.179 | 0.000 | 0.076 | 0.062 |
| 10–13 | 0.335 | 0.335 | 0.297 | **0.344** |
| topmost row with ink, per sample | 6,4,6,5 | 10,10,10,10 | 4,6,7,5 | 9,5,8,5 |

Two things follow, and the first corrects a plausible-sounding but wrong explanation:

1. **The restored digit is not displaced below the mask line.** Both variants put ink back into
   rows 4–9; the conditional topmost rows are 9, 5, 8, 5. What actually differs is the
   *distribution*: unconditional scatters thin ink high in the band (rows 4–8) in the form of
   open curls and stray strokes, while conditional builds a denser loop straddling rows 9–13 —
   its densest band (0.344) sits mostly outside the counted region. The metric rewards "any
   bright pixel in rows 0–9", which the stray strokes satisfy and the straddling loop only
   partly does. Recovery is also faint in absolute terms: the source reaches 0.518 at row 7,
   both reconstructions only ≈ 0.107.
2. **Class identity cannot measure hole-filling at all.** The classifier labels every set as
   "8" — including the *masked* input, at P(8) = 0.99–1.00. A truncated 8 still reads as an 8
   from its lower loop alone, so "is it still an 8?" is answered yes whether or not the hole
   was filled. The obvious better metric is therefore not available either.

Consequently we **do not claim a conditional advantage at t = 150**. The visual impression
favours it, both quantitative measures fail to support it, and n = 4 is too small to settle it.
What all three lines of evidence do agree on is the main finding: at t = 20 and t = 80
essentially nothing is recovered.

**Why the model does not fill the hole.** Nothing in the reverse process is told that the top
rows are missing. A flat black region is a perfectly ordinary input for MNIST — most of every
image is black. Combined with the §3 schedule property, at low `t` the masked image is still
~99 % intact as far as the model is concerned, so the denoiser's job is to remove noise, not to
question content. The mask only becomes "visible" as a problem once enough noise has been added
that the model can no longer trust the local evidence — and by then it has also lost the parts
we wanted to keep.

**Conditional vs unconditional.** At t = 20 and t = 80 the two are indistinguishable — neither
recovers anything, so there is nothing for the label to influence. At t = 150, where the model
is genuinely reconstructing rather than denoising, the label is in principle the one thing that
could supply what the mask destroyed ("this must be an 8"). Our data does not demonstrate that
it does: the images look better under conditioning, the ink metric ranks unconditional higher,
and the classifier cannot distinguish the two (or even distinguish either from the masked
input). We record this as **unresolved at n = 4** rather than reading the ambiguity in favour of
the more intuitive answer. What is clear either way is that at t = 150 this is regeneration,
not restoration — the intact lower half changes too.

**Conclusion for this experiment.** The tested mask–noise–denoise procedure did not perform
reliable inpainting. This is a statement about the *procedure*, not about diffusion models in
general: we ran an ordinary reverse process without re-imposing known pixels at any step, and at
low noise it preserved the hole while at high noise it regenerated the whole digit. The conflict
is direct — filling requires enough noise to erase the mask, and that same noise erases the
content we wanted to preserve. Dedicated diffusion inpainting resolves it by re-imposing
the known pixels at every reverse step, so the model is forced to keep the intact region while
only the masked region is generated. That algorithm is out of scope here, and the failure
observed above is precisely the reason it exists.

## 8. Conclusion

Every statement below is scoped to this model, checkpoint, schedule, seed, sample size and
evaluation setup.

1. **Gamma controls condition strength through extrapolation**, not interpolation:
   `eps_hat = eps_u + (1+γ)(eps_c - eps_u)`. Measured class fidelity rises
   0.094 → 0.719 → 0.844 → 0.953 → 0.984 for γ = −1…3 (64 samples per gamma, label "2",
   auxiliary classifier).
2. **Guidance showed a ceiling in this experiment.** Fidelity saturated at γ = 3–4 (0.984) and
   was lower at γ = 5 (0.969), while diversity decreased monotonically through γ = 4
   (23.43 → 22.25). Read as diminishing returns and mild degradation in the tested batch rather
   than a demonstrated decline — one batch, no significance test. γ = 2–3 gave the best observed
   trade-off here.
3. **γ = −1 reproduces unconditional sampling exactly** — verified numerically as bitwise
   equality (max abs difference 0.000e+00 over 64 samples), with a target-class rate of 0.094
   matching MNIST's natural base rate of 0.099. This confirms the CFG implementation.
4. **Timestep governs how much of the source survives**, and with `T = 200` far more survives
   than one would assume: even `t = 199` retains 36 % signal amplitude. Nearby noisy states
   therefore return to the *same instance* — same handwriting, including instance-specific
   quirks like a crossed 7 — at every timestep up to t = 150. Extending to t = 190 and t = 199
   locates the limit: unconditional denoising then loses the class in 19–25 % of samples, while
   conditional denoising holds it at 100 %.
5. **Labels constrain class, not instance — and only matter once the class is at risk.**
   Conditioning barely helped in Exploration 1 below t = 150 because the residual signal already
   fixed the class; at t = 190–199, where unconditional sampling starts drifting to other
   digits, conditioning substantially reduced class drift (16/16 retained vs 75–81 %,
   n = 16). It never recovers
   *instance* detail: the crossed 7 survived on residual signal alone, and no label could have
   supplied it. In Exploration 2 the label should matter for the same reason — masking destroys
   class-recoverable information — but our measurements could not establish that it does (§7),
   so we leave that one open rather than assert it.
6. **The tested mask–noise–denoise procedure did not fill holes.** It either preserved the
   hole (low `t`) or regenerated the whole digit (high `t`); in no tested setting did it restore
   the missing part while keeping the rest. This is a limitation of this procedure, not a training
   deficiency, and it motivates dedicated inpainting methods.

### Honest limitations

- The timesteps `{20, 80, 150}` and `perturb_scale = 0.1` are project choices; the assignment
  fixes neither. Conclusions about "high `t`" are relative to `T = 200` and would change under a
  longer schedule.
- The explorations use 4 samples per condition. They are illustrative, not statistically
  powered; the gamma ablation (64 samples per gamma, classifier-scored) is the quantitative
  part of this report.
- The CNN classifier in §5.2 is an analysis tool only — it is not part of the diffusion model or
  of the assignment deliverable. It is itself imperfect (98.49 % test accuracy) and was trained
  on real MNIST, so generated and malformed images are out of its training distribution and its
  judgements on them carry additional error. Its **confidence is not accuracy**: a mean
  confidence of 0.885 in §4 means the shapes resemble digit classes, not that 88.5 % of the
  images are correct or well formed. Its predicted-class distribution is likewise not a
  ground-truth distribution. Section 7 shows the sharpest limit: class prediction cannot measure
  hole-filling at all, since the classifier labels even the masked input "8" at P ≈ 0.99.
- Exploration 1's finding that identity survives at t = 150 was *not* what we expected before
  running it. It is reported as observed, and the schedule analysis in §3 is our explanation
  after the fact rather than a prediction confirmed.
- The ink-fraction metric in §7 is a weak proxy and disagrees with the images at t = 150. We
  probed the disagreement (row-wise ink profile, classifier verdicts) instead of resolving it by
  assertion, and the probe refuted our first explanation for it — restored digits are *not*
  displaced below the mask line. The revised account in §7 is what the measurement supports;
  the conditional/unconditional comparison at t = 150 is left unresolved.
- Two separate determinism claims appear in this report and do not contradict each other.
  (a) *Sampling* is bitwise deterministic given a fixed checkpoint and seed — this is what the
  `gamma = -1` check in §5.1 establishes, and it verifies the CFG implementation.
  (b) *Training* is not bitwise reproducible across runs, as described next. (a) concerns one
  forward process on fixed weights; (b) concerns how those weights were produced.
- Training is reproducible at the level of the loss curve but not bitwise. Two full runs with
  identical seeds produced identical epoch-average losses to four decimals (0.1871 → 0.0841)
  yet checkpoints differing in their SHA-256, because MPS kernels do not guarantee
  deterministic reduction order. Consequently all numbers in this report were re-measured
  against the exact checkpoint (`outputs/checkpoints/ddpm_cfg_final.pt`) that produced the
  figures shown; figures and numbers come from one and the same run.
- The class distribution in §4 comes from a single batch of 64 samples and should be read as
  indicative, not as a calibrated estimate of the model's marginal over digits.
