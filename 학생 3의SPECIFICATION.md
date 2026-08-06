# DDPM & Classifier-Free Guidance (CFG) 시스템 명세서 (Specification)

이 문서는 MNIST 데이터셋 기반의 **DDPM(Denoising Diffusion Probabilistic Model)** 및 **Classifier-Free Guidance (CFG)** 시스템의 수학적 원리, 전체 시스템 아키텍처, 모듈별 구조, 손실 함수, 샘플링 알고리즘 및 탐구(Exploration) 모듈에 대한 최신 라이브러리/노트북([`ddpm_cfg.py`](file:///Users/jcchk/GET_Project_Diffusion/ddpm_cfg.py), [`cfg_starter.ipynb`](file:///Users/jcchk/GET_Project_Diffusion/cfg_starter.ipynb)) 기준 상세 기술 명세서입니다.

---

## 1. 프로젝트 개요 (Overview)

본 프로젝트는 MNIST 손글씨 숫자 데이터셋($28 \times 28$, 1채널)에 대해 무조건부(Unconditional) 및 조건부(Conditional) 이미지 생성과 잠재 공간 서치/인페인팅(Inpainting) 탐구를 수행하는 확산 모델(Diffusion Model) 시스템입니다.

### 주요 특징
* **기반 모델**: DDPM (Denoising Diffusion Probabilistic Models, Ho et al., 2020)
* **조건부 생성 방식**: Classifier-Free Guidance (CFG, Ho & Salimans, 2022)
* **백본 아키텍처**: Adaptive Group Normalization (AdaGN) 기반의 Small U-Net (Attention-free)
* **임베딩 방식**: Sinusoidal Time Embedding 및 Class Embedding (Null Token ID=10 지원)
* **최적화 기법**: EMA (Exponential Moving Average, decay=0.999), AdamW Optimizer (lr=1e-4)
* **탐구 모듈**: Nearby Latent Codes Perturbation & RePaint 방식 Hole Filling (Inpainting)

---

## 2. 수학적 배경 및 알고리즘 명세 (Mathematical Background)

### 2.1 Forward Process (정방향 확산 과정)
원본 데이터 $x_0 \sim q(x_0)$에 각 타임스텝 $t \in \{1, \dots, T\}$마다 가우시안 노이즈를 단계적으로 추가합니다.

$$q(x_t | x_0) = \mathcal{N}(x_t; \sqrt{\bar{\alpha}_t} x_0, (1 - \bar{\alpha}_t) \mathbf{I})$$

여기서:
* $\beta_t \in (0, 1)$ : 타임스텝 $t$에서의 노이즈 스케줄 (Linear Schedule: $\beta_1 = 10^{-4} \dots \beta_T = 0.02, T=200$)
* $\alpha_t = 1 - \beta_t$
* $\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$ (누적 곱, Cumulative Product)

Reparameterization Trick을 사용하여 임의의 타임스텝 $t$의 $x_t$를 한번에 직접 샘플링합니다:
$$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, \mathbf{I})$$

---

### 2.2 Reverse Process & Posterior Distribution (역방향 과정)
노이즈가 추가된 $x_t$로부터 원본 데이터 $x_{t-1}$을 복원하는 역방향 분포 $p_\theta(x_{t-1} | x_t)$는 다음과 같습니다:

$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$

이론적인 Posterior Distribution $q(x_{t-1} | x_t, x_0)$의 평균 $\tilde{\mu}_t$ 및 분산 $\tilde{\beta}_t$는 다음과 같이 정의됩니다:

$$\tilde{\beta}_t = \frac{1 - \bar{\alpha}_{t-1}}{1 - \bar{\alpha}_t} \beta_t$$

$$\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}} \beta_t}{1 - \bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1 - \bar{\alpha}_{t-1})}{1 - \bar{\alpha}_t} x_t$$

노이즈 예측 신경망 $\epsilon_\theta(x_t, t)$를 사용할 경우, 예측된 평균 $\mu_\theta(x_t, t)$는 다음과 같이 정형화됩니다:

$$\mu_\theta(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{1 - \alpha_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right)$$

---

### 2.3 손실 함수 (Loss Functions)

#### (1) Standard DDPM Epsilon Loss ($\mathcal{L}_{\text{simple}}$)
예측된 노이즈 $\epsilon_\theta(x_t, t, y)$와 실제 가우시안 노이즈 $\epsilon$ 간의 평균 제곱 오차(MSE)를 최소화합니다.

$$\mathcal{L}_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, y) \|^2 \right]$$

#### (2) Classifier-Free Guidance (CFG) Training Loss
CFG는 별도의 분류기(Classifier) 없이 조건부 생성과 무조건부 생성을 단일 네트워크로 학습합니다.
* 조건부 Forward Pass: $\epsilon_{\text{cond}} = \epsilon_\theta(x_t, t, y)$ (실제 클래스 레이블 $y \in \{0, \dots, 9\}$)
* 무조건부 Forward Pass: $\epsilon_{\text{uncond}} = \epsilon_\theta(x_t, t, y_{\text{null}})$ ($y_{\text{null}} = 10$ Null Token 레이블)
* 두 Forward Pass의 MSE 손실 합산:

$$\mathcal{L}_{\text{CFG}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t, y) \|^2 + \| \epsilon - \epsilon_\theta(x_t, t, y_{\text{null}}) \|^2 \right]$$

---

### 2.4 Classifier-Free Guidance Sampling (CFG 샘플링)
샘플링 도중 가이던스 스케일 파라미터 $\gamma$ (혹은 $w$)를 사용하여 조건부 방향으로의 기울기를 강조합니다:

$$\tilde{\epsilon}_\theta(x_t, t, y) = (1 + \gamma) \epsilon_\theta(x_t, t, y) - \gamma \epsilon_\theta(x_t, t, y_{\text{null}})$$

* $\gamma = -1.0$ : 조건 반전 (Negative Guidance, 역조건 생성)
* $\gamma = 0.0$ : 표준 조건부 예측 ($\epsilon_\theta(x_t, t, y)$ 기본 가이던스)
* $\gamma = 1.0 \sim 2.0$ : 균형 잡힌 조건 강조
* $\gamma = 3.0 \sim 4.0$ : Optimal Range (MNIST 획 명확도 최상)
* $\gamma \ge 5.0$ : 과도한 가이던스 (Over-saturation, 외곽선 왜곡 및 대비 과다)

---

## 3. 시스템 아키텍처 (System Architecture)

```
+-----------------------------------------------------------------------+
|                           MNIST Dataset                               |
|               Shape: (B, 1, 28, 28), Normalized to [-1, 1]           |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                        DiffusionSchedule (T=200)                      |
|  - Betas: linear(1e-4 -> 0.02)                                        |
|  - Alphas, Alphas_bar, sqrt_alphas_bar, sqrt_one_minus_alphas_bar     |
|  - q_sample(x0, t, eps) -> xt                                         |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                          SmallUNet Architecture                       |
|                                                                       |
|   Inputs: xt (B,1,28,28), t (B,), y (B,)                              |
|                                                                       |
|   [Condition Embedding]                                               |
|   t -> sinusoidal_time_embedding -> proj_t \                          |
|                                              +--> cond (B, 256)       |
|   y -> class_embed (11 classes) -> proj_y  /                          |
|                                                                       |
|   [U-Net Pipeline]                                                    |
|   Input (1ch) -> Conv2d -> h0 (64ch, 28x28)                           |
|      |                                                                |
|      v                                                                |
|   ResBlock1(64->64) -> Down1(64->128) -> ResBlock2(128->128)          |
|      |                                   |                            |
|      v                                   v                            |
|   Down2(128->256) -------------> ResBlock_Mid1, Mid2 (256, 7x7)       |
|                                          |                            |
|                                          v                            |
|   ResBlock_Up1(256+256->128) <--- Up1(256->128)                       |
|      |                                                                |
|      v                                                                |
|   ResBlock_Up2(128+64->64)   <--- Up2(128->64)                        |
|      |                                                                |
|      v                                                                |
|   Out Conv2d (64->1) ----------> eps_pred (B,1,28,28)                  |
+-----------------------------------------------------------------------+
                                   |
                                   v
+-----------------------------------------------------------------------+
|                    Sampling & Exploration Modules                     |
|  - Unconditional Sampling: sample_unconditional()                     |
|  - Conditional CFG Sampling: sample_conditional(label, gamma)        |
|  - Exploration 1: Nearby Latent Codes Perturbation & Denoising        |
|  - Exploration 2: Diffusion Hole Filling (RePaint Inpainting)         |
+-----------------------------------------------------------------------+
```

---

## 4. 컴포넌트 상세 명세 (Component Details)

### 4.1 `DiffusionSchedule`
* **역할**: 전방 확산 모의 및 역방향 Posterior 산출을 위한 사전 파라미터 계산 및 GPU/CPU 디바이스 등록.
* **주요 메서드**:
  * `sample_timesteps(bsz)`: $[0, T-1]$ 범위에서 균일 분포 Uniform(0, T-1)으로 $t$ 생성.
  * `q_sample(x0, t, eps)`: $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$ 계산.
  * `posterior_mean_variance(xt, eps_pred, t)`: 예측 노이즈 $\epsilon_{\text{pred}}$로부터 평균 $\mu$ 및 분산 $\tilde{\beta}_t$ 산출.

### 4.2 `sinusoidal_time_embedding(t, dim)`
* **역할**: Transformer 위치 임베딩 기법을 시간 스텝 $t$에 적용하여 연속 vector로 매핑.
* **수식**:
  $$\text{freqs}_i = \exp\left(-\frac{\ln(10000) \cdot i}{\text{dim}/2}\right)$$
  $$\text{emb} = [\sin(t \cdot \text{freqs}), \cos(t \cdot \text{freqs})]$$

### 4.3 `AdaGN (Adaptive GroupNorm)`
* **역할**: 시간 및 클래스 조건 결합 벡터 `cond`를 받아 특징 맵에 채널별 Scale ($s$) 및 Shift ($b$) 변환 인가.
* **연산**:
  $$\text{AdaGN}(x, \text{cond}) = \text{GroupNorm}(x) \cdot (1 + s) + b$$

### 4.4 `ResBlock`
* **역할**: 3x3 Conv2d, AdaGN, SiLU 활성화 함수, Dropout 및 Residual Skip Connection을 통합한 핵심 잔차 블록.

### 4.5 `SmallUNet`
* **채널 구성**: Input (1) $\to$ Base (64) $\to$ Level 1 (128) $\to$ Level 2 (256)
* **해상도 변화**: $28 \times 28 \to 14 \times 14 \to 7 \times 7 \to 14 \times 14 \to 28 \times 28$
* **클래스 임베딩**: Total 11개 (숫자 0~9 + Null Token 10)

### 4.6 `EMA (Exponential Moving Average)`
* **역할**: Shadow Parameters 관리로 학습 가중치 진동 완화 및 샘플링 시 가중치 평활화 (decay=0.999).

---

## 5. 핵심 구현 함수 명세 (Core Functions Specification)

### 5.1 `ddpm_loss_epsilon(sched, net, x0, y)`
* **입력**: `sched`, `net`, `x0` (원본 이미지 텐서), `y` (클래스 레이블)
* **수행 내용**: $t \sim \text{Uniform}(0, T-1)$ 샘플링 후 $x_t$ 생성 $\to$ `net(xt, t, y)`로 노이즈 예측 $\to$ $\text{MSE}(\hat{\epsilon}, \epsilon)$ 반환.

### 5.2 `ddpm_loss_cfg(sched, net, x0, y, null_id)`
* **입력**: `sched`, `net`, `x0`, `y`, `null_id` (기본값=10)
* **수행 내용**: 2-Pass Forward Pass (`eps_hat_cond` 및 `eps_hat_uncond`) 진행 후 손실 합산 반환.

### 5.3 `sample_unconditional(net, sched, n=64, steps=None)` / `sample_unconditional(n=64, steps=None)`
* **입력**: 생성 개수 `n`, 타임스텝 수 `steps`
* **수행 내용**: 모든 타임스텝에서 $y_{\text{null}} = 10$ 조건으로 역방향 디노이징 진행.

### 5.4 `sample_conditional(net, sched, label, gamma=3.0, n=64, steps=None)` / `sample_conditional(label, gamma=3.0, n=64, steps=None)`
* **입력**: 타겟 레이블 `label`, 가이던스 스케일 `gamma`, 생성 개수 `n`
* **수행 내용**: 매 역방향 스텝마다 $\hat{\epsilon} = (1+\gamma)\epsilon_c - \gamma \epsilon_u$ 가이던스를 인가하여 디노이징.

---

## 6. 탐구 모듈 명세 (Exploration Modules Specification)

### 6.1 `_denoise_loop(x_start, t_start, label, use_cfg=True, gamma=3.0)`
* **역할**: 중간 타임스텝 $t_{\text{start}}$에 위치한 임의의 노이즈 텐서 $x_{\text{start}}$로부터 $t=0$까지 디노이징을 수행하는 범용 헬퍼 루프.

### 6.2 `explore_nearby_latents(label=7, t_step=50, noise_std=0.1, gamma=3.0)`
* **역할**: 생성된 이미지 $x_0$에 타임스텝 $t_{\text{step}}$ 노이즈를 주입하고 가우시안 동요(Perturbation) $\mathcal{N}(0, \sigma^2 \mathbf{I})$를 추가한 후 역방향 복원을 통해 잠재 공간의 연속성과 매니폴드 보존 특성을 측정.

### 6.3 `explore_hole_filling_fixed_inpainting(label=8, crop_rows=10, t_step=100, gamma=3.0)`
* **역할**: 이미지 상단($10 \times 28$ 픽셀)을 마스킹한 후 RePaint 확산 복원 메커니즘($x_{t-1} = M \cdot x_{\text{known}, t-1} + (1-M) \cdot x_{\text{pred}, t-1}$)을 적용하여 조건부/무조건부 완성 능력 및 문맥 복원력 평가.

---

## 7. 결론 및 참고 문헌

1. **Ho, J., Jain, A., & Abbeel, P. (2020).** *Denoising Diffusion Probabilistic Models.* Advances in Neural Information Processing Systems (NeurIPS).
2. **Ho, J., & Salimans, T. (2022).** *Classifier-Free Diffusion Guidance.* arXiv preprint arXiv:2208.11970.
3. **Lugmayr, A., et al. (2022).** *RePaint: Inpainting using Denoising Diffusion Probabilistic Models.* IEEE/CVF CVPR.
