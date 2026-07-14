# Radio-immune model POC: parameter estimation notes

## 목적

이 문서는 현재 Python POC에서:

1. 어떤 수식을 사용했는지
2. 어떤 파라미터를 추정했는지
3. 어떤 방법으로 그 파라미터를 찾는지
4. 왜 일부 파라미터만 먼저 fit 했는지

를 정리한다.

구현 파일:

- `radiology_modeling/model.py`
- `radiology_modeling/fit.py`
- `radiology_modeling/poc.py`

---

## 1. 현재 사용 중인 모델

현재 POC는 `Cho_2023 Phys. Med. Biol. 68 165010`의 Figure 4 계열과 `ref/code/PaperFig04.m`의 discrete-time radio-immune model을 Python으로 옮긴 것이다.

시간 단위는 일(day)이고, 상태변수는 다음 4개다.

- `T_n`: viable tumor volume
- `D_n`: doomed cell volume
- `L_n`: active CTL / T-cell volume
- `A_n` 또는 코드상 `DC_n`: immune triggering cell density

관측량은 Figure 4(B)와 동일하게 총 종양 부피로 둔다.

\[
V_n = T_n + D_n
\]

### 1.1 Viable tumor update

\[
T_{n+1} = T_n S_{T,n} e^{\mu - Z_n}
\]

- `\mu`: tumor growth rate
- `S_{T,n}`: tumor surviving fraction after radiation
- `Z_n`: total immune effect

### 1.2 CTL / lymphocyte update

\[
L_{n+1} = (1-\lambda_L)S_{L,n}L_n + \rho T_{n+1} + \psi \epsilon_n A_{n+1}T_{n+1}
\]

- `\lambda_L`: lymphocyte decay
- `S_{L,n}`: lymphocyte surviving fraction after radiation
- `\rho`: basal T-cell production / infiltration from live tumor
- `\psi`: radiation-damaged tumor에 의해 유발되는 추가 T-cell activation strength
- `\epsilon_n`: radiation-triggered activation level

### 1.3 Primary immune effect

\[
Z_{p,n} = \frac{\omega \sum L_n}{1 + \kappa (\sum T_n)^{2/3}(\sum L_n)}
\]

- `\omega`: CTL 양이 실제 tumor killing pressure로 바뀌는 scale
- `\kappa`: tumor immune suppression strength

### 1.4 Secondary immune effect

현재 POC는 secondary immune effect를 끄고 시작했다.

\[
Z_{s,n+1} = Z_{s,n} + \gamma Z_{p,n+1} / r
\]

지금은 `\gamma = 0`이다.

### 1.5 Total immune effect

\[
Z_n = Z_{p,n} + Z_{s,n}
\]

### 1.6 Doomed cell update

\[
D_{n+1} = (1-\lambda_D)D_n + (1-S_{T,n})T_n + S_{T,n}T_n e^{\mu}(1-e^{-Z_n})
\]

- `\lambda_D`: doomed cell clearance / decay

---

## 2. 현재 어떤 파라미터를 찾고 있는가

현재 POC의 기본 추정 대상은 아래 4개다.

1. `psi`
2. `omega`
3. `kappa` (`k`)
4. `initial_volume` (`T_0`)

즉, 현재 POC는 아래 문제를 푼다.

> synthetic noisy total volume data `V_n = T_n + D_n`를 가장 잘 설명하는 `psi`, `omega`, `kappa`, `T_0`를 찾는다.

### 2.1 현재 고정한 파라미터

문헌값 또는 실험 설정값으로 고정한 것:

- `mu`
- `lambda_t`
- `lambda_dc`
- `lambda_ln`
- `rs_t_alpha`, `rs_t_beta`
- `rs_l_alpha`, `rs_l_beta`
- `gamma = 0`
- `r = 5`
- `treatment_day = 10`
- `dose_gy = 10`
- coverage는 `50%`, `100%`

### 2.2 왜 `rho`는 기본 fit에서 뺐는가

`rho`와 `psi`는 둘 다 CTL 증가에 기여한다.

\[
L_{n+1} = ... + \rho T_{n+1} + \psi \epsilon_n A_{n+1}T_{n+1}
\]

그런데 현재 관측치는 사실상 `T + D`뿐이다. 이 경우 `rho`와 `psi`가 서로 어느 정도 대체 가능해서, 데이터가 적으면 둘을 동시에 안정적으로 식별하기 어렵다.

그래서 기본 POC는:

- `rho` 고정
- `psi`, `omega`, `kappa`, `T_0`만 추정

으로 보수적으로 구성했다.

필요하면 `EXTENDED_FIT_SPECS`로 `rho`까지 포함할 수 있지만, 현재 단계에서는 식별성 문제가 더 크다.

---

## 3. synthetic noisy data는 어떻게 만들었는가

실데이터 적용 전에 inverse problem 파이프라인이 정상 동작하는지 보기 위해 synthetic data를 먼저 만들었다.

절차:

1. 논문 기반의 true parameter를 정한다.
2. coverage `50%`와 `100%`에 대해 모델을 forward simulate 한다.
3. treatment day 이후 특정 관측일들에서 `V_n = T_n + D_n`를 샘플링한다.
4. 각 관측값에 log-normal multiplicative noise를 넣는다.

코드상 기본 관측일:

- `0, 3, 6, ..., 30` days from treatment

노이즈 모델:

\[
V_n^{obs} = V_n^{clean} \exp(\eta_n), \quad \eta_n \sim \mathcal{N}(0, \sigma^2)
\]

기본값은 `noise_sigma = 0.08`이다.

이 설계는 종양 부피 데이터가 양수이고, 오차가 상대오차 성격을 갖는 경우에 더 자연스럽다.

---

## 4. 현재 방법론이 어떻게 파라미터를 찾는가

현재 방법은 **mechanistic simulator + global optimization + local optimization + weak prior regularization**이다.

### 4.1 핵심 아이디어

어떤 파라미터 후보 `theta`가 들어오면:

1. 모델을 forward simulate 한다.
2. 관측일에서 예측 종양 부피를 꺼낸다.
3. 관측값과 예측값의 차이를 residual로 만든다.
4. residual이 가장 작아지는 `theta`를 찾는다.

여기서 현재 `theta`는 기본적으로

\[
\theta = (\psi, \omega, \kappa, T_0)
\]

이다.

### 4.2 loss / residual 정의

현재 residual은 log-space에서 계산한다.

각 관측점에 대해:

\[
r_i(\theta) = \log(V_i^{pred}(\theta) + \epsilon) - \log(V_i^{obs} + \epsilon)
\]

여기서 `epsilon`은 수치 안정성을 위한 아주 작은 양수다.

로그 스케일을 쓰는 이유:

- 종양 부피는 항상 양수
- 큰 volume과 작은 volume의 상대오차를 더 균형 있게 본다
- multiplicative noise 가정과 잘 맞는다

전체 데이터는 `50% coverage`와 `100% coverage`를 동시에 사용한다.

즉, 한 coverage만 맞추는 것이 아니라 둘을 함께 설명하는 파라미터를 찾는다.

### 4.3 weak prior regularization

지금 POC에는 약한 정규화가 들어가 있다.

각 fit 파라미터에 대해 base value에서 너무 멀어지지 않도록 log-space prior residual을 추가한다.

\[
r^{prior}_j(\theta) = \frac{\log(\theta_j) - \log(\theta_{j,base})}{\sigma_{prior}}
\]

기본값:

- `prior_log_sigma = 0.35`

이 prior는 Bayesian posterior sampling은 아니고, optimization을 안정화하는 weak constraint다.

### 4.3.1 최적화 문제의 수식화

현재 POC가 실제로 푸는 최적화 문제는 아래와 같다.

관측 군집을 `g \in \{50\%, 100\%\}`라 하고, 각 군집의 관측 시점을 `t_i`라 하면 예측 총부피는

\[
V_{g,i}^{pred}(\theta) = T_g(t_i; \theta) + D_g(t_i; \theta)
\]

이다.

현재 추정 파라미터 벡터는

\[
\theta = (\psi, \omega, \kappa, T_0)
\]

이다.

데이터 residual은

\[
r_{g,i}(\theta) = \log\bigl(V_{g,i}^{pred}(\theta)+\epsilon\bigr) - \log\bigl(V_{g,i}^{obs}+\epsilon\bigr)
\]

이고, prior residual은

\[
r_j^{prior}(\theta) = \frac{\log(\theta_j) - \log(\theta_{j,base})}{\sigma_{prior}}
\]

이다.

따라서 최종적으로 최소화하는 목적함수는

\[
J(\theta) = \sum_g \sum_i r_{g,i}(\theta)^2 + \sum_j \left(r_j^{prior}(\theta)\right)^2
\]

이다.

즉 알고리즘은 결국

\[
\hat{\theta} = \arg\min_{\theta \in \Theta} J(\theta)
\]

를 푸는 것이다.

여기서 `\Theta`는 각 파라미터의 bound 집합이다. 예를 들면

\[
\Theta = [10, 800] \times [0.01, 0.3] \times [0.05, 2.5] \times [0.005, 0.08]
\]

처럼 둔다.

역할:

- 비현실적 파라미터 폭주 방지
- 데이터가 적을 때 식별성 약한 방향을 덜 흔들리게 함
- synthetic POC에서 recovery 안정화

### 4.4 1단계: global optimization

먼저 `scipy.optimize.differential_evolution`으로 전역 탐색을 한다.

이 단계의 목적:

- 초기값 민감도를 줄이기 위함
- nonconvex surface에서 나쁜 local minimum으로 바로 빠지는 것을 줄이기 위함

현재 설정 범위 예:

- `psi`: `10` ~ `800`
- `omega`: `0.01` ~ `0.3`
- `k`: `0.05` ~ `2.5`
- `initial_volume`: `0.005` ~ `0.08`

이 global stage는 “대충 어디가 좋은 영역인지”를 찾는 역할이다.

### 4.5 2단계: local optimization

global stage에서 찾은 해를 초기값으로 해서 `scipy.optimize.least_squares`를 수행한다.

이 단계의 목적:

- residual sum of squares를 더 정밀하게 줄임
- 최종 파라미터를 미세 조정

즉 전체 흐름은:

1. `differential_evolution`
2. `least_squares`

의 2-stage optimization이다.

### 4.6 왜 이 방법을 썼는가

이 문제는:

- 모델이 이미 정해져 있음
- 상태방정식이 non-linear 함
- immune suppression과 activation이 강하게 얽혀 있음
- radiation event 때문에 동역학이 불연속적으로 바뀜
- 적은 데이터로 숨은 상태를 직접 보지 못함

이라는 특징이 있다.

그래서 현재 단계에서는 PINN보다 이 방식이 더 직접적이고 안정적이다.

---

## 5. 현재 POC 결과 해석

현재 synthetic data에 대해서는 아래 파라미터 recovery가 잘 됐다.

- `psi`
- `omega`
- `kappa`
- `initial_volume`

이건 다음을 의미한다.

- forward simulator가 동작함
- inverse fitting pipeline이 동작함
- 최소한 선택한 파라미터 집합에 대해서는 synthetic recovery가 가능함

하지만 이것이 곧바로 실데이터에서 동일 수준으로 잘 된다는 뜻은 아니다.

실데이터에서는:

- measurement noise 구조가 다를 수 있음
- `rho` 등 추가 파라미터 식별성이 더 약할 수 있음
- model mismatch가 존재할 수 있음

따라서 실데이터 단계에서는 confidence interval 또는 posterior가 필요할 가능성이 높다.

### 5.1 각 파라미터가 곡선에 주는 방향성

현재 기본 fit 파라미터 `psi`, `omega`, `kappa`, `T_0`는 Figure 4(B) 스타일의 총 종양 부피 곡선 `V = T + D`에 서로 다른 방식으로 작용한다.

#### `T_0` (initial volume)

초기조건이므로 가장 직접적으로 곡선 전체의 높이를 좌우한다.

- `T_0` 증가: 전체 곡선이 거의 위로 평행 이동하듯 커진다.
- `T_0` 감소: 전체 곡선이 전반적으로 작아진다.

초기 시점 관측점에 가장 민감하고, 다른 파라미터와 달리 radiation 이전/직후 모두에서 단순한 스케일 효과를 준다.

#### `psi` (radiation-triggered CTL activation)

`psi`는

\[
L_{n+1} = ... + \psi \epsilon_n A_{n+1} T_{n+1}
\]

에 들어가므로 radiation으로 손상된 종양이 추가 면역반응을 얼마나 강하게 일으키는지를 정한다.

- `psi` 증가: 치료 직후 immune effect가 더 크게 솟고, 후반부 재성장이 더 늦어진다.
- `psi` 감소: partial irradiation의 장기 제어 이점이 약해지고, 재성장이 빨라진다.

특히 `50% coverage` 곡선의 후반부 tail과 `50% vs 100%` 차이를 만드는 데 민감하다.

#### `omega` (CTL to killing scale)

`omega`는

\[
Z_{p,n} = \frac{\omega \sum L_n}{1 + \kappa (\sum T_n)^{2/3}(\sum L_n)}
\]

에서 CTL 양이 실제 tumor killing effect로 얼마나 효율적으로 변환되는지를 정한다.

- `omega` 증가: 같은 CTL 양으로도 더 강한 killing effect가 생겨 종양 감소가 커진다.
- `omega` 감소: CTL이 있어도 실제 volume 감소 효과가 약하다.

`psi`가 CTL을 얼마나 많이 만들지 결정한다면, `omega`는 만들어진 CTL이 얼마나 세게 듣는지를 정한다.

#### `kappa` (immune suppression)

`kappa`는 primary immune effect의 분모에 들어간다.

- `kappa` 증가: 종양이 면역을 더 잘 억제해서 `Z_p`가 약해지고, 재성장이 빨라진다.
- `kappa` 감소: 같은 CTL 양이라도 suppression이 덜해서 더 오래 tumor control이 유지된다.

`kappa`는 특히 큰 종양 부피에서 더 강하게 작용하므로, 후반부 재성장 속도와 control failure 시점에 민감하다.

### 5.2 파라미터 간 상호작용

이 모델에서 중요한 것은 각 파라미터가 완전히 독립적으로 보이지 않는다는 점이다.

#### `psi` 와 `omega`

- `psi`는 CTL의 양을 늘린다.
- `omega`는 그 CTL의 효율을 높인다.

둘 다 결과적으로 `Z`를 키우므로 어느 정도 대체 가능하다. 그래서 데이터가 적으면 `psi`를 조금 낮추고 `omega`를 조금 높여도 비슷한 `V(t)`가 나올 수 있다.

#### `omega` 와 `kappa`

식

\[
Z_{p,n} = \frac{\omega L}{1 + \kappa T^{2/3} L}
\]

만 보면:

- `omega` 증가: 분자 증가
- `kappa` 증가: 분모 증가

라서 둘은 반대 방향으로 작용한다. 즉 `omega`를 키우는 효과를 `kappa`를 키워 상쇄할 수 있다.

#### `T_0` 와 `kappa`

초기 종양이 더 크면 suppression 항 `T^{2/3}`도 더 빨리 커지므로, 큰 `T_0`는 간접적으로 더 불리한 면역 환경을 만든다. 그래서 일부 경우 `T_0`와 `kappa`도 서로 얽힌다.

### 5.3 그래서 어떤 구간이 어떤 파라미터를 주로 식별하나

대략적으로는 다음처럼 볼 수 있다.

- 초기 절대 볼륨 수준: `T_0`
- 치료 직후 immune boost의 크기: `psi`
- immune boost가 실제 감소로 이어지는 효율: `omega`
- 후반부 재성장과 control 유지 실패 시점: `kappa`
- `50%`와 `100%` 곡선 차이: 주로 `psi`, 일부 `omega`, `kappa`

즉 한 시계열만 보는 것보다 `50%`와 `100%`를 동시에 fit하는 것이 식별성에 훨씬 유리하다.

---

## 6. PINN과의 관계

현재 문제에서 PINN은 1차 주력 방법으로 쓰지 않았다.

이유:

1. 이미 mechanistic equation이 명시적이다.
2. 현재 문제는 function learning보다 parameter inference에 가깝다.
3. discrete event와 sparse observation 때문에 PINN이 더 단순하지 않다.
4. `T + D`만 관측되는 구조적 식별성 문제를 PINN이 자동으로 해결해주지 않는다.

따라서 현재는:

- **주력**: mechanistic simulator + optimization
- **후속**: Bayesian calibration / uncertainty quantification
- **보조 가능성**: surrogate model, emulator, 또는 나중의 연속시간/공간확장용 PINN

이 더 적절하다.

---

## 7. 다음 단계

실데이터 적용 시 권장 순서:

1. 실제 관측 시계열 로더 추가
2. synthetic pipeline과 동일한 loss로 실데이터 fit
3. `rho` 포함 여부를 profile likelihood 또는 posterior로 점검
4. 필요시 `gamma` 포함
5. 최종적으로는 Bayesian calibration(예: `emcee`, ABC-SMC 등)으로 uncertainty 추정

---

## 8. 실행 명령

POC 실행:

```bash
uv run python -m radiology_modeling.poc
```

결과물:

- `artifacts/poc_synthetic_fit.png`
- `artifacts/poc_fit_summary.csv`

---

## 9. 요약

현재 POC는 다음 문제를 푼다.

> Figure 4 계열 radio-immune discrete model에서, synthetic noisy total tumor volume 데이터를 가장 잘 설명하는 `psi`, `omega`, `kappa`, `T_0`를 찾는다.

그리고 그 방법은 다음이다.

> forward simulation을 반복하면서, 예측 부피와 관측 부피의 log-space residual이 최소가 되도록 `differential_evolution -> least_squares`로 파라미터를 추정한다. 여기에 weak log-prior regularization을 추가해 식별성 약한 방향의 불안정을 줄인다.

---

## 10. 논문 전체 관점에서 최종적으로 찾을 파라미터

논문 전체를 기준으로 보면, 모든 기호를 한 번에 전부 자유롭게 fit 하는 것은 권장되지 않는다. 파라미터는 역할에 따라 나눠서 다뤄야 한다.

### 10.1 직접 추정 대상이 아닌 것

아래 항목은 원칙적으로 **실험 설계 또는 문헌값으로 먼저 고정**하는 편이 낫다.

- treatment schedule: fraction 수, treatment day, prescribed dose, coverage, voxel dose distribution
- 종(species) / 실험계가 정해주는 값: 일부 half-life, 일부 radiosensitivity 초기값
- 수치 구현 선택값: clearance kernel의 mean / std 같은 보조 하이퍼파라미터

즉 이런 값들은 "찾는 파라미터"라기보다 입력 또는 strong prior 대상이다.

### 10.2 동역학 식에서 핵심 추정 대상

논문의 주 동역학 식 기준으로 최종 후보 파라미터는 아래 집합이다.

\[
\Theta_{core} = (\mu, \rho, \omega, \lambda_D, \lambda_L, \lambda_A, \psi, \kappa, \gamma)
\]

여기에 radiation sensitivity가 추가된다.

\[
\Theta_{RT} = (\alpha_T, \beta_T, \alpha_L, \beta_L)
\]

그리고 초기조건이 추가된다.

\[
\Theta_{init} = (T_0, D_0, L_0, A_0)
\]

다만 실전에서는 `D_0 = 0`, `L_0 = 0`, `A_0 = 1` 같은 단순 초기조건을 먼저 두고 시작하는 것이 보통 더 안정적이다.

### 10.3 논문 전체를 기준으로 한 실전 분류

실전에서는 아래처럼 나누는 것이 가장 합리적이다.

#### A. 문헌 고정 또는 strong prior

- `alpha_L`, `beta_L`
- `lambda_D`
- `lambda_L`
- `lambda_A`
- 경우에 따라 `alpha_T`, `beta_T`의 중심값

이들은 데이터가 충분하지 않으면 자유롭게 풀수록 식별성이 급격히 나빠진다.

#### B. 실험/종양 유형별 주요 추정 파라미터

- `mu`
- `rho`
- `omega`
- `psi`
- `kappa`
- 필요시 `gamma`
- 필요시 `alpha_T`, `beta_T`

이 집합이 실제 biological meaning과 데이터 적합도 사이의 핵심 trade-off를 만든다.

#### C. 개체별 또는 세션별 파라미터

- `T_0`
- 필요시 `A_0`, `L_0`
- patient-specific 또는 experiment-specific scaling parameter

즉 마우스 여러 마리나 환자 여러 명을 함께 보면, `T_0`는 개체별 random effect로 가는 것이 자연스럽다.

### 10.4 논문의 모든 식을 다 만족시킨다는 뜻

중요한 점은 논문의 식이 두 종류라는 것이다.

1. **주 시뮬레이션 식**: `T, L, A, D`의 recurrence
2. **유도식**: terminal volume, bifurcation threshold, critical volume 같은 식

예를 들어 equation (3), (4), (6), (9), (10)은 특정 가정 하에서 유도된 관계식이다. 따라서 이것들은 주 데이터에 대해 직접 동일한 weight로 fit 해야 하는 대상이라기보다:

- sanity check
- soft constraint
- posterior predictive diagnostic
- derived endpoint

으로 쓰는 편이 맞다.

즉 최종적으로는 **주 recurrence model을 fit**하고, 유도식들은 그 posterior / fitted parameter가 이론적으로 말이 되는지 확인하는 데 써야 한다.

### 10.5 데이터가 늘어나면 최종적으로 남는 파라미터 계층

논문 전체 수준으로 가면 파라미터는 보통 아래 3계층으로 정리된다.

#### Global / shared

- species- or cohort-level radiosensitivity
- immune cell decay constants
- recovery constants

#### Tumor-model / cohort-specific

- `mu`, `rho`, `omega`, `psi`, `kappa`, `gamma`

#### Subject-specific

- `T_0`
- 필요시 patient-specific suppression / response scaling
- retreatment case에서 session-specific latent state

이 구조가 결국 hierarchical model 구조로 이어진다.

## 11. 나중에 Bayesian을 써야 하는가

결론은 **예, 거의 반드시 쓰는 쪽이 맞다**이다. 다만 처음부터 Bayesian만 붙들 필요는 없다.

### 11.1 왜 결국 Bayesian이 필요한가

이 논문 모델은 아래 이유 때문에 점추정 하나로 끝내기 어렵다.

1. `psi`, `omega`, `kappa`, `rho` 사이에 강한 상관이 있다.
2. 관측치는 주로 `T + D`이고, `L`, `A`, `Z`는 숨은 상태다.
3. partial vs full irradiation, long-term retreatment, patient voxel model까지 가면 데이터 구조가 이질적이다.
4. 우리가 실제로 궁금한 것은 종종 단일 파라미터보다
   - bifurcation threshold
   - critical tumor volume
   - regrowth delay
   - control probability
   같은 derived quantity의 불확실성이다.

이런 상황에서는

\[
p(\theta \mid \text{data})
\]

즉 posterior 분포가 필요하다.

### 11.2 Bayesian을 쓰면 좋은 점

- 파라미터 상관관계 확인 가능
- identifiability가 약한 파라미터를 명확히 볼 수 있음
- 문헌값을 prior로 자연스럽게 넣을 수 있음
- mouse와 patient 데이터를 계층적으로 함께 다룰 수 있음
- critical volume 같은 derived endpoint의 credible interval 계산 가능

### 11.3 추천 순서

현재 프로젝트에는 아래 순서가 현실적이다.

1. **지금처럼 deterministic optimization POC**
   - 구현 검증
   - synthetic recovery 확인
2. **profile likelihood / sensitivity 분석**
   - 어떤 파라미터가 약한지 확인
3. **Bayesian calibration**
   - low-dimensional black-box면 `emcee` 가능
   - likelihood를 명시하기 어렵거나 summary statistic 중심이면 `ABC-SMC` 가능
   - 여러 마우스/환자 데이터를 함께 묶으면 hierarchical Bayesian model 권장

### 11.4 어떤 Bayesian이 맞나

현재 구현처럼 forward simulator가 있고, 관측오차를 log-normal로 둘 수 있다면 가장 직관적인 것은

\[
\log V^{obs} \sim \mathcal{N}(\log V^{pred}(\theta), \sigma^2)
\]

형태의 likelihood를 쓰는 것이다.

이 경우 posterior는

\[
p(\theta \mid data) \propto p(data \mid \theta) p(\theta)
\]

가 된다.

초기 단계에서는 `emcee` 같은 MCMC가 가장 다루기 쉽다. 반면:

- likelihood가 불안정하거나
- latent state가 많고
- 데이터 modality가 섞이고
- 유도량까지 함께 맞추고 싶으면

논문이 언급한 것처럼 **ABC 또는 Monte Carlo calibration**이 꽤 잘 맞는다.

### 11.5 내 추천

- **지금 Figure 4(B) POC 단계**: Bayesian 필수 아님
- **실데이터 단일 실험 fit 단계**: optimization + profile likelihood면 충분히 시작 가능
- **논문 전체 수준 통합, 여러 실험/환자/재치료까지 갈 단계**: Bayesian calibration 사실상 필요

가장 보수적이고 좋은 경로는:

> deterministic optimizer로 초기 후보를 찾고, 그 주변에서 Bayesian posterior를 샘플링하는 2단계 접근

이다.

---

## 12. 최종 추정 파라미터 세트 제안

아래 표는 논문 전체를 실제 프로젝트로 가져갈 때 어떤 파라미터를 어떤 수준에서 추정할지에 대한 실전 제안이다.

### 12.1 마우스 Figure 4/5 수준

|구분|파라미터|권장 처리|
|---|---|---|
|고정 또는 strong prior|`alpha_L`, `beta_L`, `lambda_D`, `lambda_L`, `lambda_A`|문헌값 또는 좁은 prior|
|상황에 따라 고정/약하게 추정|`alpha_T`, `beta_T`|종양주별 문헌값 중심|
|주요 추정|`mu`, `rho`, `omega`, `psi`, `kappa`|주 fit 대상|
|선택 추정|`gamma`|장기 데이터가 있을 때만|
|개체별|`T_0`|각 마우스별 추정|
|초기 잠재상태|`D_0=0`, `L_0=0`, `A_0=1`|우선 고정|

이 수준에서는 보통

\[
\Theta_{mouse} = (\mu, \rho, \omega, \psi, \kappa, T_0)
\]

를 중심으로 시작하고, 필요 시 `alpha_T`, `beta_T`를 추가한다.

### 12.2 환자 단일 사례 수준

|구분|파라미터|권장 처리|
|---|---|---|
|공유 또는 strong prior|`alpha_L`, `beta_L`, `lambda_D`, `lambda_L`, `lambda_A`|문헌값 + 비교적 강한 prior|
|환자/종양 주요 추정|`mu`, `rho`, `omega`, `psi`, `kappa`|핵심 추정 대상|
|상황 의존 추정|`gamma`|장기 추적/재치료가 있으면 고려|
|상황 의존 추정|`alpha_T`, `beta_T`|실제 RT response 차이가 크면 포함|
|세션별|`T_0`|각 치료 세션 시작 시점별|
|세션별 잠재상태|`A_0`, `L_0`|재치료면 latent state로 고려 가능|

환자 데이터에서는 voxel dose distribution이 직접 들어가므로, `psi`와 `kappa`의 posterior 불확실성이 더 중요해진다.

### 12.3 여러 마우스 / 여러 환자 통합 수준

이 단계에서는 계층모델이 자연스럽다.

|계층|파라미터|의미|
|---|---|---|
|Global/shared|`alpha_L`, `beta_L`, `lambda_D`, `lambda_L`, `lambda_A`|species/cohort 공통 생물학|
|Cohort/tumor-type|`mu`, `rho`, `omega`, `psi`, `kappa`, `gamma`, 필요시 `alpha_T`, `beta_T`|종양 유형 또는 코호트 수준|
|Subject-specific|`T_0`, 필요시 response scaling|개체 차이|
|Session-specific|재치료 시 `A_0`, `L_0`, session state|이전 치료 이력 반영|

이 경우 개념적으로는

\[
\theta_s \sim p(\theta_s \mid \phi), \qquad \phi = \text{cohort-level hyperparameters}
\]

형태가 된다.

즉 각 subject의 파라미터 `theta_s`는 상위 hyperparameter `phi`를 공유하는 hierarchical Bayesian 구조로 가는 것이 맞다.

### 12.4 가장 현실적인 최종 세트

현재 프로젝트를 무리 없이 확장하려면 최종적으로는 아래 세트를 중심으로 생각하는 것이 좋다.

#### 1차 핵심 세트

\[
\Theta_{stage1} = (\mu, \rho, \omega, \psi, \kappa, T_0)
\]

#### 2차 확장 세트

\[
\Theta_{stage2} = (\mu, \rho, \omega, \psi, \kappa, \gamma, \alpha_T, \beta_T, T_0)
\]

#### 3차 장기/재치료 세트

\[
\Theta_{stage3} = (\mu, \rho, \omega, \psi, \kappa, \gamma, \alpha_T, \beta_T, T_0, A_0, L_0)
\]

반면
\[
(\alpha_L, \beta_L, \lambda_D, \lambda_L, \lambda_A)
\]

는 되도록 shared parameter 또는 strong-prior parameter로 두는 편이 낫다.

### 12.5 추천 결론

- **POC / Figure 4(B)**: `psi`, `omega`, `kappa`, `T_0`에서 시작
- **실데이터 단일 실험**: `mu`, `rho`까지 확장
- **논문 전체 수준**: `gamma`, `alpha_T`, `beta_T` 추가 검토
- **여러 개체/재치료 통합**: hierarchical Bayesian으로 전환

즉 최종적으로는 “모든 파라미터를 전부 자유롭게 찾는다”가 아니라,

> **공유 파라미터 + 코호트 파라미터 + 개체 파라미터 + 세션 파라미터**

로 분리해서 찾는 것이 맞다.

---

## 13. 무엇이 찾기 쉽고, 임상적으로 진짜 중요한가

### 13.1 상대적으로 찾기 쉬운 파라미터

현재 모델에서 아래 항목은 비교적 찾기 쉽거나, 적어도 strong prior를 주기 쉬운 편이다.

#### `T_0`

- 초기 관측점이 있으면 가장 직접적으로 제약된다.
- 곡선 전체의 절대 스케일에 크게 작용한다.
- 개체별 파라미터로 두기 쉽다.

#### `mu`

- radiation 이전 성장 구간이나 충분한 추적기간이 있으면 상대적으로 안정적으로 잡힌다.
- doubling time과 직접 연결되므로 해석도 쉽다.

#### `alpha_T`, `beta_T` 중 일부 조합

- 균질 조사, 반복 fraction, 충분한 종양반응 데이터가 있으면 일부는 제약 가능하다.
- 다만 `alpha_T`와 `beta_T`를 동시에 자유롭게 풀면 불안정할 수 있어서, 보통은 문헌 중심값 + prior가 더 낫다.

#### `lambda_D`

- 관측량이 `T + D`이기 때문에 치료 직후 감소 이후의 volume tail에서 어느 정도 정보가 있다.
- 그래도 완전 자유 추정보다는 문헌 기반 prior가 안전하다.

### 13.2 찾기 어려운 파라미터

아래는 구조적으로 얽힘이 심해서 데이터가 적으면 찾기 어렵다.

#### `rho`, `psi`, `omega`

- `rho`: basal infiltration
- `psi`: radiation-triggered immune activation
- `omega`: activated CTL의 실제 killing 효율

셋 다 결국 immune effect `Z`를 키우는 방향으로 작용하므로 서로 대체되기 쉽다.

#### `kappa`

- 후반부 재성장과 suppression에 중요하지만, `omega`와 강하게 trade-off 된다.
- long follow-up이 없으면 점추정이 과신되기 쉽다.

#### `gamma`

- secondary immune memory effect라서 장기 데이터가 없으면 거의 못 잡는다.
- short-term mouse experiment에서는 사실상 고정 또는 제외가 맞다.

#### `A_0`, `L_0`

- 직접 관측이 거의 없고, 다른 면역 파라미터와 섞인다.
- retreatment 같은 특수 상황이 아니면 보통 고정이 낫다.

### 13.3 방사선 치료 관점에서 진짜 알고 싶은 것

임상적으로 정말 중요한 것은 파라미터 자체보다 **파라미터가 결정하는 의사결정 변수와 예후 지표**다.

#### 1. 지금 이 환자/종양이 immune-limited 인가, immune-escape 인가

즉 현재 종양이

- 아직 면역으로 붙잡을 수 있는 영역인지
- 이미 escape 쪽으로 넘어간 상태인지

가 핵심이다.

이는 `kappa`, `omega`, `psi`, `rho`의 조합으로 결정된다.

#### 2. critical tumor volume이 어디인가

논문 식 관점에서는

- 현재 종양 크기가 critical volume 아래인지
- 치료 지연 시 그 문턱을 넘는지

가 매우 중요하다.

임상적으로는 이게

- partial irradiation를 써도 되는지
- 먼저 빠른 debulking이 필요한지
- 치료 시점 지연이 위험한지

와 직결된다.

#### 3. partial irradiation가 full irradiation보다 유리한 조건이 뭔가

방사선 치료 입장에서는 이게 매우 중요하다.

즉 궁금한 것은:

- 어떤 `psi`, `kappa`, `rho`, `mu` 조건에서
- partial SFRT가 장기 제어에 유리한가
- 반대로 어떤 조건에서는 conventional full irradiation가 더 안전한가

이다.

#### 4. regrowth delay와 local control probability

실제로 의사결정에 가까운 출력은 다음이다.

- 재성장까지 걸리는 시간
- 일정 기간 내 control 유지 확률
- prescribed dose / coverage 변화에 대한 민감도

즉 “파라미터 값이 얼마냐”보다

> 이 파라미터 조합에서 **10 Gy 50% coverage**와 **10 Gy 100% coverage** 중 어느 쪽이 더 안전하고 오래 듣는가

가 더 중요하다.

#### 5. dose escalation이 진짜 도움이 되는가

논문에서도 보이듯 더 높은 dose가 항상 더 좋은 것은 아니다. CTL까지 같이 죽이면 오히려 면역 효과가 줄 수 있다.

따라서 진짜 알고 싶은 것은

- dose를 올리면 direct kill 이득이 immune depletion 손해보다 큰가
- optimal dose window가 어디인가

이다.

#### 6. retreatment가 가능한가

환자 사례까지 가면 핵심은 다음이다.

- 이전 치료 뒤 남아 있는 immune state가 무엇인가
- 재치료가 다시 immune boost를 줄 수 있는가
- 다음 세션 시점의 `A_0`, `L_0`를 어떻게 봐야 하는가

즉 장기적으로는 단일 세션 fit보다 **state tracking**이 더 중요해진다.

### 13.4 그래서 최종 목적함수도 바뀌어야 한다

연구 초기에야 파라미터 recovery가 목적이지만, 방사선 치료 관점의 최종 목표는 아래와 같은 derived endpoint 예측이다.

- bifurcation proximity
- critical volume exceedance risk
- regrowth delay
- local control probability
- partial vs full irradiation benefit map
- retreatment benefit / risk

즉 최종적으로는

> 파라미터를 찾는 것 자체가 목적이 아니라, **치료 의사결정에 필요한 위험도와 이득을 계산하기 위해 파라미터를 추정하는 것**

이 맞다.

---

## 14. noisy simulation design study에서 바로 얻은 결론

추가 synthetic study를 통해 아래 시나리오를 비교했다.

- `T + D` only, 2-arm (`50%`, `100%`)
- `T + D + Z`, 2-arm
- `T + D + L`, 2-arm
- `T + D + A`, 2-arm
- `T + D` only, 4-arm (`25%`, `50%`, `75%`, `100%`)
- multi-dose / multi-coverage grid

비교 기준은 posterior 90% interval의 상대폭 평균이었다.

### 14.1 관측량 추가의 효과

현재 noisy simulation에서는

- `T + D + Z` 가 가장 좋았고
- 그다음이 `T + D + L`
- 그다음이 multi-arm `T + D`
- `triggering_density` 추가는 상대적으로 이득이 작았다.

즉 synthetic 기준으로는:

> **volume 데이터만 하나 더 모으는 것보다, immune effect 또는 lymphocyte 관련 readout을 추가로 얻는 것이 식별성 개선에 더 효과적이다.**

### 14.2 dose × coverage grid의 효과

multi-dose × multi-coverage 실험에서는

- `total + immune_effect` grid가 최고
- `total` only dose-coverage grid가 그 다음
- 단순 2-arm/4-arm single-dose 디자인보다 generally better

였다.

즉 설계 관점에서 보면:

> **dose 축과 coverage 축을 동시에 흔들고, 가능하면 immune readout을 함께 보는 것이 `psi`, `omega`, `kappa` 분리에 가장 유리하다.**

### 14.3 현재까지의 실전 메시지

실제 실험/임상 데이터를 새로 설계할 수 있다면 우선순위는 대략 다음과 같다.

1. `T + D`만 반복 측정하는 것에서 벗어나기
2. 가능하면 immune-effect proxy 또는 lymphocyte proxy 추가
3. single-dose single-coverage보다 multi-arm design 채택
4. dose와 coverage를 동시에 바꾸는 grid 설계 고려

즉 현재 모델에서는 **무슨 파라미터를 fit하느냐 못지않게, 어떤 관측량과 어떤 treatment arm을 가지느냐가 식별성을 좌우한다.**

---

## 15. Figure 5형 noisy simulation에서 얻은 결론

Figure 5의 핵심은 **같은 dose라도 treatment 시점의 종양 burden이 커지면 partial irradiation가 훨씬 더 민감해진다**는 점이다. 이를 noisy synthetic setting에서 treatment day를 바꾸어 확인했다.

비교한 시나리오는 다음과 같다.

- partial only: `10 Gy @ 50%`, treatment day `10` vs `15`
- full only: `10 Gy @ 100%`, treatment day `10` vs `15`
- mixed 4-arm: partial/full × early/late
- mixed 4-arm + immune effect readout

### 15.1 결과 요약

posterior 평균 상대폭 기준으로는

- `startvol_mixed_4arm_plus_immune` = **0.885**
- `startvol_mixed_4arm_d10` = **0.902**
- `startvol_partial_pair_d10` = **1.033**
- `startvol_full_pair_d10` = **1.070**

였다.

즉 Figure 5형 문제에서도:

> **early/late treatment를 같이 포함한 mixed design이 단일 coverage pair보다 낫고, immune readout을 추가하면 더 좋아진다.**

### 15.2 해석

- treatment timing variation은 `T_0` 하나만 보는 것보다 실제로는 **treatment-time burden**을 바꾸는 역할을 한다.
- 이 변화는 `kappa` 및 `psi`의 효과가 드러나는 조건을 바꾼다.
- 따라서 Figure 5 같은 질문을 다루려면 **치료 시점 variation이 포함된 데이터**가 반드시 필요하다.

즉 단순히 “baseline volume이 다른 케이스”보다,

> **같은 시스템에서 treatment timing을 흔들어 critical regime 근처를 샘플링하는 설계**

가 더 유용하다.

### 15.3 실전 메시지

Figure 4형 질문이 “partial vs full이 왜 다른가”라면,
Figure 5형 질문은 “**언제 partial이 위험해지는가**”이다.

현재 noisy simulation 기준으로는 다음이 맞다.

1. early/late timing arm을 함께 포함하기
2. partial/full arm을 함께 포함하기
3. 가능하면 immune readout 같이 측정하기

즉 Figure 5를 제대로 이해하려면 필요한 추가 데이터는

- **treatment timing variation**
- **partial/full 동시 비교**
- **immune readout**

이다.
