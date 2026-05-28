# 논문 1 - Pepino et al. (2021) `arXiv:2104.03502`

분석 범위: p.1~4 (약 2.5페이지)  
주제: **wav2vec 사용 근거**

| PDF 페이지 | 섹션 | 코드 연결 지점 | 핵심 내용 |
| --- | --- | --- | --- |
| p.1 | Abstract | 코드 전체 배경 | 왜 wav2vec2를 SER에 쓰는지 한 문단 요약 |
| p.1~2 | Section 2. Wav2Vec 2.0 | `Wav2VecExtractor` 클래스 전체 | pretrained 모델 구조, contrastive learning 기반 학습 원리 |
| p.2 | Section 2.4 Downstream model | `last_hidden_state` 단독 사용 vs all layers 가중합 | 여러 레이어 출력을 학습 가능한 가중치(alpha)로 결합하는 근거 |
| p.3~4 | Section 3. Results and discussion | RAVDESS 5-fold, IEMOCAP 결과 비교 | 성능 기준점과 레이어 가중치 해석(중간층 중요) |

---

## 1) wav2vec2를 SER에 쓰는 이유 (p.1 중심)

- SER 데이터셋(IEMOCAP, RAVDESS 등)은 크기가 작아서, 복잡한 DNN을 처음부터 학습하면 일반화가 어렵다고 전제.
- 그래서 대규모 음성 데이터로 사전학습된 wav2vec 2.0 표현을 **전이학습 feature extractor**로 사용.
- 핵심 주장:
  - 저수준 음향 정보(local) + 문맥 정보(contextual)를 함께 제공하는 표현이 SER에 유리.
  - 작은 데이터셋에서도 복잡한 감정 특징을 안정적으로 활용 가능.

---

## 2) Section 2 정리 (p.1~2)

### 2.1 아키텍처 요지
- wav2vec 2.0은
  - local encoder (CNN 기반),
  - contextualized encoder (Transformer 기반),
  - quantization module
  로 구성됨.

### 2.2 학습 원리 요지
- self-supervised pretraining에서 마스킹된 프레임을 맞추는 contrastive objective 사용.
- 본 논문은 두 모델 비교:
  - `Wav2vec2-PT`: pretraining만 된 모델
  - `Wav2vec2-FT`: ASR로 finetuning된 모델

### 2.3~2.4 핵심 구현 포인트
- 단일 출력층만 쓰지 않고, local + transformer 내부층 + contextual 출력을 모두 활용.
- 학습 가능한 가중치 `alpha_i`로 레이어별 출력을 결합:
  - `f = (sum(alpha_i * f_i)) / (sum(alpha_i))`
- 저자 결론: 다층 결합(`All layers`)이 단일 레이어 사용보다 성능이 좋음.

---

## 3) 표/그래프 해석 (p.3~4)

### Table 1 해석
- Dense downstream 기준 평균 recall:
  - **Wav2vec2-PT + All layers**가 IEMOCAP/RAVDESS 모두 최고.
  - PT가 FT보다 전반적으로 우세.
- 해석:
  - ASR finetuning(FT) 과정에서 SER에 유용한 정보(예: prosody 일부)가 약해질 수 있다는 저자 주장과 일치.

### Figure 2 해석 (레이어 가중치 alpha)
- 중간 transformer 레이어 가중치가 크게 학습되는 경향.
- 해석:
  - 중간층이 감정 인식에 필요한 문맥 정보를 충분히 담으면서도, 최종층 대비 과도한 ASR 특화가 덜하다는 근거로 제시.

### Table 2 해석 (정규화/모델 변형)
- speaker normalization이 global normalization보다 성능 우세.
- LSTM 추가는 항상 이득을 보장하지 않음.
- eGeMAPS 결합(Fusion)은 일부 개선을 보임.

---

## 4) 구현팀 전달용 요약

- **필수 실험축 1**: `last_hidden_state` 단독 vs `hidden_states all-layer weighted sum`
- **필수 실험축 2**: `Wav2vec2-PT` vs `Wav2vec2-FT`
- **평가 기준**: 평균 recall(논문 기준), 데이터셋별 동일 분할 유지
- **기대 포인트**:
  - all-layer 결합이 단일층보다 유리
  - PT가 FT보다 SER에서 더 나을 가능성

---

## 5) 데이터셋 다운로드 메모

- IEMOCAP: [https://sail.usc.edu/iemocap/](https://sail.usc.edu/iemocap/)
- RAVDESS: [https://zenodo.org/record/1188976](https://zenodo.org/record/1188976)

