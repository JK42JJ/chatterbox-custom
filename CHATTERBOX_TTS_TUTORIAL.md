# Chatterbox TTS 완벽 가이드

> 로컬에서 무료로 실행 가능한 고품질 Text-to-Speech 시스템

## 목차

1. [아키텍처 개요](#아키텍처-개요)
2. [모델 비교](#모델-비교)
3. [설치 및 환경 설정](#설치-및-환경-설정)
4. [기본 사용법](#기본-사용법)
5. [Voice Cloning (음성 복제)](#voice-cloning-음성-복제)
6. [긴 텍스트 처리](#긴-텍스트-처리)
7. [응용 시나리오](#응용-시나리오)
8. [비용 및 라이선스](#비용-및-라이선스)
9. [FAQ](#faq)

---

## 아키텍처 개요

### 시스템 구조

```
┌─────────────────────────────────────────────────────────────────┐
│                    Chatterbox TTS Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Text       │    │   Voice      │    │   Audio      │       │
│  │   Input      │───▶│   Synthesis  │───▶│   Output     │       │
│  │              │    │   Engine     │    │   (.wav)     │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│         │                   │                                    │
│         ▼                   ▼                                    │
│  ┌──────────────┐    ┌──────────────┐                           │
│  │   Text       │    │   Reference  │                           │
│  │   Tokenizer  │    │   Voice      │                           │
│  │   (2048 max) │    │   (Optional) │                           │
│  └──────────────┘    └──────────────┘                           │
│                                                                  │
├─────────────────────────────────────────────────────────────────┤
│                      Model Variants                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐  │
│  │ ChatterboxTTS   │  │ ChatterboxTurbo │  │ Chatterbox      │  │
│  │ (500M params)   │  │ (350M params)   │  │ Multilingual    │  │
│  │                 │  │                 │  │                 │  │
│  │ • High Quality  │  │ • Low Latency   │  │ • 23 Languages  │  │
│  │ • English       │  │ • Paralinguistic│  │ • Voice Clone   │  │
│  │ • Voice Clone   │  │ • Voice Clone   │  │                 │  │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 핵심 컴포넌트

| 컴포넌트 | 역할 | 기술 |
|---------|------|------|
| **Text Encoder** | 텍스트를 임베딩으로 변환 | LLaMA 기반 |
| **Voice Encoder** | 참조 음성의 특성 추출 | Perth Network |
| **Diffusion Decoder** | 음성 파형 생성 | Flow Matching |
| **Watermarking** | 생성된 오디오에 워터마크 삽입 | Perth (무음, 감지 불가) |

### 처리 파이프라인

```
Text Input
    │
    ▼
┌─────────────────┐
│ Text Tokenizer  │ ◀── 최대 2048 토큰
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌─────────────────┐
│ Text Embedding  │◀────│ Voice Reference │ (선택적)
└────────┬────────┘     └─────────────────┘
         │
         ▼
┌─────────────────┐
│ Diffusion Model │ ◀── 1000 Sampling Steps
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Perth Watermark │ ◀── 무음 워터마크 삽입
└────────┬────────┘
         │
         ▼
   Audio Output (.wav, 24kHz)
```

---

## 모델 비교

### 3가지 모델 상세 비교

| 특성 | ChatterboxTTS | ChatterboxTurbo | ChatterboxMultilingual |
|------|---------------|-----------------|------------------------|
| **파라미터** | 500M | 350M | - |
| **언어** | 영어 전용 | 영어 전용 | 23개 언어 |
| **품질** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **속도** | 보통 | 빠름 | 보통 |
| **지연시간** | 높음 | 낮음 | 높음 |
| **Voice Cloning** | ✅ | ✅ | ✅ |
| **Paralinguistic** | ❌ | ✅ | ❌ |
| **용도** | 고품질 영어 TTS | 실시간/스트리밍 | 다국어 지원 |

### Paralinguistic Tags (Turbo 전용)

Turbo 모델은 감정과 비언어적 표현을 지원합니다:

```python
# 사용 가능한 태그
text = "That's hilarious! [laughs] I can't believe it."
text = "I'm not sure about this... [sighs] Let me think."
text = "[chuckle] Well, that was unexpected!"
```

| 태그 | 의미 | 예시 |
|------|------|------|
| `[laughs]` | 웃음 | "That's funny! [laughs]" |
| `[chuckle]` | 낮은 웃음 | "[chuckle] Nice one." |
| `[sighs]` | 한숨 | "[sighs] I don't know..." |
| `[clears throat]` | 헛기침 | "[clears throat] Ahem..." |

### 모델 선택 가이드

```
어떤 모델을 선택해야 할까요?

         ┌─────────────────────────────┐
         │ 영어 외 다른 언어가 필요한가? │
         └─────────────┬───────────────┘
                       │
           ┌───────────┴───────────┐
           ▼                       ▼
          Yes                     No
           │                       │
           ▼                       ▼
  ┌─────────────────┐     ┌─────────────────┐
  │ Multilingual    │     │ 실시간 응답이    │
  │ (23개 언어)      │     │ 필요한가?       │
  └─────────────────┘     └────────┬────────┘
                                   │
                       ┌───────────┴───────────┐
                       ▼                       ▼
                      Yes                     No
                       │                       │
                       ▼                       ▼
              ┌─────────────────┐     ┌─────────────────┐
              │ Turbo (350M)    │     │ TTS (500M)      │
              │ 저지연, 빠른 응답 │     │ 최고 품질       │
              └─────────────────┘     └─────────────────┘
```

---

## 프로젝트 및 모델 파일 구조

### 전체 구조 개요

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    Chatterbox 파일 구조 개요                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  소스 코드 (Git Clone)                  학습된 모델 (자동 다운로드)        │
│  ┌─────────────────────┐                ┌─────────────────────┐         │
│  │ ~/chatterbox/       │                │ ~/.cache/           │         │
│  │   chatterbox/       │                │   huggingface/      │         │
│  │     src/            │   실행 시 →    │     hub/            │         │
│  │     examples/       │   자동 다운로드  │       models--*/    │         │
│  └─────────────────────┘                └─────────────────────┘         │
│                                                                          │
│  • 소스: GitHub에서 clone                • 모델: HuggingFace Hub에서 캐시  │
│  • 용량: ~10MB                          • 용량: ~10GB (전체 모델)         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### 프로젝트 소스 코드 구조

```
chatterbox/                          # 프로젝트 루트
├── src/
│   └── chatterbox/                  # 메인 패키지
│       ├── __init__.py
│       ├── tts.py                   # ChatterboxTTS (500M 영어)
│       ├── tts_turbo.py             # ChatterboxTurboTTS (350M 저지연)
│       ├── mtl_tts.py               # ChatterboxMultilingualTTS (23개 언어)
│       ├── vc.py                    # ChatterboxVC (Voice Conversion)
│       │
│       └── models/                  # 모델 아키텍처 정의
│           ├── t3/                  # T3 (Text-to-Token) 모델
│           │   ├── t3.py            # 메인 T3 클래스
│           │   ├── llama_configs.py # LLaMA 기반 설정
│           │   ├── inference/       # 추론 관련
│           │   └── modules/         # 서브 모듈
│           │
│           ├── s3gen/               # S3Gen (Token-to-Audio) 모델
│           │   ├── flow.py          # Flow Matching
│           │   ├── decoder.py       # 오디오 디코더
│           │   ├── hifigan.py       # HiFi-GAN 보코더
│           │   └── transformer/     # Transformer 레이어
│           │
│           ├── s3tokenizer/         # Speech Tokenizer
│           │   └── s3tokenizer.py
│           │
│           ├── voice_encoder/       # Voice Encoder (음성 특성 추출)
│           │   ├── voice_encoder.py
│           │   ├── melspec.py       # Mel Spectrogram
│           │   └── config.py
│           │
│           └── tokenizers/          # Text Tokenizer
│               └── tokenizer.py
│
├── example_tts.py                   # 기본 사용 예제
├── example_tts_turbo.py             # Turbo 모델 예제
├── example_for_mac.py               # Mac MPS 최적화 예제
├── example_vc.py                    # Voice Conversion 예제
│
├── gradio_tts_app.py                # Gradio 웹 데모
├── gradio_tts_turbo_app.py          # Turbo Gradio 데모
├── multilingual_app.py              # 다국어 Gradio 데모
│
└── pyproject.toml                   # 패키지 설정
```

### 학습된 모델 파일 구조

#### 방법 1: 프로젝트 로컬 폴더 (권장 - 오프라인 사용)

모델을 프로젝트 폴더에 직접 저장하여 오프라인에서 사용할 수 있습니다.

```
chatterbox/
└── models/                              # 로컬 모델 폴더 (13GB)
    │
    ├── chatterbox/                      # Standard 모델 (9GB)
    │   ├── t3_cfg.safetensors           # T3 영어 모델 (2GB)
    │   ├── t3_mtl23ls_v2.safetensors    # T3 다국어 버전 (2GB)
    │   ├── t3_23lang.safetensors        # T3 23개 언어 (2GB)
    │   ├── s3gen.safetensors            # S3Gen (Token → Audio, 1GB)
    │   ├── s3gen.pt                     # S3Gen PyTorch 포맷
    │   ├── ve.safetensors               # Voice Encoder (5.4MB)
    │   ├── ve.pt                        # Voice Encoder PyTorch
    │   ├── conds.pt                     # 조건부 임베딩 (105KB)
    │   ├── tokenizer.json               # 텍스트 토크나이저
    │   ├── mtl_tokenizer.json           # 다국어 토크나이저
    │   ├── grapheme_mtl_merged_expanded_v1.json  # Grapheme 매핑
    │   └── Cangjie5_TC.json             # 중국어 입력 매핑
    │
    └── chatterbox-turbo/                # Turbo 모델 (3.8GB)
        ├── t3_turbo_v1.safetensors      # Turbo T3 모델 (1.8GB)
        ├── t3_turbo_v1.yaml             # 모델 설정
        ├── s3gen.safetensors            # S3Gen (1GB)
        ├── s3gen_meanflow.safetensors   # Mean Flow 최적화 (1GB)
        ├── ve.safetensors               # Voice Encoder (5.4MB)
        ├── conds.pt                     # 기본 음성 조건 (165KB)
        ├── vocab.json                   # BPE 어휘 (976KB)
        ├── merges.txt                   # BPE 병합 규칙 (446KB)
        ├── tokenizer_config.json        # 토크나이저 설정
        ├── special_tokens_map.json      # 특수 토큰
        └── added_tokens.json            # 추가 토큰
```

#### 방법 2: HuggingFace Cache (기본값)

모델은 최초 실행 시 자동으로 다운로드되어 HuggingFace 캐시에 저장됩니다.

```
~/.cache/huggingface/hub/
├── models--ResembleAI--chatterbox/           # Standard 모델
│   └── snapshots/<hash>/
│       └── (위와 동일한 파일들)
│
└── models--ResembleAI--chatterbox-turbo/     # Turbo 모델
    └── snapshots/<hash>/
        └── (위와 동일한 파일들)
```

### 모델 파일 상세

| 파일 | 크기 (대략) | 역할 |
|------|------------|------|
| `t3_cfg.safetensors` | ~2GB | 텍스트 → 음성 토큰 변환 (500M) |
| `t3_turbo_v1.safetensors` | ~1.4GB | Turbo 버전 T3 (350M) |
| `s3gen.safetensors` | ~1.5GB | 음성 토큰 → 오디오 파형 |
| `ve.safetensors` | ~50MB | 참조 음성 특성 추출 |
| `conds.pt` | ~10MB | 기본 음성 조건부 임베딩 |
| `tokenizer.json` | ~500KB | 텍스트 토큰화 규칙 |

### 모델 다운로드 위치 변경

```python
import os

# 환경 변수로 캐시 위치 변경
os.environ["HF_HOME"] = "/custom/path/huggingface"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/custom/path/huggingface/hub"

# 이후 모델 로드
from chatterbox.tts import ChatterboxTTS
model = ChatterboxTTS.from_pretrained(device="mps")
```

### 오프라인 사용 설정

#### 방법 1: 프로젝트 로컬 폴더에 복사 (권장)

```bash
# 1. 모델 다운로드
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ResembleAI/chatterbox')
snapshot_download('ResembleAI/chatterbox-turbo')
"

# 2. 프로젝트 폴더로 복사
mkdir -p models/chatterbox models/chatterbox-turbo

# Standard 모델 복사
STANDARD=$(ls ~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/)
cp -L ~/.cache/huggingface/hub/models--ResembleAI--chatterbox/snapshots/$STANDARD/* models/chatterbox/

# Turbo 모델 복사
TURBO=$(ls ~/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/snapshots/)
cp -L ~/.cache/huggingface/hub/models--ResembleAI--chatterbox-turbo/snapshots/$TURBO/* models/chatterbox-turbo/
```

#### 방법 2: HuggingFace 캐시 사용

```bash
# 모델 사전 다운로드 (인터넷 연결 시)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('ResembleAI/chatterbox')
snapshot_download('ResembleAI/chatterbox-turbo')
"
# 이후 오프라인에서 자동으로 캐시 사용
```

### 모델 용량 요약

| 모델 | HuggingFace Repo | 용량 | 파라미터 |
|------|------------------|------|---------|
| **Standard** | `ResembleAI/chatterbox` | ~9GB | 500M |
| **Turbo** | `ResembleAI/chatterbox-turbo` | ~3.8GB | 350M |
| **전체 (로컬 복사 시)** | - | **~13GB** | - |

> **참고**: 로컬 복사 시 중복 파일이 포함되어 13GB, HuggingFace 캐시는 공유 파일로 약 10GB입니다.

---

## 설치 및 환경 설정

### 요구 사항

- Python 3.11+
- PyTorch 2.0+
- 8GB+ GPU VRAM (또는 MPS for Apple Silicon)
- 10GB+ 디스크 공간 (모델 다운로드)

### 설치

```bash
# 저장소 클론
git clone https://github.com/resemble-ai/chatterbox.git
cd chatterbox

# 의존성 설치 (방법 1: 개발 모드)
pip install -e .

# 또는 (방법 2: requirements.txt 사용)
pip install -r requirements.txt
```

### 필수 라이브러리 (requirements.txt)

```txt
# Chatterbox TTS Requirements
# Python >= 3.10 required

# Core ML Frameworks
torch==2.6.0
torchaudio==2.6.0
numpy>=1.24.0,<1.26.0

# HuggingFace
transformers==4.46.3
diffusers==0.29.0
safetensors==0.5.3
huggingface_hub

# Audio Processing
librosa==0.11.0
pyloudnorm

# Speech & Text Processing
s3tokenizer
conformer==0.3.2
spacy-pkuseg
pykakasi==2.3.0

# Watermarking
resemble-perth==1.0.1

# Configuration
omegaconf

# Web UI (Optional - for Gradio demos)
gradio==5.44.1
```

#### 라이브러리 설명

| 카테고리 | 라이브러리 | 용도 |
|---------|-----------|------|
| **Core ML** | `torch`, `torchaudio` | PyTorch 딥러닝 프레임워크 |
| | `numpy` | 수치 연산 |
| **HuggingFace** | `transformers` | Transformer 모델 |
| | `diffusers` | Diffusion 모델 |
| | `safetensors` | 안전한 모델 로딩 |
| | `huggingface_hub` | 모델 다운로드 |
| **Audio** | `librosa` | 오디오 처리 및 분석 |
| | `pyloudnorm` | 오디오 정규화 |
| **Speech/Text** | `s3tokenizer` | Speech 토큰화 |
| | `conformer` | Conformer 아키텍처 |
| | `spacy-pkuseg` | 중국어 분절 |
| | `pykakasi` | 일본어 변환 |
| **Watermark** | `resemble-perth` | Perth 워터마킹 |
| **Config** | `omegaconf` | 설정 관리 |
| **Web UI** | `gradio` | 웹 데모 인터페이스 |

### Mac (Apple Silicon) 환경 설정

Apple M1/M2/M3/M4 칩에서는 MPS 백엔드를 사용합니다:

```python
import torch

# MPS 장치 설정
device = "mps" if torch.backends.mps.is_available() else "cpu"

# torch.load 패치 (MPS 호환성)
map_location = torch.device(device)
torch_load_original = torch.load

def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)

torch.load = patched_torch_load
```

> **중요**: 이 패치는 모델 로딩 전에 적용해야 합니다.

### CUDA (NVIDIA GPU) 환경

```python
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
```

### 빠른 실행 (예제 파일)

프로젝트에 포함된 예제 파일로 바로 테스트할 수 있습니다:

```bash
# 프로젝트 디렉토리로 이동
cd chatterbox

# 1. 기본 TTS 테스트 (영어 + 다국어)
python example_tts.py
# 출력: test-1.wav (영어), test-2.wav (프랑스어)

# 2. Turbo 모델 테스트 (저지연)
python example_tts_turbo.py
# 출력: test-turbo.wav

# 3. Mac 최적화 버전 (긴 텍스트)
python example_for_mac.py
# 출력: sleep-article.wav

# 4. Gradio 웹 데모 실행
python gradio_tts_app.py
# 브라우저에서 http://localhost:7860 접속

# 5. Turbo Gradio 데모
python gradio_tts_turbo_app.py

# 6. 다국어 Gradio 데모
python multilingual_app.py
```

#### 예제 파일 설명

| 파일 | 모델 | 설명 |
|------|------|------|
| `example_tts.py` | Standard + Multilingual | 기본 영어/다국어 TTS |
| `example_tts_turbo.py` | Turbo | 저지연 모델, paralinguistic tags |
| `example_for_mac.py` | Standard | Mac MPS 최적화, 긴 텍스트 처리 |
| `example_vc.py` | Voice Conversion | 음성 변환 (소스 파일 필요) |
| `gradio_tts_app.py` | Standard | 웹 UI 데모 |
| `gradio_tts_turbo_app.py` | Turbo | Turbo 웹 UI 데모 |
| `multilingual_app.py` | Multilingual | 다국어 웹 UI 데모 |

---

## 기본 사용법

### 1. 기본 TTS (영어)

```python
import torchaudio as ta
from chatterbox.tts import ChatterboxTTS

# 모델 로드
model = ChatterboxTTS.from_pretrained(device="mps")  # 또는 "cuda"

# 텍스트를 음성으로 변환
text = "Hello, this is a test of the Chatterbox text-to-speech system."
wav = model.generate(text)

# 파일 저장
ta.save("output.wav", wav, model.sr)
print(f"Sample rate: {model.sr}")  # 24000 Hz
```

### 2. Turbo 모델 (저지연)

```python
from chatterbox.tts_turbo import ChatterboxTurboTTS

# Turbo 모델 로드
model = ChatterboxTurboTTS.from_pretrained(device="mps")

# 기본 음성으로 생성
text = "This is the Turbo model with lower latency."
wav = model.generate(text)
ta.save("turbo_output.wav", wav, model.sr)

# Paralinguistic 태그 사용
text_with_emotion = "That's amazing! [laughs] I love it!"
wav = model.generate(text_with_emotion)
ta.save("turbo_emotional.wav", wav, model.sr)
```

### 3. 다국어 TTS

```python
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="mps")

# 프랑스어
text_fr = "Bonjour, comment ça va?"
wav = model.generate(text_fr, language_id="fr")
ta.save("french.wav", wav, model.sr)

# 한국어
text_ko = "안녕하세요, 반갑습니다."
wav = model.generate(text_ko, language_id="ko")
ta.save("korean.wav", wav, model.sr)
```

#### 지원 언어 코드

| 코드 | 언어 | 코드 | 언어 |
|------|------|------|------|
| en | English | ko | 한국어 |
| fr | Français | ja | 日本語 |
| de | Deutsch | zh | 中文 |
| es | Español | ru | Русский |
| it | Italiano | pt | Português |

---

## Voice Cloning (음성 복제)

### 원리

Voice Cloning은 참조 음성 파일의 특성(음색, 톤, 억양)을 추출하여 새로운 텍스트에 적용합니다.

```
┌─────────────────┐     ┌─────────────────┐
│ Reference Audio │────▶│ Voice Encoder   │
│ (5+ seconds)    │     │ (Perth Network) │
└─────────────────┘     └────────┬────────┘
                                 │
                                 ▼
                        ┌─────────────────┐
                        │ Voice Embedding │
                        └────────┬────────┘
                                 │
                                 ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│ New Text        │────▶│ TTS Engine      │────▶│ Cloned Voice    │
│                 │     │ + Voice Embed   │     │ Output          │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 사용 방법

```python
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="mps")

# 참조 음성 파일 경로 (5초 이상 권장)
reference_audio = "reference_voice.wav"

# Voice Cloning으로 생성
text = "This will be spoken in the cloned voice."
wav = model.generate(
    text,
    audio_prompt_path=reference_audio,
    exaggeration=1.0,  # 음성 특성 강조 (기본값: 1.0)
    cfg_weight=0.5     # 가이던스 가중치
)
ta.save("cloned_output.wav", wav, model.sr)
```

### 파라미터 설명

| 파라미터 | 범위 | 설명 |
|---------|------|------|
| `audio_prompt_path` | 파일 경로 | 참조 음성 파일 (.wav) |
| `exaggeration` | 0.0 ~ 2.0+ | 음성 특성 강조 정도 |
| `cfg_weight` | 0.0 ~ 1.0 | 텍스트-음성 정렬 강도 |

### 좋은 참조 음성의 조건

1. **길이**: 5~15초가 이상적
2. **품질**: 깨끗한 녹음 (잡음 최소화)
3. **내용**: 다양한 발음이 포함된 문장
4. **형식**: WAV 파일 권장

---

## 긴 텍스트 처리

### 문제점

Chatterbox TTS는 **최대 2048 토큰** 제한이 있습니다. 긴 텍스트는 자동으로 잘리거나 메모리 오류가 발생할 수 있습니다.

### 해결 방법: 청킹 (Chunking)

```python
import torch
import torchaudio as ta
import gc
from chatterbox.tts_turbo import ChatterboxTurboTTS

# MPS 호환 패치
device = "mps"
map_location = torch.device(device)
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = map_location
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

# 모델 로드
model = ChatterboxTurboTTS.from_pretrained(device=device)

# 긴 텍스트를 청크로 분할
texts = [
    "First paragraph of your long text...",
    "Second paragraph continues here...",
    "Third paragraph with more content...",
    # 각 청크는 500-1000자 권장
]

# 청크별 생성 및 결합
all_wavs = []
for i, text in enumerate(texts):
    print(f"Generating chunk {i+1}/{len(texts)}...")
    wav = model.generate(text)
    all_wavs.append(wav)
    gc.collect()  # 메모리 정리 (MPS에서 중요)

# 모든 오디오 결합
final_wav = torch.cat(all_wavs, dim=1)
ta.save("long_audio.wav", final_wav, model.sr)
print(f"Total duration: {final_wav.shape[1] / model.sr:.1f} seconds")
```

### 청킹 가이드라인

| 항목 | 권장 값 | 이유 |
|------|---------|------|
| 청크 크기 | 500-1000자 | 메모리 효율성 |
| 문장 끊기 | 마침표 기준 | 자연스러운 억양 |
| 메모리 정리 | 청크마다 `gc.collect()` | MPS 메모리 누수 방지 |

### 자동 청킹 함수

```python
def chunk_text(text, max_chars=800):
    """텍스트를 문장 단위로 청킹"""
    sentences = text.replace('!', '.').replace('?', '.').split('.')
    chunks = []
    current_chunk = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if len(current_chunk) + len(sentence) < max_chars:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks

# 사용 예시
long_text = """Your very long text here..."""
chunks = chunk_text(long_text)
print(f"Split into {len(chunks)} chunks")
```

---

## 응용 시나리오

### 시나리오 1: 영어 학습 오디오북

```python
"""
영어 학습용 오디오북 생성
- 원어민 수준의 발음
- 자연스러운 억양
- 긴 텍스트 자동 처리
"""
import torch
import torchaudio as ta
import gc
from chatterbox.tts_turbo import ChatterboxTurboTTS

# 설정
device = "mps"
# ... (MPS 패치 코드)

model = ChatterboxTurboTTS.from_pretrained(device=device)

# 학습 콘텐츠
lesson_content = [
    """Most sleep experts advise that adults get seven to nine
    hours of sleep per night for good health and emotional
    well-being, although that changes as you get older.""",

    """Sleep quality is just as important as sleep time.
    A lot of people tend to focus on how many hours of sleep
    they're getting but neglect the quality of their sleep.""",

    # 추가 콘텐츠...
]

# 오디오북 생성
all_wavs = []
for i, text in enumerate(lesson_content):
    print(f"Recording section {i+1}...")
    wav = model.generate(text)
    all_wavs.append(wav)
    gc.collect()

final_wav = torch.cat(all_wavs, dim=1)
ta.save("english_lesson.wav", final_wav, model.sr)
print(f"Audiobook created: {final_wav.shape[1]/model.sr/60:.1f} minutes")
```

### 시나리오 2: 다국어 발표 자료

```python
"""
동일한 내용을 여러 언어로 변환
- 글로벌 프레젠테이션용
- 언어별 자연스러운 발음
"""
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

model = ChatterboxMultilingualTTS.from_pretrained(device="mps")

presentations = {
    "en": "Welcome to our product launch event.",
    "fr": "Bienvenue à notre événement de lancement de produit.",
    "de": "Willkommen zu unserer Produkteinführungsveranstaltung.",
    "ja": "製品発表イベントへようこそ。",
    "ko": "제품 출시 행사에 오신 것을 환영합니다.",
}

for lang_code, text in presentations.items():
    wav = model.generate(text, language_id=lang_code)
    ta.save(f"presentation_{lang_code}.wav", wav, model.sr)
    print(f"Generated: presentation_{lang_code}.wav")
```

### 시나리오 3: 캐릭터 음성 생성

```python
"""
Voice Cloning을 활용한 캐릭터별 음성
- 게임/애니메이션용
- 일관된 캐릭터 음성
"""
from chatterbox.tts import ChatterboxTTS

model = ChatterboxTTS.from_pretrained(device="mps")

characters = {
    "hero": {
        "reference": "voices/hero_sample.wav",
        "lines": [
            "I will protect this city!",
            "Evil will not prevail!",
        ]
    },
    "villain": {
        "reference": "voices/villain_sample.wav",
        "lines": [
            "You cannot stop me...",
            "This world will be mine!",
        ]
    }
}

for char_name, data in characters.items():
    for i, line in enumerate(data["lines"]):
        wav = model.generate(
            line,
            audio_prompt_path=data["reference"],
            exaggeration=1.2
        )
        ta.save(f"{char_name}_line_{i+1}.wav", wav, model.sr)
```

### 시나리오 4: 팟캐스트/내레이션 제작

```python
"""
전문적인 내레이션 음성 생성
- Turbo 모델의 감정 태그 활용
- 자연스러운 대화체
"""
from chatterbox.tts_turbo import ChatterboxTurboTTS

model = ChatterboxTurboTTS.from_pretrained(device="mps")

script = [
    "Welcome back to Tech Talk! [clears throat]",
    "Today we're discussing the future of AI. [sighs] It's complicated.",
    "But first... [chuckle] let me tell you a story.",
    "Isn't that amazing? [laughs] I thought so too!",
]

all_wavs = []
for line in script:
    wav = model.generate(line)
    all_wavs.append(wav)

final_wav = torch.cat(all_wavs, dim=1)
ta.save("podcast_episode.wav", final_wav, model.sr)
```

### 시나리오 5: 접근성 (화면 읽기)

```python
"""
시각 장애인을 위한 문서 읽기
- 웹페이지/문서 내용 음성 변환
- 긴 문서 자동 처리
"""
import requests
from bs4 import BeautifulSoup
from chatterbox.tts_turbo import ChatterboxTurboTTS

def webpage_to_audio(url, output_file):
    # 웹페이지 텍스트 추출
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # 본문 추출 (예시)
    paragraphs = soup.find_all('p')
    text_content = ' '.join([p.get_text() for p in paragraphs])

    # 청킹
    chunks = chunk_text(text_content, max_chars=800)

    # TTS 변환
    model = ChatterboxTurboTTS.from_pretrained(device="mps")
    all_wavs = []

    for chunk in chunks:
        wav = model.generate(chunk)
        all_wavs.append(wav)
        gc.collect()

    final_wav = torch.cat(all_wavs, dim=1)
    ta.save(output_file, final_wav, model.sr)

    return output_file
```

---

## 비용 및 라이선스

### 비용: 완전 무료

| 항목 | 비용 | 설명 |
|------|------|------|
| 모델 다운로드 | $0 | HuggingFace에서 무료 |
| API 호출 | $0 | 로컬 실행, API 없음 |
| 음성 생성 | $0 | 무제한 생성 가능 |
| 상업적 사용 | 확인 필요 | 라이선스 조건 참조 |

### 왜 무료인가?

```
┌─────────────────────────────────────────────────────────────┐
│              Chatterbox TTS 비용 구조                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  클라우드 TTS (유료)          Chatterbox (무료)              │
│  ┌─────────────────┐          ┌─────────────────┐           │
│  │ 텍스트 전송     │          │ 로컬 처리       │           │
│  │      ↓         │          │      ↓         │           │
│  │ 서버 처리      │          │ GPU/CPU 연산    │           │
│  │      ↓         │          │      ↓         │           │
│  │ 과금 ($$$)     │          │ 비용 없음       │           │
│  │      ↓         │          │      ↓         │           │
│  │ 결과 반환      │          │ 즉시 저장       │           │
│  └─────────────────┘          └─────────────────┘           │
│                                                              │
│  • 인터넷 필요               • 오프라인 가능                 │
│  • 사용량 제한               • 무제한 사용                   │
│  • 개인정보 전송             • 데이터 로컬 유지              │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Perth Watermarking

모든 생성 오디오에는 Perth 워터마크가 삽입됩니다:

- **감지 불가**: 사람 귀로 들을 수 없음
- **추적 가능**: Chatterbox로 생성된 오디오 식별 가능
- **제거 불가**: 오디오 압축/변환에도 유지됨
- **용도**: AI 생성 콘텐츠 식별

---

## FAQ

### Q1: "ModuleNotFoundError: No module named 'chatterbox'" 오류

**원인**: 잘못된 Python 환경

**해결**:
```bash
# Homebrew Python 사용 (Mac)
python3.11 example_tts.py

# 또는 설치 확인
pip show chatterbox
```

### Q2: MPS 메모리 오류 (out of memory)

**원인**: 긴 텍스트 또는 메모리 누수

**해결**:
```python
import gc

# 청킹 사용
chunks = chunk_text(long_text, max_chars=500)

# 청크마다 메모리 정리
for chunk in chunks:
    wav = model.generate(chunk)
    all_wavs.append(wav)
    gc.collect()  # 중요!
```

### Q3: Voice Cloning 결과가 원본과 다름

**원인**: 참조 음성 품질 또는 파라미터

**해결**:
1. 참조 음성 5~15초 사용
2. 깨끗한 녹음 (배경 잡음 제거)
3. 파라미터 조정:
```python
wav = model.generate(
    text,
    audio_prompt_path="reference.wav",
    exaggeration=0.8,  # 낮추면 원본에 더 가까움
    cfg_weight=0.7     # 높이면 음성 특성 강화
)
```

### Q4: CUDA 오류 (Mac에서)

**원인**: Mac에는 NVIDIA GPU 없음

**해결**: MPS 사용
```python
device = "mps"  # "cuda" 대신
```

### Q5: 다국어 모델에서 RuntimeError

**원인**: CUDA 전용 저장 모델

**해결**: torch.load 패치 적용
```python
torch_load_original = torch.load
def patched_torch_load(*args, **kwargs):
    if 'map_location' not in kwargs:
        kwargs['map_location'] = torch.device("mps")
    return torch_load_original(*args, **kwargs)
torch.load = patched_torch_load

# 이후 모델 로드
```

### Q6: 생성 속도가 느림

**팁**:
1. Turbo 모델 사용 (350M, 더 빠름)
2. 짧은 텍스트로 분할
3. GPU 가속 확인 (MPS/CUDA)

---

## API 서버 구축 (상용 서비스 연동)

Chatterbox를 유료 서비스의 백엔드 모듈로 활용하는 방법입니다.

### 아키텍처 개요

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Production TTS Service Architecture               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────────┐     ┌──────────────────┐ │
│  │   Frontend   │     │   API Gateway    │     │   TTS Service    │ │
│  │   (Web/App)  │────▶│   (FastAPI)      │────▶│   (Chatterbox)   │ │
│  │              │     │                  │     │                  │ │
│  └──────────────┘     └──────────────────┘     └──────────────────┘ │
│         │                     │                        │            │
│         │              ┌──────┴──────┐                 │            │
│         │              ▼             ▼                 ▼            │
│         │      ┌────────────┐ ┌────────────┐   ┌────────────┐      │
│         │      │ Rate Limit │ │ Auth/Billing│   │ Model Pool │      │
│         │      │ & Caching  │ │   Service  │   │ (GPU×N)    │      │
│         │      └────────────┘ └────────────┘   └────────────┘      │
│         │                                              │            │
│         ▼                                              ▼            │
│  ┌──────────────┐                              ┌──────────────┐    │
│  │ Audio Player │◀─────────────────────────────│ Audio Storage│    │
│  │ (Streaming)  │                              │ (S3/CDN)     │    │
│  └──────────────┘                              └──────────────┘    │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### FastAPI 서버 구현

```python
# tts_server.py
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional
import torch
import torchaudio as ta
import gc
import uuid
import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 설정 및 초기화
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="Chatterbox TTS API",
    description="Production-ready TTS service powered by Chatterbox",
    version="1.0.0"
)

# 장치 설정
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
OUTPUT_DIR = Path("./generated_audio")
OUTPUT_DIR.mkdir(exist_ok=True)

# MPS 호환 패치
if DEVICE == "mps":
    map_location = torch.device(DEVICE)
    torch_load_original = torch.load
    def patched_torch_load(*args, **kwargs):
        if 'map_location' not in kwargs:
            kwargs['map_location'] = map_location
        return torch_load_original(*args, **kwargs)
    torch.load = patched_torch_load

# 모델 사전 로드 (서버 시작 시)
print(f"Loading models on {DEVICE}...")
from chatterbox.tts import ChatterboxTTS
from chatterbox.tts_turbo import ChatterboxTurboTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

MODELS = {
    "standard": ChatterboxTTS.from_pretrained(device=DEVICE),
    "turbo": ChatterboxTurboTTS.from_pretrained(device=DEVICE),
    "multilingual": ChatterboxMultilingualTTS.from_pretrained(device=DEVICE),
}
print("Models loaded successfully!")


# ─────────────────────────────────────────────────────────────
# 데이터 모델 정의
# ─────────────────────────────────────────────────────────────

class ModelType(str, Enum):
    STANDARD = "standard"       # 500M, 최고 품질
    TURBO = "turbo"             # 350M, 저지연
    MULTILINGUAL = "multilingual"  # 다국어 지원

class UseCasePreset(str, Enum):
    # 어학 학습용 - 명확한 발음
    LANGUAGE_LEARNING = "language_learning"
    # 안내 음성 - 자연스럽고 친근함
    ANNOUNCEMENT = "announcement"
    # 긴급 알림 - 긴박하고 명확함
    EMERGENCY_ALERT = "emergency_alert"
    # 내레이션 - 감정 표현 풍부
    NARRATION = "narration"
    # 캐릭터 음성 - 개성 있는 표현
    CHARACTER = "character"
    # 기본값
    DEFAULT = "default"

class TTSRequest(BaseModel):
    """TTS 생성 요청"""
    text: str = Field(..., min_length=1, max_length=5000, description="변환할 텍스트")
    language: str = Field(default="en", description="언어 코드 (en, ko, ja, fr, de, ...)")
    model: ModelType = Field(default=ModelType.TURBO, description="사용할 모델")
    use_case: UseCasePreset = Field(default=UseCasePreset.DEFAULT, description="용도 프리셋")

    # 고급 옵션
    voice_reference: Optional[str] = Field(default=None, description="Voice Cloning용 참조 음성 ID")
    exaggeration: Optional[float] = Field(default=None, ge=0.0, le=2.0, description="음성 특성 강조")
    speed: Optional[float] = Field(default=1.0, ge=0.5, le=2.0, description="재생 속도")

class TTSResponse(BaseModel):
    """TTS 생성 응답"""
    success: bool
    audio_id: str
    audio_url: str
    duration_seconds: float
    model_used: str
    preset_applied: str


# ─────────────────────────────────────────────────────────────
# 용도별 프리셋 설정
# ─────────────────────────────────────────────────────────────

PRESET_CONFIGS = {
    UseCasePreset.LANGUAGE_LEARNING: {
        "description": "어학 학습용 - 명확한 발음, 적절한 속도",
        "recommended_model": ModelType.STANDARD,
        "exaggeration": 0.8,
        "cfg_weight": 0.7,
        "speed": 0.9,  # 약간 느리게
    },
    UseCasePreset.ANNOUNCEMENT: {
        "description": "안내 음성 - 자연스럽고 친근함",
        "recommended_model": ModelType.TURBO,
        "exaggeration": 1.0,
        "cfg_weight": 0.5,
        "speed": 1.0,
    },
    UseCasePreset.EMERGENCY_ALERT: {
        "description": "긴급 알림 - 긴박하고 명확함",
        "recommended_model": ModelType.TURBO,
        "exaggeration": 1.3,
        "cfg_weight": 0.8,
        "speed": 1.1,  # 약간 빠르게
        "text_prefix": "[clears throat] ",  # Turbo paralinguistic
    },
    UseCasePreset.NARRATION: {
        "description": "내레이션 - 감정 표현 풍부",
        "recommended_model": ModelType.TURBO,
        "exaggeration": 1.2,
        "cfg_weight": 0.6,
        "speed": 1.0,
    },
    UseCasePreset.CHARACTER: {
        "description": "캐릭터 음성 - 개성 있는 표현",
        "recommended_model": ModelType.STANDARD,
        "exaggeration": 1.5,
        "cfg_weight": 0.4,
        "speed": 1.0,
    },
    UseCasePreset.DEFAULT: {
        "description": "기본값",
        "recommended_model": ModelType.TURBO,
        "exaggeration": 1.0,
        "cfg_weight": 0.5,
        "speed": 1.0,
    },
}


# ─────────────────────────────────────────────────────────────
# 핵심 TTS 생성 함수
# ─────────────────────────────────────────────────────────────

def generate_tts(request: TTSRequest) -> tuple[torch.Tensor, int, str]:
    """TTS 생성 핵심 로직"""

    preset = PRESET_CONFIGS[request.use_case]

    # 모델 선택 (언어에 따라 자동 조정)
    if request.language != "en" and request.model != ModelType.MULTILINGUAL:
        model_key = "multilingual"
    else:
        model_key = request.model.value

    model = MODELS[model_key]

    # 파라미터 설정 (사용자 지정 > 프리셋 > 기본값)
    exaggeration = request.exaggeration or preset.get("exaggeration", 1.0)

    # 텍스트 전처리
    text = request.text
    if request.use_case == UseCasePreset.EMERGENCY_ALERT and model_key == "turbo":
        text = preset.get("text_prefix", "") + text

    # 생성 파라미터
    gen_kwargs = {}

    if request.voice_reference:
        # Voice Cloning 모드
        voice_path = f"./voices/{request.voice_reference}.wav"
        if os.path.exists(voice_path):
            gen_kwargs["audio_prompt_path"] = voice_path
            gen_kwargs["exaggeration"] = exaggeration
            gen_kwargs["cfg_weight"] = preset.get("cfg_weight", 0.5)

    # 다국어 모델인 경우
    if model_key == "multilingual":
        gen_kwargs["language_id"] = request.language

    # 음성 생성
    wav = model.generate(text, **gen_kwargs)

    # 속도 조정
    speed = request.speed or preset.get("speed", 1.0)
    if speed != 1.0:
        wav = adjust_speed(wav, model.sr, speed)

    gc.collect()

    return wav, model.sr, model_key


def adjust_speed(wav: torch.Tensor, sr: int, speed: float) -> torch.Tensor:
    """오디오 속도 조정 (리샘플링 방식)"""
    if speed == 1.0:
        return wav

    # 간단한 리샘플링으로 속도 조정
    new_sr = int(sr * speed)
    resampler = ta.transforms.Resample(new_sr, sr)
    return resampler(wav)


# ─────────────────────────────────────────────────────────────
# API 엔드포인트
# ─────────────────────────────────────────────────────────────

@app.post("/api/v1/tts/generate", response_model=TTSResponse)
async def generate_speech(request: TTSRequest):
    """
    텍스트를 음성으로 변환합니다.

    ## 용도별 프리셋
    - `language_learning`: 어학 학습용 (명확한 발음, 느린 속도)
    - `announcement`: 안내 음성 (자연스러움)
    - `emergency_alert`: 긴급 알림 (긴박함, 빠른 속도)
    - `narration`: 내레이션 (감정 표현)
    - `character`: 캐릭터 음성 (개성 강조)
    """
    try:
        # TTS 생성
        wav, sr, model_used = generate_tts(request)

        # 파일 저장
        audio_id = str(uuid.uuid4())
        output_path = OUTPUT_DIR / f"{audio_id}.wav"
        ta.save(str(output_path), wav, sr)

        # 응답
        duration = wav.shape[1] / sr

        return TTSResponse(
            success=True,
            audio_id=audio_id,
            audio_url=f"/api/v1/tts/audio/{audio_id}",
            duration_seconds=round(duration, 2),
            model_used=model_used,
            preset_applied=request.use_case.value,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/v1/tts/audio/{audio_id}")
async def get_audio(audio_id: str):
    """생성된 오디오 파일 다운로드"""
    audio_path = OUTPUT_DIR / f"{audio_id}.wav"

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")

    return FileResponse(
        path=str(audio_path),
        media_type="audio/wav",
        filename=f"{audio_id}.wav"
    )


@app.get("/api/v1/tts/stream/{audio_id}")
async def stream_audio(audio_id: str):
    """오디오 스트리밍 (대용량 파일용)"""
    audio_path = OUTPUT_DIR / f"{audio_id}.wav"

    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Audio not found")

    def iterfile():
        with open(audio_path, "rb") as f:
            while chunk := f.read(65536):  # 64KB chunks
                yield chunk

    return StreamingResponse(iterfile(), media_type="audio/wav")


@app.get("/api/v1/tts/presets")
async def list_presets():
    """사용 가능한 프리셋 목록"""
    return {
        preset.value: {
            "name": preset.value,
            "description": config["description"],
            "recommended_model": config["recommended_model"].value,
        }
        for preset, config in PRESET_CONFIGS.items()
    }


@app.get("/api/v1/tts/languages")
async def list_languages():
    """지원 언어 목록"""
    return {
        "languages": [
            {"code": "en", "name": "English", "models": ["standard", "turbo", "multilingual"]},
            {"code": "ko", "name": "한국어", "models": ["multilingual"]},
            {"code": "ja", "name": "日本語", "models": ["multilingual"]},
            {"code": "zh", "name": "中文", "models": ["multilingual"]},
            {"code": "fr", "name": "Français", "models": ["multilingual"]},
            {"code": "de", "name": "Deutsch", "models": ["multilingual"]},
            {"code": "es", "name": "Español", "models": ["multilingual"]},
            {"code": "it", "name": "Italiano", "models": ["multilingual"]},
            {"code": "pt", "name": "Português", "models": ["multilingual"]},
            {"code": "ru", "name": "Русский", "models": ["multilingual"]},
        ]
    }


@app.get("/health")
async def health_check():
    """서버 상태 확인"""
    return {
        "status": "healthy",
        "device": DEVICE,
        "models_loaded": list(MODELS.keys()),
    }


# ─────────────────────────────────────────────────────────────
# 서버 실행
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

### 서버 실행

```bash
# 의존성 설치
pip install fastapi uvicorn python-multipart

# 서버 실행
python tts_server.py

# 또는 프로덕션 모드
uvicorn tts_server:app --host 0.0.0.0 --port 8000 --workers 4
```

### API 사용 예시

#### cURL

```bash
# 기본 TTS 생성
curl -X POST "http://localhost:8000/api/v1/tts/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Welcome to our service!",
    "language": "en",
    "model": "turbo",
    "use_case": "announcement"
  }'

# 어학 학습용 (명확한 발음)
curl -X POST "http://localhost:8000/api/v1/tts/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog.",
    "language": "en",
    "model": "standard",
    "use_case": "language_learning"
  }'

# 긴급 알림 (긴박함)
curl -X POST "http://localhost:8000/api/v1/tts/generate" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Warning! Please evacuate the building immediately.",
    "language": "en",
    "model": "turbo",
    "use_case": "emergency_alert"
  }'
```

#### JavaScript (Frontend)

```javascript
// TTS API 클라이언트
class ChatterboxTTS {
  constructor(baseUrl = 'http://localhost:8000') {
    this.baseUrl = baseUrl;
  }

  async generate(options) {
    const response = await fetch(`${this.baseUrl}/api/v1/tts/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: options.text,
        language: options.language || 'en',
        model: options.model || 'turbo',
        use_case: options.useCase || 'default',
        speed: options.speed,
        exaggeration: options.exaggeration,
      }),
    });

    if (!response.ok) {
      throw new Error(`TTS Error: ${response.statusText}`);
    }

    return response.json();
  }

  getAudioUrl(audioId) {
    return `${this.baseUrl}/api/v1/tts/audio/${audioId}`;
  }

  // 편의 메서드
  async speak(text, useCase = 'default') {
    const result = await this.generate({ text, useCase });
    const audio = new Audio(this.getAudioUrl(result.audio_id));
    audio.play();
    return result;
  }
}

// 사용 예시
const tts = new ChatterboxTTS();

// 어학 학습용
await tts.speak("Hello, how are you today?", "language_learning");

// 긴급 알림
await tts.speak("Fire alarm activated!", "emergency_alert");

// 안내 방송
await tts.speak("Your order is ready for pickup.", "announcement");
```

#### Python 클라이언트

```python
import requests

class ChatterboxClient:
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url

    def generate(
        self,
        text: str,
        language: str = "en",
        model: str = "turbo",
        use_case: str = "default",
        **kwargs
    ) -> dict:
        """TTS 생성 요청"""
        response = requests.post(
            f"{self.base_url}/api/v1/tts/generate",
            json={
                "text": text,
                "language": language,
                "model": model,
                "use_case": use_case,
                **kwargs
            }
        )
        response.raise_for_status()
        return response.json()

    def download_audio(self, audio_id: str, output_path: str):
        """생성된 오디오 다운로드"""
        response = requests.get(
            f"{self.base_url}/api/v1/tts/audio/{audio_id}",
            stream=True
        )
        response.raise_for_status()

        with open(output_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

# 사용 예시
client = ChatterboxClient()

# 어학 학습 콘텐츠 생성
result = client.generate(
    text="Repeat after me: The weather is beautiful today.",
    language="en",
    use_case="language_learning"
)
client.download_audio(result["audio_id"], "lesson.wav")

# 다국어 안내 방송
for lang in ["en", "ko", "ja"]:
    result = client.generate(
        text="Welcome to our store." if lang == "en" else "환영합니다." if lang == "ko" else "いらっしゃいませ。",
        language=lang,
        use_case="announcement"
    )
    client.download_audio(result["audio_id"], f"welcome_{lang}.wav")
```

### 용도별 최적 설정 가이드

| 용도 | 프리셋 | 모델 | 특징 |
|------|--------|------|------|
| **어학 학습** | `language_learning` | standard | 명확한 발음, 0.9x 속도 |
| **안내 음성** | `announcement` | turbo | 자연스러움, 빠른 응답 |
| **긴급 알림** | `emergency_alert` | turbo | 1.1x 속도, 긴박함 |
| **내레이션** | `narration` | turbo | 감정 표현, paralinguistic |
| **캐릭터** | `character` | standard | 강한 개성, Voice Cloning |

### Docker 배포

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# 시스템 의존성
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Python 의존성
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드
COPY . .

# 모델 캐시 디렉토리
ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p $HF_HOME

EXPOSE 8000

CMD ["uvicorn", "tts_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker-compose.yml
version: '3.8'
services:
  tts-api:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./generated_audio:/app/generated_audio
      - ./voices:/app/voices
      - huggingface_cache:/app/.cache/huggingface
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  huggingface_cache:
```

---

## 결론

Chatterbox TTS는 **무료**로 **로컬**에서 실행되는 고품질 TTS 시스템입니다.

### 핵심 포인트

| 장점 | 설명 |
|------|------|
| 💰 무료 | API 비용 없음, 무제한 사용 |
| 🔒 프라이버시 | 데이터가 외부로 전송되지 않음 |
| 🎯 고품질 | 자연스러운 발음과 억양 |
| 🌍 다국어 | 23개 언어 지원 |
| 🎭 Voice Cloning | 원하는 음성으로 복제 가능 |
| ⚡ 유연성 | 3가지 모델로 용도별 최적화 |

### 권장 사용 사례

- ✅ 영어 학습 오디오북
- ✅ 팟캐스트/내레이션
- ✅ 게임/앱 캐릭터 음성
- ✅ 접근성 도구
- ✅ 프로토타이핑

---

*이 문서는 MacBook Pro M4 24GB 환경에서 테스트된 내용을 기반으로 작성되었습니다.*
