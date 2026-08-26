# gemini-watermark-cleaner-local

Gemini로 생성된 이미지에 표시되는 눈에 보이는 워터마크(스파클 로고)를 찾아서
지우고, 그 영역을 주변 배경과 자연스럽게 복원하는 macOS(Apple Silicon)용
완전 로컬 배치 처리 도구입니다.

**외부 서버 업로드 없음 · Google API 사용 없음 · 유료 API 사용 없음 · 클라우드 처리 없음.**
모든 이미지 처리는 이 컴퓨터 안에서만 일어납니다. 유일한 네트워크 접속은
최초 실행 시 pip 패키지와 LaMa 모델 가중치를 한 번 내려받는 것뿐이며 (다른
파이썬 라이브러리를 설치할 때와 동일), 이후에는 완전히 오프라인으로 동작합니다.

## 사용 방법

1. `input/` 폴더에 처리할 이미지를 원하는 만큼 넣습니다 (PNG, JPG, JPEG, WEBP).
2. 아래 중 하나로 실행합니다.
   - Finder에서 `start.command` 더블클릭
   - 터미널에서 `./start.command`
   - 이미 가상환경이 준비된 경우: `python main.py`
3. `output/` 폴더에서 `원본이름_clean.확장자` 형태로 결과를 확인합니다.
4. 워터마크를 찾지 못했거나 복원에 실패한 이미지는 원본 그대로
   `failed/` 폴더에 **복사**됩니다 (원본은 삭제/수정되지 않습니다).

최초 실행 시 `start.command`가 가상환경(`.venv`)을 만들고 `requirements.txt`를
설치합니다. 이후 실행부터는 설치 과정 없이 바로 처리가 시작됩니다.

### Watch Mode (자동 감시)

```
./start.command --watch
```

프로그램이 실행되어 있는 동안 `input/`에 새 이미지가 추가되면 자동으로
감지해 처리합니다 (기본 2초 간격 폴링). 이미 처리한 파일(파일명 + 내용 해시로
판별)은 다시 처리하지 않으므로, 켜둔 채로 계속 이미지를 넣어도 안전합니다.
운영체제 파일 이벤트(FSEvents) 대신 폴링 방식을 쓴 이유는 추가 의존성 없이
macOS에서 더 안정적으로 동작하기 때문입니다.

## 처리 결과 출력 예시

```
========================================
총 20장
성공 18장
실패 2장
========================================
```

## 폴더 구조

```
gemini-watermark-cleaner-local/
├── input/      # 처리할 원본 이미지를 넣는 곳
├── output/     # 워터마크가 제거된 결과 (원본이름_clean.ext)
├── failed/     # 탐지/복원 실패 시 원본 복사본
├── models/     # (선택) watermark_template.png — 직접 채취한 워터마크 크롭
├── src/        # 탐지·복원·상태관리 파이프라인 코드
├── main.py     # CLI 진입점
├── start.command
└── requirements.txt
```

## 워터마크 탐지 방식

Gemini 워터마크는 이미지 크기에 비례한 % 가 아니라, 출력 해상도에 따라
**고정된 픽셀 크기 + 여백**으로 우측 하단에 찍힙니다.

| 조건 | 크기 | 여백(margin) |
|---|---|---|
| 가로·세로 모두 1024px 초과 | 96×96px | 64px |
| 가로 또는 세로가 1024px 이하 | 48×48px | 32px |

이 값은 공개된 워터마크 알파 블렌딩 역공학 분석 결과를 참고했습니다
(출처는 아래 "Attribution & Sources" 참고, 코드는 그대로 가져오지 않았습니다).

탐지 로직(`src/detector.py`)은 이 위치를 기준으로 ±6px 범위를 탐색하며,
그레이스케일 그래디언트(Sobel) 기반 템플릿 매칭으로 신뢰도 점수를 계산합니다.
점수가 임계값(기본 0.28) 미만이면 "탐지 실패"로 보고 `failed/`로 보냅니다.

기본 템플릿은 근사치로 그린 스파클 모양입니다. **정확도를 높이려면**
본인이 가진 Gemini 이미지 중 하나에서 워터마크 부분만 정사각형으로 크롭해
`models/watermark_template.png`로 저장해두세요. (본인 로컬 파일을 매칭용
템플릿으로만 사용하며, 배포되거나 외부로 전송되지 않습니다.)

## 복원(인페인팅) 방식

탐지된 영역(+여유 패딩)만 마스킹해 **LaMa** 모델로 복원합니다. 이미지 전체를
자르거나 리사이즈하지 않으며, 원본 해상도·비율이 그대로 유지됩니다.

- Apple Silicon에서는 우선 MPS(GPU) 가속을 시도하고, LaMa의 푸리에 컨볼루션이
  MPS에서 지원되지 않는 연산으로 실패하면 자동으로 CPU로 폴백합니다.
- 사용 라이브러리: [`simple-lama-inpainting`](https://github.com/enesmsahin/simple-lama-inpainting)
  (Apache-2.0) — 최초 실행 시 LaMa 체크포인트(`big-lama.pt`)를 한 번
  내려받아 로컬에 캐시합니다. 이후에는 인터넷 연결 없이 동작합니다.

## Attribution & Sources

이 프로젝트는 아래 오픈소스를 **의존성(pip 패키지)으로만 사용**하며, 코드를
그대로 복사하지 않았습니다. 각 라이선스 조건(고지 유지, 저작권 표시)을
준수합니다.

- **LaMa** — *Resolution-robust Large Mask Inpainting with Fourier Convolutions*
  (Suvorov et al., WACV 2022). https://github.com/advimman/lama — Apache License 2.0.
  실제 복원을 수행하는 딥러닝 모델입니다.
- **simple-lama-inpainting** by enesmsahin. https://github.com/enesmsahin/simple-lama-inpainting
  — Apache License 2.0. LaMa 체크포인트를 손쉽게 로드/추론하는 파이썬 래퍼로
  사용했습니다.

워터마크의 위치/크기 규칙(위 표)은 다음 공개 분석 글을 참고해 검증했습니다
(코드는 가져오지 않고, 기하학적 사실만 참고):

- *Removing Gemini AI Watermarks: A Deep Dive into Reverse Alpha Blending* —
  https://antigravityide.org/blog/removing-gemini-ai-watermarks-reverse-alpha-blending/

## 제한 사항

- 워터마크 탐지는 "우측 하단, 알려진 고정 크기" 라는 가정에 기반합니다.
  Gemini가 워터마크 위치/디자인을 바꾸면 `src/detector.py`의 상수를
  업데이트해야 할 수 있습니다.
- 기본 템플릿은 실제 Gemini 워터마크의 정밀한 복제가 아닌 근사치이므로,
  탐지 신뢰도를 높이려면 `models/watermark_template.png`에 실제 크롭을
  넣는 것을 권장합니다.
- LaMa는 "그럴듯한 복원"을 생성하는 생성 모델이므로, 배경이 매우 복잡하거나
  세밀한 패턴일 경우 완벽하게 자연스럽지 않을 수 있습니다.
