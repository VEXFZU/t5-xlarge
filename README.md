# T5-XL 모델을 사용한 한국어-점자 번역 모델 훈련

본 레포지터리는 T5-XLarge 모델(3B)을 사용한 한국어-점자 번역 모델 훈련을 위한 코드를 담고 있습니다.

## 환경 설정

### 요구 사양
- Python 3.10+
- Cuda 12.4+
- PyTorch 2.11+
- Transformers 4.45.2+
- 40GB+ GPU 권장

필요 라이브러리 설치:
```
pip install -r requirements.txt
```

## 간편한 실행
### 훈련 스크립트 실행
1. `dataset/` 폴더에 학습 데이터를 준비합니다.
2. `train.sh` 파일에 훈련 파라미터를 설정합니다.
3. 훈련을 실행합니다.
```
chmod +x train.sh
./train.sh
```

### 평가 스크립트 실행
1. 평가하고자 하는 점역 모델을 huggingface hub에 업로드 하거나, local에 저장합니다.
2. `benchmark.sh` 파일에 huggingface hub에 업로드 한 모델 이름 또는 local에 저장한 모델 경로를 설정합니다.
3. 사용하고자 하는 모델의 특정 버전이 있을 경우, `benchmark.sh` 파일의 `revision` 파라미터를 설정합니다. 
4. 모델 평가에 사용할 데이터셋을 `benchmark.sh` 파일의 `benchmark_path` 파라미터에 설정합니다.
5. 평가를 실행합니다.
```
chmod +x benchmark.sh
./benchmark.sh
```

## Project Structure
```
t5-xlarge/
├── dataset/             # 학습 데이터
├── main.py              # 훈련 스크립트
├── data.py              # 데이터 처리 스크립트
├── train.sh             # 훈련 실행 스크립트
├── benchmark.py         # 평가 스크립트
├── benchmark.sh         # 평가 실행 스크립트
├── special_braille.txt  # special token으로 추가할 점자 목록
├── requirements.txt     # 필요 라이브러리
└── README.md
```

### 사용 데이터
본 프로젝트에서는 국립 국어원에서 제공하는 묵자-점자 병렬 말뭉치 2023을 사용합니다. 이 데이터셋은 한국어 텍스트와 그에 대응하는 점자 번역을 포함하고 있으며, 모델 훈련 및 평가에 사용됩니다.

- **데이터 형식**: JSON
- **데이터 크기**: 약 125,000쌍 한국-점자 병렬 문장
- **사용 권한**: 연구 및 기술 개발용으로 승인된 목적에 한하여 사용 가능
- **출처**: [국립 국어원](https://kli.korean.go.kr/)


### 베이스 모델
본 프로젝트는 huggingface hub에 업로드 된`sangmin6600/t5-v1_1-xl-ko` 를 fine-tuning한 모델을 사용하였습니다.

fine-tuning을 진행할 베이스 모델 변경을 원할 경우, train.sh 파일의 `--model_name`과 `--tokenizer_name` 파라미터를 수정하세요.
예시:
```
MODEL_NAME=${MODEL_NAME:-"사용할 모델 이름"}
TOKENIZER_NAME=${TOKENIZER_NAME:-"사용할 토크나이저 이름"}
```

## Evaluation

Run evaluation on a test set:
```
python src/evaluate.py \
    --model_path models/checkpoint-best \
    --test_file data/test.json
```

## Contact
For questions or issues, please open a GitHub issue.