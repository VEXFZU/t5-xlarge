# T5-XL 모델을 사용한 한국어-점자 번역 모델 훈련

본 레포지터리는 T5-XLarge 모델(3B)을 사용한 한국어-점자 번역 모델 훈련을 위한 코드를 담고 있습니다.

## 환경 설정

### 요구 사양
- Python 3.10+
- PyTorch 2.11+
- Cuda 12.4+
- Transformers 4.45.2+
- 40GB+ GPU recommended

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
1.
2.
3.
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
├── requierments.txt     # 필요 라이브러리
└── README.md
```

## 모델 훈련
본 훈련 스크립트는 다음을 지원합니다.
- Mixed precision
- Checkpoint 저장
- Evaluation 중 WER 및 CER 메트릭 계산 및 로깅
- Wandb 로깅

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

## License
MIT

## Contact
For questions or issues, please open a GitHub issue.