# audio-meta
음악 데이터의 메타 정보를 추출하기 위한 AI 모델을 생성하고, 학습된 모델과 메타 데이터를 압축하여 [TorchServe](https://github.com/pytorch/serve)를 이용하기 위해 저장한다


## Environment Setup
python3 (python version: 3.8)을 사용했으며 필요한 requiremets는 다음 명령으로 설치할 수 있다

```
pip install -r requirements.txt
```

## Create a Model
모델 생성 과정은 **main.py**에서 컨트롤되며 기본적인 실행 명령은 다음과 같다

```
python3 main.py -v "versions" -m "mode" -p "process"
```

### version
프로그램 실행을 위한 버전은 다음과 같이 구체적으로 명시해줘야 한다

데이터 버전, 전처리 버전, 특징 추출 버전, 모델 버전의 4가지를 점(.)을 이용하여 분리해서 표현한다

database = 'db', pre_processing = 'pp', transform = 'tr', extractor = 'ex'의 약어로 명시해준다

> X.X.X.X (database_version.pre_processing_version.transform_version.extractor_version)
> 
> e.g. -v db1.pp0.tr0.ex1

### mode
프로세스 중 실행을 원하는 부분을 설정해줘야 한다

데이터 생성(db_audio_gen), 전처리 수행(preprocessing), 특징 추출(db_spectrum_gen), 모델 생성 및 추출(extraction) 중 원하는 프로세스를 명시한다

-m all을 이용하여 한번에 전체 프로세스를 수행할 수 있다

> e.g. -m db_audio_gen

### process
원하는 모델과 관련된 프로세스를 명시해줘야 한다

'music_detection', 'audio_meta', 'audio_analysis' 중 한가지를 선택한다

> e.g. -p audio_meta


## Create .MAR file

학습된 AI 모델을 [TorchServe](https://github.com/pytorch/serve)로 서빙하기 위해 모델과 관련된 메타 데이터를 압축하여 .MAR 파일을 생성한다

> ./make_mar.sh


### torch-model-archiver
모델과 메타 데이터를 압축 하기 위해 serve에서 제공하는 라이브러리인 torch-model-archiver를 사용한다

> e.g. torch-model-archiver --model-name audio_meta --version 0.1 --serialized-file assets/network/regression/db1.pp0.tr0.ex1/final_epoch.pth --export-path ./ --extra-file extra_files --handler infer_handler.py

```
torch-model-archiver -h
usage: torch-model-archiver [-h] --model-name MODEL_NAME  --version MODEL_VERSION_NUMBER
                      --model-file MODEL_FILE_PATH --serialized-file MODEL_SERIALIZED_PATH
                      --handler HANDLER [--runtime {python,python3}]
                      [--export-path EXPORT_PATH] [-f] [--requirements-file] [--config-file]
```

### inference handler
모델 서빙에서 inference에 관련된 프로세스를 관리하기 위한 모듈

AI 모델과 메타 데이터를 이용하여 실제 서비스 서버에서 handler의 프로세스 대로 값을 반환한다

```
# ModelHandler defines a custom model handler.
from ts.torch_handler.base_handler import BaseHandler
    class ModelHandler(BaseHandler):
    ...
```
