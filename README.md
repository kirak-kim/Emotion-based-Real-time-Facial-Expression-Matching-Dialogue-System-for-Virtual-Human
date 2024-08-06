# 감정에 기반한 가상인간의 대화 및 표정 실시간 생성 시스템 구현

*김기락, 연희연, 은태영 and 정문열. (2022). 감정에 기반한 가상인간의 대화 및 표정 실시간 생성 시스템 구현. 한국컴퓨터그래픽스학회논문지, 28(3), 23-29.*
[Paper](https://www.kci.go.kr/kciportal/ci/sereArticleSearch/ciSereArtiView.kci?sereArticleSearchBean.artiId=ART002862290)

<img width="995" alt="그림1" src="https://user-images.githubusercontent.com/91187839/184279281-9c66f97b-fe58-483f-b121-afacb49cbbac.png">

>**본 코드에는 Cerevoice의 License 파일과 대화 생성 및 감정 분석 모델 파일이 포함되어 있지 않습니다**

## 👨‍💻 대화 모델 돌리기

**1) 필요한 라이브러리 다운 받기** 

```python
pip install -r ./Dialogue modeling/requirements_python.txt
```

**2) 모델 다운 받기**

구글 드라이브 링크 : https://drive.google.com/file/d/1EUoZsSTmzI2KpSkGFmEEze7yIW1-0pG6/view?usp=sharing

위 드라이브 링크에서 다운 받아서 Dialogue modeling 폴더안에 saved_models 폴더 만들고 그 안에 넣어주세요.

**3) 실행하기**

Dialogue modeling 폴더에서 실행해 주세요.

```python
python src/main.py
```

## :studio_microphone: 사용자 입력을 음성으로 받기
**1) 유니티에서 마이크 음성 캡처** 

[Link](https://github.com/kirak-kim/Real-time_Audio_From_Mic)

**2) STT 적용** 
