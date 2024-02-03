# Gemini를 이용한 RAG 구성 방법

여기서는 Google의 Gemini를 AWS와 연동하여 RAG를 구현하는 방법에 대해 설명합니다.

## Gemini API 설정

[Key를 비롯한 API 설정](https://yunwoong.tistory.com/297)에 따라 Gemini API를 활성화하고 key 값을 보관합니다.

필요한 라이브러리는 아래와 같이 설치합니다.

```text
pip install "google-cloud-aiplatform>=1.38"
pip install pillow
pip install matplotlib
```


## Reference

[Bard Chatbot](https://bard.google.com/chat)
