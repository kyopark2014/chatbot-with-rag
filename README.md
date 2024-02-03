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

[Google Cloud Vertex AI](https://python.langchain.com/docs/integrations/llms/google_vertex_ai_palm)에 따라서 아래와 같이 langchain-google-vertexai을 설치합니다.

```text
pip install --upgrade --quiet  langchain-core langchain-google-vertexai
```

## Reference

[Bard Chatbot](https://bard.google.com/chat)

[Getting Started with LangChain 🦜️🔗 + Vertex AI PaLM API](https://github.com/GoogleCloudPlatform/generative-ai/blob/main/language/orchestration/langchain/intro_langchain_palm_api.ipynb)
