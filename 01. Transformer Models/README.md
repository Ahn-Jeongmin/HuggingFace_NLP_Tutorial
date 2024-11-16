## 1\. Natural Language Processing

#### NLP란 무엇인가?

정의 : 인간 언어와 관련된 모든 것을 이해하는 데 초점을 맞춘 언어학 및 머신러닝 분야

목적 : 개별 단어를 이해하는 것뿐만 아니라 해당 단어의 맥락을 이해하는 것

#### 자연어처리 작업 예시 

| task | description |
| --- | --- |
| 전체 문장 분류 | 리뷰의 감정 파악, 이메일이 스팸인지 감지, 문장이 문법적으로 올바른지 또는 두 문장이 논리적으로 관련되어 있는지 여부 확인 |
| 문장의 각 단어 분류 | 문장의 문법적 구성 요소 (품사) 또는 명명된 개체 (사람, 조직, 위치) 식별 |
| 텍스트 콘텐츠 생성 | 자동 생성된 텍스트로 프롬프트 완성, 가려진 단어로 텍스트의 빈칸 채우기 |
| 텍스트에서 답변 추출 | 질문과 맥락이 주어지면 맥락에 제공된 정보를 기반으로 질문에 대한 답변을 추출 |
| 입력 텍스트에서 새로운 문장 생성 | 텍스트를 다른 언어로 번역, 텍스트 요약 |

하지만 자연어처리는 서면 텍스트에만 국한되지 않고, 음성 인식 등의 음성 샘플 대본 생성 등의 다양한 분야에서 활용된다. 또 이미지 생성 시 설명을 생성하는 것에도 활용되기도 한다.

## 2\. Transformers, what can they do?

Transformers 라이브러리에서 가장 기본적인 객체는 pipeline() 함수이다.

이 함수는 모델을 필수적인 전처리 및 후처리 과정들과 연결해주고, 사용자가 바로 모델을 활용해 가치있는 답변을 얻을 수 있도록 허용한다. 한 번에 여러 문장을 전달 가능하다.

기본적으로 이 파이프라인은 영어로 작성된 입력에 대해 감정분석이 진행되도록 finetuned된 모델을 선택한다.

classifier 객체를 만들면 모델이 다운로드되고 캐시된다.

데이터 파이프라인을 거치는 과정에서는 다음 3가지 단계가 포함된다.

1\. 텍스트가 모델이 이해할 수 있는 형식으로 사전 처리 -> 패딩, 토큰, 불용어, lowercase 등의 전처리 진행  
2\. 전처리된 입력이 모델에게 전달된다   
3\. 모델의 예측은 이후 진행되어 output이 반환된다.


    
현재 활용이 가능한 pipeline은 다음과 같다 :

-   feature-extraction (get the vector representation of a text)
-   fill-mask
-   ner (named entity recognition)
-   question-answering
-   sentiment-analysis
-   summarization
-   text-generation
-   translation
-   zero-shot-classification

해당 파이프라인 객체 생성 시, model 변수를 지정하지 않으면 기본 모델이 사용되지만, hub에서 특정 모델을 선택하여 특정 작업의 파이프라인에서 사용이 가능하다.

```
generator = pipeline("text-generation", model="distilgpt2")
```

## 3\. How do Transformers work?

모든 Transformer 모델(GPT, BERT, BART, T5 등)은 언어 모델 로 훈련되었다 .

대부분의 언어 모델은 self-supervised 방식으로 대량의 원시 텍스트에 대해 훈련, 모델의 입력에서 목표가 자동으로 계산

즉, 사람이 데이터에 레이블을 지정할 필요가 없음

이 유형의 모델은 학습된 언어에 대한 통계적 이해를 개발, 이 때문에 일반적으로 사전 학습된 모델은 **전이 학습** 이라는 프로세스를 거칩니다 . ( transfer learning)

![image](https://github.com/user-attachments/assets/945bbea0-dd07-483e-a642-f82bb91079b0)



작업의 한 예는 이전 n 개 단어를 읽고 문장의 **다음 단어를 예측**하는 것

이를 causal language modeling  이라고 하는데 , 출력은 과거와 현재 입력에 따라 달라지지만 미래 입력에는 달라지지 않기 때문입니다.




또 다른 예는 masked language modeling  으로 , 이는 모델이 문장에서 **마스크된 단어를 예측**하는 것

![image](https://github.com/user-attachments/assets/b1fe60ae-cdda-4dff-b49a-4daae4f74be9)

**General Architechture**

![image](https://github.com/user-attachments/assets/7252ad14-7ec5-4f91-859a-a1b724ad18b4)

이 모델은 주로 두 개의 블록으로 구성된다.

-   인코더(왼쪽) : 인코더는 입력을 받고 그것의 **표현(특징)을 구축**. 즉, 모델은 입력으로부터 이해를 얻도록 최적화.
-   디코더(오른쪽) : 디코더는 인코더의 표현(특징)과 다른 입력을 함께 사용하여 대상 시퀀스를 생성. 즉, 모델은 출력을 생성하기 위해 최적화.

각 부분은 작업에 따라 독립적으로 사용할 수 있습니다.

-   Encoder-only models  : 문장 분류 및 명명된 엔터티 인식과 같이 입력에 대한 이해가 필요한 작업에 적합합니다.
-   Decoder-only models  : 텍스트 생성과 같은 생성 작업에 적합합니다.
-   Encoder-decoder models or sequence-to-sequence models  : 번역이나 요약 등 입력이 필요한 생성 작업에 적합합니다.

**Attention Layer**

Transformer 모델의 주요 특징은 어텐션 레이어 라고 하는 특수 레이어로 구축된다는 것

이 레이어가 각 단어의 표현을 처리할 때 전달한 문장의 특정 단어에 특별히 주의를 기울이고(그리고 다른 단어는 거의 무시하도록) 모델에 지시한다

영어에서 프랑스어로 텍스트를 번역하는 작업

\=> 입력 "You like this course"가 주어졌을 때 번역 모델은 "like"라는 단어에 대한 적절한 번역을 얻기 위해 인접한 단어 "You"에도 주의를 기울여야 함. 프랑스어에서 동사 "like"는 주어에 따라 다르게 활용되기 때문.

\=> 그러나 나머지 문장은 해당 단어의 번역에 유용하지 않음. 같은 맥락에서 "this"를 번역할 때 모델은 "course"라는 단어에도 주의를 기울여야 함. "this"는 연관된 명사가 남성형인지 여성형인지에 따라 다르게 번역되기 때문.

\=> 즉, 문장의 다른 단어는 "course"의 번역에 중요하지 않다. 더 복잡한 문장(및 더 복잡한 문법 규칙)의 경우 모델은 각 단어를 적절하게 번역하기 위해 문장에서 더 멀리 나타날 수 있는 단어에 특별히 주의를 기울여야 함.

단어 자체는 의미를 가지고 있지만, 그 의미는 맥락에 의해 큰 영향을 받는데, 맥락이란 공부하는 단어 앞이나 뒤에 있는 다른 단어(들)일 수 있다는 전제 하에 동작하는 아키텍처임

**The original architecture**

![image](https://github.com/user-attachments/assets/9abe5c75-4b8f-4e61-97e3-419a953b9587)


Transformer 아키텍처는 원래 번역을 위해 설계 -> 훈련하는 동안 인코더는 특정 언어로 입력(문장)을 받는 반면 디코더는 원하는 대상 언어로 동일한 문장을 받게 되는 식으로 훈련

인코더에서 어텐션 레이어는 문장의 모든 단어를 사용할 수 있다(주어진 단어의 번역은 문장에서 그 뒤와 앞에 있는 것에 따라 달라질 수 있기 때문).

그러나 디코더는 순차적으로 작동하며 이미 번역한 문장의 단어에만 주의 가능(즉, 현재 생성 중인 단어의 앞 단어에만 주의를 기울일 수 있다). 예를 들어 번역된 대상의 처음 세 단어를 예측한 경우 이를 디코더에 제공하고 디코더는 인코더의 모든 입력을 사용하여 네 번째 단어를 예측하려고 시도.

훈련 중에 속도를 높이기 위해(모델이 타겟 문장에 접근할 수 있을 때) 디코더에 전체 타겟을 제공하지만, 미래의 단어를 사용할 수 없다, 예를 들어, 네 번째 단어를 예측하려고 할 때, 어텐션 레이어는 위치 1~3의 단어에만 접근할 수 있습니다.

디코더 블록의 첫 번째 어텐션 레이어는 디코더에 대한 모든 (과거) 입력에 주의를 기울이지만 두 번째 어텐션 레이어는 인코더의 출력을 사용. 따라서 전체 입력 문장에 액세스하여 현재 단어를 가장 잘 예측 가능.

이는 다른 언어가 단어를 다른 순서로 배치하는 문법 규칙을 가질 수 있거나 문장의 후반에 제공된 일부 맥락이 주어진 단어의 최상의 번역을 결정하는 데 도움이 될 수 있으므로 매우 유용.

어텐션 마스크는 인코더/디코더에서도 모델이 일부 특수 단어에 주의를 기울이지 않도록 하는 데 사용될 수 있는데, 예를 들어, 문장을 일괄 처리할 때 모든 입력을 같은 길이로 만드는 데 사용되는 특수 패딩 단어

| Encoder Model | Decoder Model | Seq2Seq Model |
| --- | --- | --- |
| 주어진 문장을 어떻게든 손상시키고(예를 들어, 문장에서 무작위 단어를 마스크함) 모델에 초기 문장을 찾거나 재구성하는 작업을 맡기는 역할을 수행      인코더 모델은 문장 분류, 명명된 엔터티 인식(더 일반적으로 단어 분류), 추출적 질의 응답 등 전체 문장에 대한 이해가 필요한 작업에 가장 적합하 |  각 단계에서 주어진 단어에 대해 어텐션 레이어는 문장에서 해당 단어 앞에 위치한 단어에만 액세스할 수 있습니다. 이러한 모델은 종종 자기 회귀 모델 이라고 함   디코더 모델의 사전 학습은 일반적으로 문장의 다음 단어를 예측하는 데 중점을 둠      이러한 모델은 텍스트 생성과 관련된 작업에 가장 적합 | Transformer 아키텍처의 두 부분을 모두 사용하며 각 단계에서 인코더의 어텐션 레이어는 초기 문장의 모든 단어에 액세스할 수 있는 반면, 디코더의 어텐션 레이어는 입력에서 주어진 단어 앞에 위치한 단어에만 액세스 가능      시퀀스-투-시퀀스 모델은 요약, 번역 또는 생성적 질의응답과 같이 주어진 입력에 따라 새로운 문장을 생성하는 작업에 가장 적합합니다. |
|[ALBERT](https://huggingface.co/docs/transformers/model_doc/albert), [BERT](https://huggingface.co/docs/transformers/model_doc/bert) ,[DistilBERT](https://huggingface.co/docs/transformers/model_doc/distilbert) ,[ELECTRA](https://huggingface.co/docs/transformers/model_doc/electra) ,[RoBERTa](https://huggingface.co/docs/transformers/model_doc/roberta)   |   [CTRL](https://huggingface.co/transformers/model_doc/ctrl), [GPT](https://huggingface.co/docs/transformers/model_doc/openai-gpt) ,[GPT-2](https://huggingface.co/transformers/model_doc/gpt2) ,[Transformer XL](https://huggingface.co/transformers/model_doc/transfo-xl)   |   [BART](https://huggingface.co/transformers/model_doc/bart) ,[mBART](https://huggingface.co/transformers/model_doc/mbart) , [Marian](https://huggingface.co/transformers/model_doc/marian) , [T5](https://huggingface.co/transformers/model_doc/t5)   |



<br>
<br>

✅  전이학습

![image](https://github.com/user-attachments/assets/31949fad-dfe9-4e3b-8271-9d4e38361b23)

Transferring the knowledge of a pretrained model to a new model by initializing the second model with the first model's weights.

즉, 기존 모델의 지식을 새로운 작업에 적용함으로써 효율성을 높이고 더 빠른 학습과 성능 향상을 이끌어내는 중요한 방법

새로운 모델을 훈련시킬 때 기존의 pretrained model의 가중치와 같은 정보를 초기값으로 설정하고 다른 데이터셋으로 학습을 진행하는 것
