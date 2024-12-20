## 1\. Behind the Pipeline

![image](https://github.com/user-attachments/assets/e5d6101f-a675-4b12-a1a5-fe3fccbc190c)


Huggingface에서 제공하는 파이프라인은 다음 3가지 단계를 압축하여 제공한다

(1) 데이터 전처리 (2) 모델에 input 값 입력 (3) 데이터 후처리
<br>
<br>

**(1) 데이터 전처리**

다른 NN과 같이 트랜스포머 모델 또한 input 텍스트를 바로 이해할 수 없음 -> 모델이 이해 가능한 형태의 input이 필요

따라서 우리는 tokenizer 을 활용하여 다음 세 가지를 맡긴다.

-   토큰이라고 불리는 형태로 청킹 (단어, 문장부호 등으로 분리)
-   각각의 토큰을 정수 형태의 인덱스로 매핑
-   모델에게 도움이 될 수 있는 추가적인 input을 더함

위 세 가지는 모델이 최초훈련이 될 당시의 조건과 정확하게 일치해야하며, 이는 Huggingface Model Hub에서 확인해야

\-> 이러한 과정을 간단히 하기 위해서는 Huggingface에서 Autotokenizer 클래스를 제공, from\_pretrained() 함수 활용

해당 모델의 체크포인트 이름을 활용하며 데이터를 자동으로 변환하여 캐시하게 된다.
<br>
<br>

Transformer 모델의 경우 tensor 텐서만 입력으로 허용된다

반환하려는 **텐서의 유형 (** **PyTorch, TensorFlow 또는 일반 NumPy ) 을 지정하려면 return\_tensors 인수**를 사용하여 지정하면 된다

```
raw_inputs = [
    "I've been waiting for a HuggingFace course my whole life.",
    "I hate this so much!",
]
inputs = tokenizer(raw_inputs, padding=True, truncation=True, return_tensors="pt")
print(inputs)
```

```
{
    'input_ids': tensor([
        [  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172, 2607,  2026,  2878,  2166,  1012,   102],
        [  101,  1045,  5223,  2023,  2061,  2172,   999,   102,     0,     0,     0,     0,     0,     0,     0,     0]
    ]), 
    'attention_mask': tensor([
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
}
```

하나의 문장이나 문장 목록을 전달할 수 있고, 반환하려는 텐서 유형을 지정할 수 있다

출력 자체는 **두 개의 키 (input\_ids, attention\_mask)** 를 포함하는 사전

input\_ids 는 하나의 문장을 각각 정수로 표현한 것을 반환하는 부분이다. -> 단어 정수화

<br>
<br>

**(2) 모델에 input 값 입력**

자 이제 토큰화 시켰으면 모델에 넣어야 함

토크나이저에서 진행했던 것과 동일하게 사전 학습된 모델을 불러올 수 있음 

**AutoModel 클래스 제공, from\_pretrained() 함수를 활용**해서 모델을 변수에 저장하여 바로 사용이 가능함

```
from transformers import AutoModel

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
model = AutoModel.from_pretrained(checkpoint)
```

동일하게 체크포인트를 다운로드하고, 이를 사용하여 **모델을 변수에 저장하며 인스턴스화**함

이 아키텍처는 기본 Transformer 모듈만 포함

input 입력값이 주어지면 **output으로 feature 특징값이라고도 알려져 있는 hidden states들을 출력**한다.

각각의 모델 입력값에 대해 Transformer 모델의 **"문맥적 이해"를 표현하는 고차원의 벡터를 복구**하는 작업을 수행한다.

<br>

Transformer 모듈의 벡터 출력 (hidden states) 은 일반적으로 매우 크며, 일반적으로 3가지 차원이 존재함

-   **배치 크기** : 한 번에 처리하는 시퀀스 수(예에서는 2개)
-   **시퀀스 길이** : 시퀀스의 숫자형 표현 길이(예시에서는 16)
-   **숨겨진 크기** : 각 모델 입력의 벡터 차원입니다.

마지막 값이 대체로 큰 경우가 많기 때문에 고차원이라고 불림 (작은 모델은 768, 큰 모델에서는 3072까지 도달 가능)

Transformer 모델의 출력은 namedtuples 또는 딕셔너리로 동작

속성(우리가 한 것처럼)이나 키( )로 요소에 액세스할 수 있으며 outputs\["last\_hidden\_state"\], 찾고 있는 것이 정확히 어디에 있는지 알고 있다면 인덱스( outputs\[0\])로 액세스 가능

<br>
<br>

\*\* Model Head

![image](https://github.com/user-attachments/assets/0388b1a2-c721-4716-ab85-f56466d6d793)


모델헤드의 경우 hidden state의 고차원 벡터를 입력으로 받아서 다른 차원으로 매핑하게 됨

일반적으로 여러 개의 선형 layer로 구성됨

Transformer 모델의 출력은 처리를 위해 모델 헤드로 직접 전송됨

위의 이미지의 다이어그램에서 모델은 임베딩 레이어와 그 이후의 레이어로 표현이 되게 되는데, 임베딩 레이어는 토큰화된 입력의 각 입력 ID를 연관된 토큰으로 나타내는 벡터로 변환 (원핫벡터)

그 이후의 레이어는 어텐션 메커니즘을 사용하여 해당 벡터를 조작해 문장의 최종 표현을 생성한다

Transformer에는 다양한 아키텍처가 존재하며, 각각의 아키텍처는 특정 작업을 해결하도록 설계되어있다.

-   \*Model
-   \*ForCausalLM
-   \*ForMaskedLM
-   \*ForMultipleChoice
-   \*ForQuestionAnswering
-   \*ForSequenceClassification
-   \*ForTokenClassification
-   기타
<br>
<br>

**(3) 데이터 후처리**

우리 모델에서 출력으로 얻은 값은 반드시 그 자체로 의미가 있는 것은 아님 -> 모델 출력은 확률이 아니라 로짓 , 즉 모델의 마지막 레이어에서 출력된 원시적이고 정규화되지 않은 점수

이를 확률로 변환하려면 softmax 함수를 거쳐야 함

```
import torch

#소프트맥스 함수에 넣어서 원시적인 값 -> 확률값 으로 변환하는 데이터 매핑을 진행
#torch 라이브러리에서 지원하는 nn.functional.softmax 에 매개변수로 로짓값을 전달

predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
```

```
tensor([[4.0195e-02, 9.5980e-01],
        [9.9946e-01, 5.4418e-04]], grad_fn=<SoftmaxBackward>)
        
# 첫 번째 문장: 부정적: 0.0402, 긍정적: 0.9598
# 두 번째 문장: 부정: 0.9995, 긍정: 0.0005
```

각 위치에 해당하는 라벨을 얻으려면 id2label모델 구성의 속성을 검사하면 됨
<br>
<br>
## 2\. Models

클래스 AutoModel 는 체크포인트에 적합한 모델 아키텍처를 자동으로 추측한 다음 이 아키텍처로 모델을 인스턴스화

하지만 사용하고 싶은 모델 유형을 알고 있다면 해당 아키텍처를 직접 정의하는 클래스를 사용할 수 있음

모델을 로드하는 것은 두 가지 방법이 있을 수 있음

**(1) 기본 구성으로 모델 생성하기** : 모델이 임의의 값으로 초기화된다

```
from transformers import BertConfig, BertModel 

config = BertConfig() 
model = BertModel(config)
```

이 상태에서 모델을 사용할 수는 있으나 학습을 진행해야하는 상태의 모델임

즉, 구조와 아키텍처만 제공하는 버전인 것 -> 따아서 오랜 시간과 데이터를 투자하여 모델을 학습해야함

**(2) 이미 학습된 pretrained 모델을 로드하여 가져오는 것** : from\_pretrained() 메서드를 활용하여 수행 가능

```
from transformers import BertModel

model = BertModel.from_pretrained("bert-base-cased")
```

위의 코드 샘플에서는 BertConfig 를 사용하지 않고 대신 bert-base-cased 식별자를 통해 사전 학습된 모델을 로드함

이 모델은 이제 체크포인트의 모든 가중치로 초기화된 상태임

학습된 작업에 대한 추론에 직접 사용할 수 있으며, 새로운 작업에 대해 파인튜닝도 가능 -> 처음부터가 아닌 사전 학습된 가중치로 학습하면 빠르게 좋은 결과를 얻을 수 있음

모든 모델 식별자는 다음 링크에서 확인 가능 -> [https://huggingface.co/models?other=bert](https://huggingface.co/models?other=bert)
<br>
<br>

**(3) 모델 저장 -> save\_pretrained() 메서드 활용**

config 정보 ( config.json ) 와 모델 파일 ( pytorch\_model.bin )-> 총 두 가지 파일을 컴퓨터의 디렉토리에 저장

config.json 파일 을 살펴보면 모델 아키텍처를 빌드하는 데 필요한 속성 파악 가능

체크포인트가 시작된 위치와 마지막으로 체크포인트를 저장했을 때 사용했던 Transformers 버전과 같은 일부 메타데이터도 포함

pytorch\_model.bin 파일은 state dictionary 으로 알려져 있으며 , 모든 모델의 가중치를 포함, 즉 매개변수에 대한 정보를 저장하는 파일임

config 파일은 모델의 아키텍처를 아는 데 필요한 반면, 모델 가중치는 모델의 매개변수이다
<br>
<br>



## 3\. Tokenizers

NLP 파이프라인에서 토크나이저의 역할은 중요함 -> input 데이터를 모델이 처리할 수 있는 데이터로 변환하는 것

모델은 숫자만 처리할 수 있기 때문에 **토크나이저는 텍스트 입력을 숫자 데이터로 변환**해야함

토크나이저의 목표 : 가장 의미있는 표현과 가장 가능한 작은 표현을 찾는 것

토큰화 알고리즘에는 몇 가지 예시가 있다

**(1-1) 단어 기반 word-based**

일반적으로 몇 가지 규칙만 있으면 사용하기 매우 쉽고 좋은 결과를 보여준다.

![image](https://github.com/user-attachments/assets/f527bd00-1015-4d51-8095-b4d0a1bf9835)


텍스트를 이와 같이 분할하는 방법 -> python의 split() 함수 -> 공백을 기준으로 텍스트 토큰화

구두점에 대한 추가 규칙이 있는 단어 토크나이저도 존재,

이러한 토크나이저를 활용하면 꽤 큰 "어휘집 (vocabularies)"을 얻을 수 있는데, 여기서 어휘집은 우리의 corpus에 있는 독립 토큰의 총 개수에 의해 정의된다.

각 단어에는 0부터 시작해 어휘의 크기까지 올라가는 ID가 할당, 모델은 이 ID를 사용하여 각 단어를 식별

단어 기반 토크나이저로 언어를 완전히 포괄하려면 언어의 각 단어에 대한 식별자가 필요하며, 이를 통해 엄청난 양의 토큰이 생성

예를 들어 영어에는 500,000개가 넘는 단어가 있으므로 각 단어에서 입력 ID로의 맵을 작성하려면 그만큼의 ID를 추적해야 하고, 또한 "dog"와 같은 단어는 "dogs"와 같은 단어와 다르게 표현되며, 모델은 처음에는 "dog"와 "dogs"가 비슷하다는 것을 알 수 없기 때문에 두 단어가 관련이 없다고 식별

또한 우리는 어휘 사전에 없는 단어를 나타내는 사용자 지정 토큰이 필요 -> ”\[UNK\]” or ”<unk>”

토크나이저가 이렇게 토큰을 많이 생산해내는 것은 좋은 징조가 아님

토크나이저가 단어의 합리적인 표현을 찾지 못 하였고 그 과정에서 정보의 loss가 늘어나고 있다는 뜻이기 땜누

어휘를 만들 때의 목표는 토크나이저가 가능한 한 적은 단어를 ”\[UNK\]” or ”<unk>” 토큰으로 처리하는 방식을 채택하는 것

**(1-2) 문자 기반 Character-based**

![image](https://github.com/user-attachments/assets/bcc5811b-feb7-4e09-9c6e-5f8265c4c3a7)


해당 토크나이저는 단어가 아니라 문자 단위로 문장을 토크나이징 -> 다음 2가지 장점이 존재

-   어휘량이 훨씬 적습니다.
-   모든 단어는 문자로 구성될 수 있으므로 어휘에서 벗어난(알 수 없는) 토큰은 훨씬 적습니다.

그러나 해당 접근방식도 완벽하지 않음

표현을 단어가 아닌 문자 기반으로 하기 때문에 직관적으로 보면 의미가 감소되었음

각 문자는 그 자체로 큰 의미가 없지만 단어를 생성함에 따라서 의미를 가지게 되기 때문

**(1-3) 하위 단어 기반 subword tokenization**

![image](https://github.com/user-attachments/assets/7cd25fbb-d49f-4009-8fcb-69df404c0dcf)


하위 단어 토큰화 알고리즘은 자주 사용되는 단어를 더 작은 하위 단어로 나누지 않고 드물게 사용되는 단어를 의미 있는 하위 단어로 분해해야 한다는 원칙에 의존한다.

즉, 자주 사용되는 단어는 나누지 않고 하나의 vocab corpus로 인정

\-> 교착어 토크나이징에 유리

**(2) 로딩 및 저장**

토크나이저를 로딩하는 것은 모델과 동일하게 클래스를 import 한 뒤 from\_pretrained() 메서드를 활용해서 가져오면 됨

-   특정 클래스에서 임포트해서 활용하는 방법

```
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
```

-   디폴트로 임포트라는 경우

```
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
```

**Encoding 인코딩** : 텍스트를 숫자로 변환하는 것

인코딩은 토큰화, 그 다음 입력 ID로의 변환이라는 두 단계 프로세스로 수행됨

첫 번째 토큰화 단계는 텍스트를 단어(subwords)로 분할하는 것, 이를 토큰이라고 함

이 프로세스를 제어 가능한 여러 규칙이 있기 때문에 모델 이름을 사용해서 토크나이저를 인스턴스화 (from\_pretrained() 활용해서 변수에 저장) 하여 모델이 사전학습되었을 때 사용한 것과 동일한 규칙을 사용해서 토크나이징 해야 함

\-> 이 부분이 일치하지 않으면 정확성이 떨어짐

두 번째 단계는 토큰을 숫자로 변환하여 텐서를 빌드하고 모델에 공급하는 것

이를 위해 토크나이저에는 vocabularies가 존재 -> 인스턴스화 시 다운로드됨

Decoding 디코딩 : 어휘 인덱스를 보고 원본 문자열을 retrieve

decode는 인덱스를 토큰으로 다시 변환할 뿐만 아니라 동일한 단어의 일부였던 토큰을 그룹화하여 읽을 수 있는 문장 생성
<br>
<br>
## 4\. Handling Multiple Sequences

**(1) Models expect a batch of inputs**

트랜스포머 모델의 경우 하나의 단일 시퀀스 입력을 기대하는 것이 X

기본 디폴트로 연속적인 일괄 다중 시퀀스 입력을 기대 -> 단일 시퀀스 입력 시 에러

```
tokenized_inputs = tokenizer(sequence, return_tensors="pt")
print(tokenized_inputs["input_ids"])
```

```
tensor([[  101,  1045,  1005,  2310,  2042,  3403,  2005,  1037, 17662, 12172,
          2607,  2026,  2878,  2166,  1012,   102]])
```

자세히 보면 토크나이저가 입력id 몯록을 텐서로 단순히 변환하지 않고 그 위에 차원을 하나 추가함

**(2) 입력패딩**

두 개 이상의 문장을 함께 배치하려고 할 때 문장의 길이가 다를 시 문제가 발생함

텐서는 "직사각형" 모양이어야 입력 id 목록을 텐서로 직접 변환이 가능함

일반적으로 이 문제를 해결하기 위해 입력을 패딩 padding

**(3) 더 긴 시퀀스**

Transformer 모델을 사용하면 모델에 전달할 수 있는 시퀀스 길이에 제한이 있습니다. 대부분의 모델은 최대 512개 또는 1024개 토큰의 시퀀스를 처리하며 더 긴 시퀀스를 처리하라는 요청을 받으면 충돌합니다. 이 문제에 대한 두 가지 해결책이 있습니다.

-   지원되는 시퀀스 길이가 더 긴 모델을 사용하세요.
-   시퀀스를 잘라냅니다.

모델마다 지원되는 시퀀스 길이가 다르고, 일부는 매우 긴 시퀀스를 처리하는 데 특화되어 있습니다. [Longformer](https://huggingface.co/docs/transformers/model_doc/longformer) 가 한 예이고, 또 다른 예는 [LED](https://huggingface.co/docs/transformers/model_doc/led) 입니다 . 매우 긴 시퀀스가 ​​필요한 작업을 진행 중이라면 해당 모델을 살펴보는 것이 좋습니다.

그렇지 않으면 매개변수를 지정하여 시퀀스를 잘라내는 것이 좋습니다
