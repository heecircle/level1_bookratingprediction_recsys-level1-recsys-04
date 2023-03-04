### 1. CatBoost 사용 배경
Baseline code들이 뚜렷한 성과가 안 나오고 있던 중에 GBM(Gradient Boosting Machine)을 사용하자는 의견이 나왔고, 
GBM계열 모델 3가지(XGBoost, LightGBM, CatBoost) 중 가장 나중에 나온 CatBoost를 채택했다.
CatBoost는 Categorical Boosting의 약자로 category 문제를 다루는데 특화된 모델이다. 
Category가 아닌 numerical 문제에도 사용가능하고 각종 대회에서 우수한 성능을 보였다는 후기들이 있다.

### 2. CatBoost 시행착오
가장 처음에 당황했던건 CatBoost가 PyTorch 기반이 아니라는 것이었다. Baseline 모델들은 전부 PyTorch 기반이었고, 그 외의 다른 라이브러리를 사용할줄은 몰랐다.
그러나 CatBoost는 자체 라이브러리를 사용해야 했으며 내부적으로 이미 완성되어 내가 추가로 할 수 있는 무언가는 거의 없어보였다. 
CatBoost를 사용하기 위해 기존의 baseline 코드를 거의 대부분 새롭게 고쳤고, PyTorch 기반이 아니기 때문에 wandb 또한 사용할 수 없었다. 
기존 코드들이 사용했던 epoch을 CatBoost의 iteration에 바로 대입해서 사용했으나 CatBoost의 연산이 더 단순해서인지 훨씬 더 많은 횟수가 필요했다. 
Learning rate 또한 기존의 모델들은 1e-3이었으나 시행착오 끝에 이보다 큰 0.05 정도로 설정하는 것이 더 효율적이었다.

### 3. 본격적인 CatBoost 사용
처음에는 인터넷에 있는 CatBoost 예제를 거의 그대로 가져왔었기 때문에 사용 data는 user_id, isbn, rating이 전부였다. 
그러나 이 단순한 조합만으로 모든 Baseline 모델들보다 더 뛰어난 성능을 보였다.

    CatBoost RMSE: 2.1971
    FM RMSE: 2.4133
    FFM RMSE: 2.4253
    DCN RMSE: 2.4609
    CNN_FM RMSE: 23.0441

전처리 되지 않은 data 만으로도 월등한 성능을 보였기때문에 CatBoost를 더 파고들기 시작했다. 
train data에 user_id와 isbn 뿐만 아니라 age, location, book_title, book_author, year_of_publication, publisher, language, category 까지 image와 summary를 제외한
모든 field를 추가시켰고, 학습시킨 결과 성능이 더 좋게 나왔다.
Data 팀에서 결측치를 채우는 작업을 계속 진행함과 동시에 그 data를 바탕으로 CatBoost를 사용하였다. 
EDA가 거의 끝나가면서 CatBoost의 성능도 한계에 다다랐고, image를 제외한 모든 field를 포함시킴으로써 더 이상의 성능향상은 기대하기 어려워졌다.

### 4. 기존 모델과의 앙상블
한 번 테스트해본다는 생각으로 CatBoost 이전에 가장 성능이 좋았던 결과와 앙상블을 시켜보았다.
결과는 놀랍게도 RMSE가 약 0.02 감소하면서 일시적으로 1등으로 치고 올라가게 되었다. 
CatBoost와 앙상블한 모델은 FFM과 DCN을 합성한 모델이었다. 
결국 ML계열 모델, DL계열 모델, GBM계열 모델을 전부 이용한 결과가 가장 좋은 성능을 보인다는 결과를 얻었다.
