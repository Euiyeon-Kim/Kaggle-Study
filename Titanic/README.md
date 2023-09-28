# Titanic - Machine Learning from Disaster


[Competition Link](https://www.kaggle.com/c/titanic/overview)

# 모델별 성능 비교

## Decision Tree - 0.74
~~~python
dt = DecisionTreeClassifier(max_depth=12, random_state=1234)
~~~

## Random Forest - 0.77
~~~python
rf = RandomForestClassifier(n_estimators=50,
                            max_depth=5,
                            max_samples=0.9,
                            random_state=1234)
~~~
## Gradient Boosting - 0.77
~~~python
gbc = GradientBoostingClassifier(n_estimators=5000,
                                 subsample=0.67,
                                 learning_rate=0.02,
                                 max_depth=4,
                                 validation_fraction=0.05,
                                 n_iter_no_change=10,
                                 verbose=0,
                                 random_state=1234)
 ~~~
## Support Vector Machine - 0.78
~~~python
svc = LinearSVC(loss='hinge',
                tol=1e-4, 
                C=0.5,
                max_iter=500000,
                random_state=1234)
~~~


# Pandas
- 파일 읽기
    ~~~python3
    train_data = pd.read_csv("datas/train.csv")
    train_data.head()
    ~~~
- 결측치 탐지
    ~~~python3
    train_data.isnull().sum()
    ~~~
- 결측치 채우기
    ~~~python3
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
    ~~~
- 데이터 항목 없애기
  ~~~python3
  train_data = train_data.drop(['quality'], axis = 1)
  ~~~
- 각 항목 갯수 세기
    ~~~python3
    train_data['Cabin'].value_counts(dropna=False)    
    ~~~
- 모수 분석
    ~~~python3
    train_data['Age'].describe()
    ~~~
- 데이터 df 형태로 가져오기
    ~~~python3
    train_data.iloc[[0, 1, 2, 3]]
    train_data.loc[[0, 1, 2, 3]]
    ~~~

## 제출하기
~~~bash
kaggle competitions submit -c titanic -f submission.csv -m "Message"
~~~