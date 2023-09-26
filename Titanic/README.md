# Titanic - Machine Learning from Disaster


[Competition Link](https://www.kaggle.com/c/titanic/overview)


# Pandas
- 파일 읽기
    ~~~python3
    import pandas as pd
    train_data = pd.read_csv("datas/train.csv")
    train_data.head()
    ~~~
- 데이터 항목 없애기
    ~~~python3
    train_data = train_data.drop(['quality'], axis = 1)
    ~~~
- 결측치 탐지
    ~~~python3
    train_data.isnull().sum()
    ~~~
- 결측치 채우기
    ~~~python3
    train_data['Age'].fillna(train_data['Age'].median(), inplace=True)
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