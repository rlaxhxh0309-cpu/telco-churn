import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

st.set_page_config(page_title="고객 이탈 예측", layout="centered")
st.title("통신사 고객 이탈 예측 (LightGBM)")

# 데이터 준비 (지금까지 한 전처리와 같은 순서)
@st.cache_data
def load_data():
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = df.drop('customerID', axis=1)                                   # 고객 번호 제거
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})                   # 정답을 0/1로
    X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)        # 문자 열 → 숫자 열
    y = df['Churn']
    return X, y

# 모델 학습 (한 번만 학습하고 결과를 기억해 둠)
@st.cache_resource
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=156)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=156)
    model = LGBMClassifier(n_estimators=400, learning_rate=0.05, verbose=-1)
    model.fit(X_tr, y_tr, eval_metric='logloss', eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50)])
    return model, X_test, y_test

X, y = load_data()
model, X_test, y_test = train_model(X, y)

# 성능 지표
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:, 1]

c1, c2, c3 = st.columns(3)
c1.metric("정확도", f"{accuracy_score(y_test, pred):.3f}")
c2.metric("F1", f"{f1_score(y_test, pred):.3f}")
c3.metric("ROC-AUC", f"{roc_auc_score(y_test, proba):.3f}")

# 중요 변수 그래프
st.subheader("이탈에 영향을 많이 준 변수 TOP 10")
imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 5))
imp.sort_values().plot(kind='barh', ax=ax)
st.pyplot(fig)

# 고객 한 명씩 확인하기
st.subheader("고객별 이탈 확률 확인")
idx = st.slider("테스트 고객 번호", 0, len(X_test) - 1, 0)
st.write(f"예측 이탈 확률: **{proba[idx]:.1%}**")
st.write("실제 결과:", "이탈" if y_test.iloc[idx] == 1 else "유지")