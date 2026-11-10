import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)

st.set_page_config(page_title="고객 이탈 예측", layout="centered")
st.title("통신사 고객 이탈 예측 (LightGBM + 로지스틱 앙상블)")


@st.cache_data
def load_raw():
    # 원본 데이터 정리 (분석 화면에서도 같이 씀)
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')
    df = df.drop('customerID', axis=1)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
    df['TenureGroup'] = pd.cut(df['tenure'], bins=[-1, 12, 24, 48, 72],
                               labels=['0-12개월', '13-24개월', '25-48개월', '49-72개월']).astype(str)
    return df


@st.cache_data
def load_data(raw):
    df = raw.copy()
    # 새 변수 만들기
    service_cols = ['PhoneService', 'MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                    'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    df['ServiceCount'] = (df[service_cols] == 'Yes').sum(axis=1)
    df['AvgChargePerMonth'] = df['TotalCharges'] / (df['tenure'] + 1)
    df['ContractInternet'] = df['Contract'] + '_' + df['InternetService']

    X = pd.get_dummies(df.drop('Churn', axis=1), drop_first=True)
    y = df['Churn']
    return X, y


@st.cache_resource
def train_model(X, y):
    # 1) 테스트 데이터는 마지막 평가에만 씁니다.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=156)

    # 2) LightGBM 설정을 학습 데이터 안에서 교차검증으로 찾습니다.
    param_dist = {
        'num_leaves': [7, 15, 31, 63],
        'learning_rate': [0.01, 0.03, 0.05],
        'min_child_samples': [10, 20, 40, 60],
        'reg_lambda': [0, 1, 5, 10],
        'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
        'subsample': [0.7, 0.85, 1.0],
    }
    base = LGBMClassifier(n_estimators=300, subsample_freq=1,
                          random_state=156, verbose=-1)
    search = RandomizedSearchCV(
        base, param_dist, n_iter=30, scoring='accuracy',
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=156),
        random_state=156, n_jobs=-1)
    search.fit(X_train, y_train)
    lgbm = search.best_estimator_

    # 3) 로지스틱 회귀(표준화 포함)와 LightGBM을 섞은 앙상블
    logit = make_pipeline(StandardScaler(),
                          LogisticRegression(max_iter=2000, C=0.5))
    ensemble = VotingClassifier(
        estimators=[('lgbm', lgbm), ('logit', logit)],
        voting='soft', weights=[2, 1])

    # 4) 앙상블을 학습 데이터 안에서 5번 나눠 검증 (정확도, AUC)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=156)
    cv_acc, cv_auc = [], []
    for tr, va in cv.split(X_train, y_train):
        ensemble.fit(X_train.iloc[tr], y_train.iloc[tr])
        p = ensemble.predict_proba(X_train.iloc[va])[:, 1]
        cv_acc.append(accuracy_score(y_train.iloc[va], (p >= 0.5).astype(int)))
        cv_auc.append(roc_auc_score(y_train.iloc[va], p))

    # 5) 최종 모델은 학습 데이터 전체로 다시 학습하고, 테스트는 한 번만 평가
    ensemble.fit(X_train, y_train)
    return ensemble, X_test, y_test, search.best_params_, cv_acc, cv_auc


raw = load_raw()
X, y = load_data(raw)
with st.spinner("모델을 학습하는 중입니다. 처음 한 번만 조금 걸려요."):
    model, X_test, y_test, best_params, cv_acc, cv_auc = train_model(X, y)
proba = model.predict_proba(X_test)[:, 1]

# 사이드바
st.sidebar.header("설정")
thr = st.sidebar.slider("예측 기준 임계값", 0.10, 0.90, 0.50, 0.01,
                        help="이탈 확률이 이 값 이상이면 이탈로 예측해요. "
                             "높이면 정밀도가 오르고, 낮추면 재현율이 올라요.")
rate = st.sidebar.number_input("환율 (1달러당 원)", min_value=1, value=1450, step=10,
                               help="요금을 원으로 바꿀 때 쓰는 환율이에요. 원본 데이터 요금을 달러로 보고 계산해요.")
pred = (proba >= thr).astype(int)

# 1. 성능 지표
st.subheader("모델 성능 지표")
c1, c2, c3 = st.columns(3)
c1.metric("정확도", f"{accuracy_score(y_test, pred):.3f}")
c2.metric("정밀도", f"{precision_score(y_test, pred, zero_division=0):.3f}")
c3.metric("재현율", f"{recall_score(y_test, pred):.3f}")
c4, c5 = st.columns(2)
c4.metric("F1", f"{f1_score(y_test, pred):.3f}")
c5.metric("ROC-AUC", f"{roc_auc_score(y_test, proba):.3f}")
st.caption(f"교차검증 정확도(학습 데이터 기준): {np.mean(cv_acc):.3f}")

with st.expander("LightGBM 최적 설정 보기"):
    st.write(best_params)

# 2. 오차 행렬
st.subheader("오차 행렬 (Confusion Matrix)")
cm = confusion_matrix(y_test, pred)
cm_df = pd.DataFrame(cm,
                     index=["실제 유지", "실제 이탈"],
                     columns=["예측 유지", "예측 이탈"])
st.table(cm_df)
tn, fp, fn, tp = cm.ravel()
st.caption(f"맞게 예측: 유지 {tn}명, 이탈 {tp}명 / "
           f"틀리게 예측: 유지를 이탈로 {fp}명, 이탈을 유지로 {fn}명")

# 3. 변수 중요도 (LightGBM 기준)
st.subheader("이탈에 영향을 많이 준 변수 TOP 10")
lgbm_model = model.named_estimators_['lgbm']
imp = pd.Series(lgbm_model.feature_importances_, index=X.columns) \
        .sort_values(ascending=False).head(10)
fig, ax = plt.subplots(figsize=(8, 5))
imp.sort_values().plot(kind='barh', ax=ax)
st.pyplot(fig)

# 4. 고객별 결과 (슬라이더에 따라 표가 함께 바뀜)
st.subheader("고객별 이탈 확률 확인")
idx = st.slider("테스트 고객 번호", 0, len(X_test) - 1, 0)

actual = "이탈" if y_test.iloc[idx] == 1 else "유지"
predicted = "이탈" if pred[idx] == 1 else "유지"
result_df = pd.DataFrame({
    "값": [
        f"{proba[idx]:.1%}",
        actual,
        predicted,
        "정답" if actual == predicted else "오답",
    ]
}, index=["예측 이탈 확률", "실제 결과", "예측 결과", "예측 정답 여부"])
st.table(result_df)

st.markdown("**이 고객의 주요 정보**")
info = X_test.iloc[idx]
info_df = pd.DataFrame({
    "값": [
        f"{int(info['tenure'])}개월",
        f"{info['MonthlyCharges'] * rate:,.0f} 원",
        f"{info['TotalCharges'] * rate:,.0f} 원",
    ]
}, index=["가입 기간", "월 요금", "총 요금"])
st.table(info_df)


# ==============================================================
# 추가 분석
# ==============================================================
st.divider()
st.header("추가 분석")

# 5. 고객 특성별 이탈률
st.subheader("5. 어떤 고객이 많이 떠날까? (특성별 이탈률)")
overall = raw['Churn'].mean()
st.write(f"전체 고객 {len(raw):,}명 중 이탈률은 **{overall:.1%}**입니다.")

feature_map = {
    "계약 형태": "Contract",
    "인터넷 서비스": "InternetService",
    "가입 기간 구간": "TenureGroup",
    "결제 방법": "PaymentMethod",
    "기술 지원 가입": "TechSupport",
    "온라인 보안 가입": "OnlineSecurity",
    "고령자 여부": "SeniorCitizen",
}
choice = st.selectbox("살펴볼 특성을 고르세요", list(feature_map))
col = feature_map[choice]
group_key = raw[col].map({0: "아니오", 1: "예"}) if col == "SeniorCitizen" else raw[col]
grp = raw.groupby(group_key)['Churn'].agg(['mean', 'count'])
grp.columns = ['이탈률', '고객 수']
grp = grp.sort_values('이탈률', ascending=False)
grp.index.name = choice
st.bar_chart(grp['이탈률'])
show = grp.copy()
show['이탈률'] = show['이탈률'].map('{:.1%}'.format)
show['전체 대비'] = (grp['이탈률'] / overall).map('{:.1f}배'.format)
st.table(show)

st.markdown("**이탈 고객 vs 유지 고객 평균 비교**")
cmp = raw.groupby('Churn')[['tenure', 'MonthlyCharges', 'TotalCharges']].mean()
cmp_df = pd.DataFrame({
    "평균 가입 기간": cmp['tenure'].map('{:.1f}개월'.format),
    "평균 월 요금": (cmp['MonthlyCharges'] * rate).map('{:,.0f} 원'.format),
    "평균 총 요금": (cmp['TotalCharges'] * rate).map('{:,.0f} 원'.format),
})
cmp_df.index = ["유지 고객" if i == 0 else "이탈 고객" for i in cmp_df.index]
st.table(cmp_df)

# 6. 기준선 비교
st.subheader("6. 모델이 정말 쓸모 있을까? (기준선 비교)")
base_acc = 1 - y_test.mean()
model_acc = accuracy_score(y_test, pred)
b1, b2, b3 = st.columns(3)
b1.metric("기준선 정확도", f"{base_acc:.3f}", help="모델 없이 모든 고객을 '유지'라고 찍었을 때의 정확도")
b2.metric("모델 정확도", f"{model_acc:.3f}")
b3.metric("개선 폭", f"{(model_acc - base_acc) * 100:+.1f}%p")
st.caption("이탈 고객이 적은 데이터라서 '전부 유지'라고만 해도 정확도가 꽤 높게 나와요. "
           "모델의 가치는 이 기준선보다 얼마나 나은지로 판단해야 해요.")

# 7. 모델 안정성 (교차검증)
st.subheader("7. 결과가 운이 아닐까? (교차검증 안정성)")
cv_df = pd.DataFrame({
    "정확도": cv_acc,
    "ROC-AUC": cv_auc,
}, index=[f"{i + 1}번째" for i in range(len(cv_acc))])
cv_show = cv_df.map('{:.3f}'.format)
cv_show.loc["평균 ± 편차"] = [
    f"{np.mean(cv_acc):.3f} ± {np.std(cv_acc):.3f}",
    f"{np.mean(cv_auc):.3f} ± {np.std(cv_auc):.3f}",
]
st.table(cv_show)
st.caption("학습 데이터를 5번 다르게 나눠 검증한 결과예요. 편차가 작을수록 "
           "테스트 점수가 운이 아니라 모델의 실력이라고 볼 수 있어요.")

# 8. 임계값별 성능 변화
st.subheader("8. 임계값을 바꾸면 성능이 어떻게 변할까?")
ths = np.round(np.arange(0.10, 0.91, 0.05), 2)
rows = []
for t in ths:
    p_t = (proba >= t).astype(int)
    rows.append([
        accuracy_score(y_test, p_t),
        precision_score(y_test, p_t, zero_division=0),
        recall_score(y_test, p_t),
        f1_score(y_test, p_t, zero_division=0),
    ])
th_df = pd.DataFrame(rows, index=ths, columns=['정확도', '정밀도', '재현율', 'F1'])
th_df.index.name = '임계값'
st.line_chart(th_df)
st.write(f"정확도가 가장 높은 임계값: **{th_df['정확도'].idxmax():.2f}** "
         f"(정확도 {th_df['정확도'].max():.3f}) / "
         f"F1이 가장 높은 임계값: **{th_df['F1'].idxmax():.2f}** "
         f"(F1 {th_df['F1'].max():.3f})")
st.caption("참고용 그래프예요. 테스트 데이터로 임계값을 고르면 점수가 살짝 부풀려질 수 있어요.")

# 9. 타겟팅 효과
st.subheader("9. 마케팅에 쓰면 얼마나 효과적일까? (타겟팅 효과)")
top_pct = st.slider("이탈 확률이 높은 상위 몇 %의 고객에게 리텐션 마케팅을 할까요?",
                    5, 50, 20, 5)
order = np.argsort(-proba)
y_arr = y_test.values
n_target = max(1, int(len(proba) * top_pct / 100))
targeted = y_arr[order[:n_target]]
captured = targeted.sum() / y_arr.sum()
hit = targeted.mean()
lift = hit / y_arr.mean()
t1, t2, t3 = st.columns(3)
t1.metric("타겟 고객 수", f"{n_target}명")
t2.metric("실제 이탈자 포착률", f"{captured:.1%}")
t3.metric("타겟 중 실제 이탈 비율", f"{hit:.1%}", f"무작위 대비 {lift:.1f}배")
st.write(f"이탈 확률 상위 {top_pct}% 고객({n_target}명)에게만 마케팅해도 "
         f"실제 이탈자의 **{captured:.1%}**를 잡을 수 있어요. "
         f"무작위로 같은 수를 고르면 약 {top_pct}%만 잡혀요.")

pcts = np.arange(1, 101)
cap = [y_arr[order[:max(1, int(len(order) * p / 100))]].sum() / y_arr.sum() for p in pcts]
gain_df = pd.DataFrame({"모델로 고를 때": cap, "무작위로 고를 때": pcts / 100}, index=pcts)
gain_df.index.name = "상위 고객 비율(%)"
st.line_chart(gain_df)
st.caption("두 선 사이가 벌어질수록 모델이 이탈 고객을 잘 골라낸다는 뜻이에요.")
