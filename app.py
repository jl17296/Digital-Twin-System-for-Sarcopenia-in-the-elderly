import subprocess
import sys

# 强制在运行时安装缺失的库
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# 这里补上你需要的所有库
required_packages = ['seaborn', 'pandas', 'numpy', 'matplotlib', 'scikit-learn']

for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        install(package)

# 下面才是你原本的代码
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# 设置页面
st.set_page_config(page_title="老年肌少症数字孪生系统", layout="wide")


# 1. 加载数据并训练模型 (因为你没存模型文件，我们网页启动时临时训练一个，保证能用)
@st.cache_resource
def get_model_and_data():
    df = pd.read_csv('老年肌少症数字孪生专题数据集_已标注.csv')

    # 【新增：清洗数据】
    # 1. 自动填补所有空值（用中位数填充，这在医学数据处理里最稳妥）
    df = df.fillna(df.median())

    # 2. 检查一下标签列有没有缺失值，如果有，直接删掉那一行
    if df['Sarcopenia_Label'].isnull().any():
        df = df.dropna(subset=['Sarcopenia_Label'])

    exclude_cols = ['ID', 'ivy', 'bloodweight', 'lgrip', 'rgrip', 'wspeed', 'max_grip', 'Sarcopenia_Label']
    X_cols = [c for c in df.columns if c not in exclude_cols]

    # 临时训练模型
    X = df[X_cols]
    y = df['Sarcopenia_Label']
    model = RandomForestClassifier().fit(X, y)
    return model, X_cols, df[X_cols].median().to_dict()

with st.spinner('正在激活数字孪生系统计算引擎，请稍后...'):
    rf_model, X_cols_base, feature_medians = get_model_and_data()
st.success('数字孪生模拟系统已准备就绪')

# 2. 侧边栏 UI 输入区
st.sidebar.header("👤 人口学基本表型")
gender = st.sidebar.selectbox("性别", [("男性", 1.0), ("女性", 2.0)], format_func=lambda x: x[0])[1]
age = st.sidebar.slider("年龄(岁)", 60, 95, 82)
bmi = st.sidebar.slider("BMI", 14.0, 35.0, 15.8)

st.sidebar.header("🏃 行为与身体功能")
sleep = st.sidebar.slider("睡眠时间(h)", 3.0, 10.0, 4.0)
iadl = st.sidebar.slider("IADL评分", 0.0, 10.0, 5.0)
smokev = st.sidebar.selectbox("吸烟史", [("否", 0.0), ("是", 1.0)], format_func=lambda x: x[0])[1]
chronic = st.sidebar.slider("慢性病种数", 0, 8, 3)
fall = st.sidebar.selectbox("两年跌倒史", [("无", 0.0), ("有", 1.0)], format_func=lambda x: x[0])[1]

st.sidebar.header("🩸 微观生化指标")
crea = st.sidebar.slider("初始血肌酐", 0.2, 1.5, 0.45)
cysc = st.sidebar.slider("C胱抑素C", 0.5, 2.5, 1.65)
hgb = st.sidebar.slider("血红蛋白", 8.0, 18.0, 10.5)
glu = st.sidebar.slider("空腹血糖", 3.0, 12.0, 5.5)
crp = st.sidebar.slider("C反应蛋白", 0.1, 15.0, 6.2)
wbc = st.sidebar.slider("白细胞计数", 3.0, 15.0, 7.0)

# 3. 干预决策区
st.subheader("🎯 临床虚拟管理策略")
chk_exercise = st.checkbox("处方 A：定制化抗阻功能训练")
chk_nutrition = st.checkbox("处方 B：高蛋白膳食与纠正贫血")
chk_biomed = st.checkbox("处方 C：微观代谢纠正与抗炎调理")

# 4. 计算引擎
if st.button("🔮 启动临床机制推演"):
    # 构建 twin_v0
    twin_v0 = {col: feature_medians[col] for col in X_cols_base}
    twin_v0.update({'gender': gender, 'age': age, 'bmi': bmi, 'sleep': sleep, 'iadl': iadl,
                    'smokev': smokev, 'chronic': chronic, 'fall_down': fall, 'bl_crea': crea,
                    'bl_cysc': cysc, 'bl_hgb': hgb, 'bl_glu': glu, 'bl_crp': crp, 'bl_wbc': wbc, 'exercise': 0.0})

    prob_v0 = rf_model.predict_proba(pd.DataFrame([twin_v0]))[0, 1]

    # 模拟干预
    twin_v1 = twin_v0.copy()
    synergy = 1.25 if (chk_exercise and chk_nutrition) else 1.0

    if chk_exercise:
        twin_v1.update({'exercise': 1.0, 'bl_crea': twin_v1['bl_crea'] * (1 + 0.2 * synergy),
                        'iadl': max(0, twin_v1['iadl'] - 2 * synergy)})
    if chk_nutrition:
        twin_v1['bl_hgb'] = min(twin_v1['bl_hgb'] * (1 + 0.15 * synergy), feature_medians['bl_hgb'])
    if chk_biomed:
        twin_v1.update({'bl_crp': twin_v1['bl_crp'] * 0.6, 'bl_cysc': twin_v1['bl_cysc'] * 0.85, 'sleep': 7.0,
                        'bl_glu': feature_medians['bl_glu'], 'bl_wbc': feature_medians['bl_wbc'], 'smokev': 0.0})

    prob_v1 = rf_model.predict_proba(pd.DataFrame([twin_v1]))[0, 1]

    # 展示图表
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.barplot(x=['干预前', '干预后'], y=[prob_v0, prob_v1], palette=['#e74c3c', '#2ecc71'])
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    st.pyplot(fig)

    st.write(f"### 评估结果: 风险概率从 {prob_v0:.2%} 降至 {prob_v1:.2%}")

