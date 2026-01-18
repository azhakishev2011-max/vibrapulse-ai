
import streamlit as st
import pandas as pd
import catboost as cb
import matplotlib.pyplot as plt

# Загружаем модель
model = cb.CatBoostClassifier()
try:
    model.load_model('/content/esp_failure_model_multi.cbm')
except Exception as e:
    st.error(f"Ошибка загрузки модели: {e}")
    st.stop()

st.title("VibraPulse AI (демо)")

st.write("Загрузите CSV с данными вибрации (без id и label).")

uploaded_file = st.file_uploader("Выберите CSV-файл", type="csv")

if uploaded_file is not None:
    # Читаем файл с разделителем ; (для твоих CSV)
    df = pd.read_csv(uploaded_file, sep=';')
    
    # Удаляем ненужные столбцы, если есть
    cols_to_drop = ['id', 'esp_id', 'label']
    df = df.drop([col for col in cols_to_drop if col in df.columns], axis=1, errors='ignore')
    
    # Предсказываем вероятности по классам
    proba = model.predict_proba(df)
    classes = model.classes_
    
    predicted_type = [classes[i] for i in proba.argmax(axis=1)]
    max_proba = proba.max(axis=1) * 100
    
    df['Risk (%)'] = max_proba
    df['Тип поломки'] = predicted_type
    
    st.subheader("Результаты предсказания")
    st.dataframe(df[['Risk (%)', 'Тип поломки']])
    
    # ПРОСТОЙ ПРОГНОЗ ВРЕМЕНИ
    if len(df) >= 10:
        last_risks = df['Risk (%)'].tail(10)
        risk_growth = last_risks.diff().mean()
        
        if risk_growth > 5:
            days_est = int(70 / risk_growth)
            if days_est <= 3:
                text = "в ближайшие 1–3 дня — срочно проверить!"
            elif days_est <= 7:
                text = f"примерно через {days_est}–7 дней"
            else:
                text = f"через 7–{days_est + 7} дней"
            
            st.warning(f"⚠️ Поломка может случиться {text}")
        else:
            st.success("Риск не растёт — пока всё спокойно")
    else:
        st.info("Мало данных для прогноза времени (нужно минимум 10 записей)")
    
    # РЕКОМЕНДАЦИИ — только если тип НЕ Normal и риск > 85%
    st.subheader("Рекомендации по предотвращению")
    has_recommendations = False
    
    for i, row in df.iterrows():
        risk = row['Risk (%)']
        failure_type = row['Тип поломки']
        
        if failure_type != 'Normal' and risk > 85:
            has_recommendations = True
            if failure_type == 'Unbalance':
                st.warning(f"Запись {i}: Высокий риск дисбаланса. Рекомендация: Проверьте балансировку ротора, снизьте нагрузку на 10–15%, осмотрите вал. Это может снизить риск на 40–60%.")
            elif failure_type == 'Rubbing':
                st.warning(f"Запись {i}: Высокий риск трения. Рекомендация: Осмотрите подшипники, проверьте на задевание деталей, очистите от отложений.")
            elif failure_type == 'Faulty sensor':
                st.warning(f"Запись {i}: Высокий риск неисправного датчика. Рекомендация: Проверьте и замените датчик вибрации или давления.")
            elif failure_type == 'Misalignment':
                st.warning(f"Запись {i}: Высокий риск несоосности. Рекомендация: Проведите центровку вала и мотора, проверьте крепления.")
            else:
                st.info(f"Запись {i}: Риск высокий, тип: {failure_type}. Рекомендация: Осмотрите насос полностью.")
    
    if not has_recommendations:
        st.success("Нет записей с высоким риском и реальной поломкой — рекомендации не требуются")
    
    # График
    fig, ax = plt.subplots()
    ax.plot(df.index, df['Risk (%)'], marker='o', color='red')
    ax.set_xlabel('Запись')
    ax.set_ylabel('Risk (%)')
    ax.set_title('Изменение риска поломки')
    ax.axhline(70, color='orange', linestyle='--', label='Порог 70%')
    ax.axhline(90, color='red', linestyle='--', label='Критический 90%')
    ax.legend()
    st.pyplot(fig)
    
    # Алерты
    max_risk = df['Risk (%)'].max()
    if max_risk > 90:
        st.error(f"Критическая тревога! Макс риск: {max_risk:.1f}%")
    elif max_risk > 85:
        st.warning(f"Высокий риск! Макс риск: {max_risk:.1f}%")
    else:
        st.success(f"Всё в норме. Макс риск: {max_risk:.1f}%")
