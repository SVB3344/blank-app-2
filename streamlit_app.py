import sys
print(sys.executable)  # Проверит текущий интерпретатор
import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("🖼️ СудКул: Обработка изображений")

uploaded_file = st.file_uploader("Загрузите изображение", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    
    # Конвертация в градации серого
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Оригинал", use_column_width=True)
    with col2:
        st.image(gray, caption="Черно-белое", use_column_width=True, clamp=True)
        import streamlit as st
import numpy as np

st.title("📷 Калибровка камеры")

st.markdown("""
### Инструкция:
1. Загрузите несколько изображений шахматной доски
2. Укажите размер сетки (например, 9x6)
""")

grid_width = st.number_input("Ширина сетки (количество углов)", min_value=3, value=9)
grid_height = st.number_input("Высота сетки (количество углов)", min_value=3, value=6)

uploaded_files = st.file_uploader("Загрузите изображения для калибровки", 
                                type=["jpg", "png"], 
                                accept_multiple_files=True)

if uploaded_files:
    st.success(f"Загружено {len(uploaded_files)} изображений. Режим калибровки...")
    # Здесь можно добавить реальный код калибровки с использованием OpenCV
    import streamlit as st
from PIL import Image
import torch

st.title("🚗 Детекция транспортных средств")

model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

uploaded_file = st.file_uploader("Загрузите изображение с транспортным средством", type=["jpg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    results = model(image)
    
    st.image(
        results.render()[0], 
        caption=f"Обнаружено {len(results.xyxy[0])} объектов",
        use_column_width=True
    )
    
    # Фильтрация только транспортных средств
    vehicles = results.pandas().xyxy[0][results.pandas().xyxy[0]['name'].isin(['car', 'truck', 'bus'])]
    st.write("Обнаруженные транспортные средства:")
    st.dataframe(vehicles)
    import streamlit as st
from PIL import Image
import torch
import pandas as pd
import numpy as np
import streamlit as st
from ultralytics import YOLO
import cv2

# --- Часть 1: Визуализация данных ---
st.header("📊 Анализ данных датчиков")

# Генерация тестовых данных
data = pd.DataFrame({
    'Время': pd.date_range(start='2023-01-01', periods=100, freq='H'),
    'Температура': np.random.normal(25, 5, 100),
    'Влажность': np.random.uniform(30, 80, 100)
})

# График
st.line_chart(data.set_index('Время'))

# Фильтрация
min_temp, max_temp = st.slider(
    "Диапазон температур",
    float(data['Температура'].min()),
    float(data['Температура'].max()),
    (20.0, 30.0)
)

filtered_data = data[(data['Температура'] >= min_temp) & (data['Температура'] <= max_temp)]
st.write(f"Найдено {len(filtered_data)} записей:")
st.dataframe(filtered_data)

st.title("Мое приложение - Streamfile")
st.header("Обработка компьютерного зрения")

# Панель управления в сайдбаре
with st.sidebar:
    st.subheader("Настройки")
    
    # Настройки кадров
    st.markdown("### Кадры")
    start_frame = st.number_input("Начальный кадр", value=0.0, step=0.1)
    end_frame = st.number_input("Конечный кадр", value=1.0, step=0.1)
    
    # Настройки модели
    st.markdown("### Модель")
    confidence = st.slider("Порог уверенности", 0.1, 1.0, 0.5)
    
    st.markdown("### YBG Кодирование")
    ybg_value = st.text_input("YBG на Rebelt", "2,3")
    downboxes = st.text_input("Downboxes", "5,3")

# Основной интерфейс
tab1, tab2 = st.tabs(["Обработка", "Результаты"])

with tab1:
    uploaded_file = st.file_uploader("Загрузите видео/изображение", 
                                  type=["mp4", "avi", "jpg", "png"])
    
    if uploaded_file:
        # Обработка в зависимости от типа файла
        if uploaded_file.type.startswith('video'):
            st.video(uploaded_file)
            # Здесь будет логика обработки видео
        else:
            # Конвертация изображения
            image = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(image, cv2.IMREAD_COLOR)
            st.image(image, caption="Загруженное изображение", channels="BGR")
            
            # Детекция объектов YOLO
            if st.button("Запустить детекцию"):
                model = YOLO('yolov8n.pt')
                results = model.predict(image, conf=confidence)
                res_plotted = results[0].plot()  # Визуализация результатов
                st.image(res_plotted, caption="Результаты детекции", channels="BGR")

with tab2:
    st.write("Здесь будут отображаться результаты обработки")
    st.write(f"Диапазон кадров: {start_frame} - {end_frame}")
    st.write(f"YBG кодирование: {ybg_value}")
    st.write(f"Downboxes: {downboxes}")