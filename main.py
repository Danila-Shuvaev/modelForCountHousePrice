import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#TODO:
# 1. Сделать мини датасет для обучения
# 2. Сделать метод для парсинга данных
# 3. Сделать метод для создания модели
# 4. Сделать метод для обучения модели
# 5. Сделать метод для построения прогноза
# 6. Сделать метод для ввода данных для определения стоимости дома
# 6. Сделать общий метод для вызова других

# Функция для подготовки данных
def parseData(data):

    X = data[:, :3]  # Площадь, этаж и год дома для стоимости
    y = data[:, 3]   # Стоимость дома

    # Нормализация входных данных
    X_scaler = StandardScaler()
    X_scaled = X_scaler.fit_transform(X)

    # Нормализация целевой переменной (стоимости)
    y_scaler = StandardScaler()
    y_scaled = y.reshape(-1, 1)  # делаем y двумерным для нормализации
    y_scaled = y_scaler.fit_transform(y_scaled).flatten()

    return X_scaled, y_scaled, X_scaler, y_scaler


# Функция для создания модели нейросети
def createModel(input_dim):

    model = nn.Sequential(
        nn.Linear(input_dim, 64), # 1 скрытый слой
        nn.ReLU(),
        nn.Linear(64, 32), # 2 скрытый слой
        nn.ReLU(),
        nn.Linear(32, 1) # выходной слой
    )
    return model


# Функция для обучения модели
def trainModel(model, X_train, y_train, epochs=100, batch_size=8):

    criterion = nn.MSELoss()  # Расчет ср. кв. ошибки
    optimizer = optim.Adam(model.parameters())  # Оптимизируем модель при помощи Adam

    model.train()

    for epoch in range(epochs):
        for i in range(0, len(X_train), batch_size):
            X_batch = torch.tensor(X_train[i:i + batch_size], dtype=torch.float32)
            y_batch = torch.tensor(y_train[i:i + batch_size], dtype=torch.float32).view(-1, 1)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()


def predictPrice(model, new_house, X_scaler, y_scaler):

    model.eval()
    new_house_scaled = X_scaler.transform(new_house)
    new_house_tensor = torch.tensor(new_house_scaled, dtype=torch.float32)

    with torch.no_grad():
        predicted_price_scaled = model(new_house_tensor)
    predicted_price = y_scaler.inverse_transform(predicted_price_scaled.detach().numpy().reshape(-1, 1)).flatten()[0]

    return predicted_price

def inputPredictData():

    area = float(input("Введите площадь дома в кв. метрах: "))
    floors = int(input("Введите количество этажей: "))
    dateYear = int(input("Введите год оценки: "))

    return np.array([[area, floors, dateYear]])


# Функция для полного процесса
def run_house_price_prediction():
    # Пример данных о загородных домах с учетом года
    data = np.array([
        [150, 2, 2018, 300000],
        [200, 3, 2018, 450000],
        [100, 1, 2018, 150000],
        [250, 3, 2019, 500000],
        [180, 2, 2019, 350000],
        [120, 1, 2020, 200000],
        [300, 4, 2020, 600000],
        [220, 2, 2021, 400000],
        [160, 1, 2021, 250000],
        [270, 3, 2022, 550000]
    ])

    X, y, X_scaler, y_scaler = parseData(data) # Парсим датасет

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42) # разделяем датасет на тестовый массив и обучающий

    model = createModel(input_dim=X_train.shape[1])

    trainModel(model, X_train, y_train)

    # Прогноз для нового дома
    new_house = inputPredictData()
    predicted_price = predictPrice(model, new_house, X_scaler, y_scaler)

    print(f"Прогнозируемая стоимость дома: {predicted_price:.2f} рублей")

run_house_price_prediction()
