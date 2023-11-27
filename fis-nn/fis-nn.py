import numpy as np
import matplotlib.pyplot as plt
import skfuzzy as fuzz
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score

# Загрузка данных
data = np.array([
    [0.8, 0.9, 0.2, 'Хороший'],
    [0.4, 0.6, 0.8, 'Плохой'],
    [0.6, 0.7, 0.3, 'Хороший'],
    [0.9, 0.8, 0.1, 'Хороший'],
    [0.2, 0.5, 0.9, 'Плохой'],
    [0.7, 0.6, 0.4, 'Плохой'],
    [0.5, 0.9, 0.3, 'Хороший'],
    [0.3, 0.4, 0.7, 'Плохой']
])

# Разделение данных на признаки и метки
X = data[:, :-1].astype(float)
y = np.where(data[:, -1] == 'Хороший', 1, 0)

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(X_train)
print(X_test)
print(y_test)
# Создание нейро-нечеткой сети
model = Sequential()
model.add(Dense(8, input_dim=3, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Компиляция модели
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Обучение модели
history = model.fit(X_train, y_train, epochs=50, batch_size=2, validation_data=(X_test, y_test), verbose=2)

# Визуализация обучения
plt.plot(history.history['accuracy'], label='Точность на обучающей выборке')
plt.plot(history.history['val_accuracy'], label='Точность на тестовой выборке')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()

# Задание форм функций принадлежности для входных и выходных переменных
mu_input_1 = fuzz.trimf(X_train[:, 0], [0, 0.5, 1])
mu_input_2 = fuzz.trimf(X_train[:, 1], [0, 0.5, 1])
mu_output = fuzz.trimf(X_train[:, 2], [0, 1, 1])

# Преобразование входных данных для нечеткой системы
inputs_fuzzy = np.column_stack((mu_input_1, mu_input_2, mu_output))

# Задание нечетких множеств и правил вывода
def create_fuzzy_system():
    input_1 = np.linspace(0, 1, 5)
    input_2 = np.linspace(0, 1, 5)
    output = np.linspace(0, 1, 2)

    # Задание форм функций принадлежности для входных и выходных переменных
    mu_input_1 = fuzz.trimf(input_1, [0, 0.5, 1])
    mu_input_2 = fuzz.trimf(input_2, [0, 0.5, 1])
    mu_output = fuzz.trimf(output, [0, 1, 1])

    # Визуализация функций принадлежности
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

    ax0.plot(input_1, mu_input_1, 'b', linewidth=1.5, label='Вход 1')
    ax0.set_title('Функции принадлежности для входа 1')

    ax1.plot(input_2, mu_input_2, 'g', linewidth=1.5, label='Вход 2')
    ax1.set_title('Функции принадлежности для входа 2')

    ax2.plot(output, mu_output, 'r', linewidth=1.5, label='Выход')
    ax2.set_title('Функции принадлежности для выхода')

    for ax in (ax0, ax1, ax2):
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Задание правил вывода
    rule1 = fuzz.fuzzy_and(inputs_fuzzy[:, 0], inputs_fuzzy[:, 1])
    rule2 = inputs_fuzzy[:, 2]

    # Структура системы нечеткого вывода
    fis = [rule1, rule2]

    return fis

# Обучение нечеткой системы
fis = create_fuzzy_system()
fuzzy_model = fuzz.defuzz(X_train[:, :2], fis)

# Преобразование входных данных для тестовой выборки
inputs_fuzzy_test = np.column_stack((mu_input_1, mu_input_2, mu_output))

# ...

# Производим оценку производительности нечеткой модели на тестовой выборке
predictions_fuzzy = []
for i in range(len(X_test)):
    fuzzy_model.input['Вход 1'] = inputs_fuzzy_test[i, 0]
    fuzzy_model.input['Вход 2'] = inputs_fuzzy_test[i, 1]
    fuzzy_model.input['Выход'] = inputs_fuzzy_test[i, 2]
    fuzzy_model.compute()
    predictions_fuzzy.append(fuzzy_model.output['Выход'])

predictions_fuzzy = np.array(predictions_fuzzy)
predictions_fuzzy_binary = np.where(predictions_fuzzy > 0.5, 1, 0)
