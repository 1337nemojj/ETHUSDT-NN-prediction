import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import time
from tqdm import tqdm

def progress_bar(iteration, total, prefix='', suffix='', decimals=1, bar_length=100):
    # Create the progress bar
    progress = bar_length * iteration // total
    filled = '#' * progress
    remaining = '-' * (bar_length - progress)

    # Print the progress bar
    print('\r%s |%s| %s%% %s' % (prefix, filled, int(100 * iteration / total), suffix), end='')

    # Check if the iteration is complete
    if iteration == total:
        print()

# Создаем систему нечеткого вывода FIS типа Сугено (пример из предыдущего ответа)
def create_fis_system():
        
    # Задаем нечеткие переменные для входов
    speed = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'speed')
    blocking_efficiency = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'blocking_efficiency')
    resource_usage = ctrl.Antecedent(np.arange(0, 1.1, 0.1), 'resource_usage')

    # Задаем нечеткую переменную для выхода (класс)
    output_class = ctrl.Consequent(np.arange(0, 1.1, 0.1), 'output_class')

    # Определяем нечеткие множества для входов и выхода
    speed['low'] = fuzz.trimf(speed.universe, [0, 0, 0.5])
    speed['medium'] = fuzz.trimf(speed.universe, [0, 0.5, 1])
    speed['high'] = fuzz.trimf(speed.universe, [0.5, 1, 1])

    blocking_efficiency['low'] = fuzz.trimf(blocking_efficiency.universe, [0, 0, 0.5])
    blocking_efficiency['medium'] = fuzz.trimf(blocking_efficiency.universe, [0, 0.5, 1])
    blocking_efficiency['high'] = fuzz.trimf(blocking_efficiency.universe, [0.5, 1, 1])

    resource_usage['low'] = fuzz.trimf(resource_usage.universe, [0, 0, 0.5])
    resource_usage['medium'] = fuzz.trimf(resource_usage.universe, [0, 0.5, 1])
    resource_usage['high'] = fuzz.trimf(resource_usage.universe, [0.5, 1, 1])

    output_class['good'] = fuzz.trimf(output_class.universe, [0, 0, 0.5])
    output_class['bad'] = fuzz.trimf(output_class.universe, [0.5, 1, 1])

    # Правила вывода
    rule1 = ctrl.Rule(speed['high'] & blocking_efficiency['high'] & resource_usage['low'], output_class['good'])
    rule2 = ctrl.Rule(speed['low'] & blocking_efficiency['low'] & resource_usage['high'], output_class['bad'])
    rule3 = ctrl.Rule(speed['medium'] | blocking_efficiency['medium'] | resource_usage['medium'], output_class['good'])

    # Создаем систему нечеткого вывода и добавляем правила
    fis = ctrl.ControlSystem([rule1, rule2, rule3])
    fis_sim = ctrl.ControlSystemSimulation(fis)

    return fis_sim

# Загрузка данных
# ... (как в предыдущих примерах)
file_path = 'expanded_data.csv'
data = pd.read_csv(file_path)

# Создаем систему нечеткого вывода FIS
fis_system = create_fis_system()

# Получаем вывод FIS для обучающих данных
column_names = ['Speed', 'Blocking Efficiency', 'Resource Usage']
if all(name in data.columns for name in column_names):
    X_fis = data[column_names]
    y_fis = data['Class']
    fis_outputs = []
else:
    raise KeyError(f"Один или несколько столбцов {column_names} не найдены в данных.")

fis_outputs = []

for index, row in X_fis.iterrows():
    fis_system.input['speed'] = row['Speed']
    fis_system.input['blocking_efficiency'] = row['Blocking Efficiency']
    fis_system.input['resource_usage'] = row['Resource Usage']
    fis_system.compute()
    fis_outputs.append(fis_system.output['output_class'])


max_iter=500000
# Создаем нейронную сеть для обработки вывода FIS
X_train, X_test, y_train, y_test, fis_outputs_train, fis_outputs_test = train_test_split(X_fis, y_fis, fis_outputs, test_size=0.08, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(20,), max_iter=max_iter, random_state=42)
mlp.fit(X_train, y_train)

# Предсказание на тестовых данных
y_pred = mlp.predict(X_test)

# Оценка точности и других метрик
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f'Tочность модели: {accuracy}')
print('Матрица ошибок:')
print(conf_matrix)


# 1. График обучения нейронной сети
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.plot(mlp.loss_curve_)
plt.title('График обучения нейронной сети')
plt.xlabel('Эпохи')
plt.ylabel('Ошибка')

# 2. Гистограмма ошибок на обучающей и тестовой выборках
train_errors = 1 - mlp.score(X_train, y_train)
test_errors = 1 - mlp.score(X_test, y_test)
plt.subplot(1, 3, 2)
plt.bar(['Обучающая', 'Тестовая'], [train_errors, test_errors])
plt.title('Ошибка на обучающей и тестовой выборках')
plt.ylabel('Ошибка')

# 3. График предсказанных значений и истинных меток на тестовой выборке
plt.subplot(1, 3, 3)
plt.scatter(range(len(y_test)), y_test, label='Истинные значения', marker='o')
plt.scatter(range(len(y_test)), y_pred, label='Предсказанные значения', marker='x')
plt.title('Предсказанные и истинные значения на тестовой выборке')
plt.xlabel('Примеры')
plt.ylabel('Класс')
plt.legend()

plt.tight_layout()
plt.show()