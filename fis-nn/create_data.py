import pandas as pd
import numpy as np

# Создание примера набора данных с увеличенным объемом данных
np.random.seed(42)  # Для воспроизводимости результатов

# Создаем набор данных
num_samples = 10000  # Увеличьте это число для получения большего объема данных

speed = np.random.uniform(0, 1, num_samples)
blocking_efficiency = np.random.uniform(0, 1, num_samples)
resource_usage = np.random.uniform(0, 1, num_samples)
label = np.random.choice(['good', 'bad'], size=num_samples)

data = pd.DataFrame({'Speed': speed, 'Blocking Efficiency': blocking_efficiency,
                     'Resource Usage': resource_usage, 'Class': label})

# Сохраняем данные в файл
data.to_csv('expanded_data.csv', index=False)