############################################
# Полный пример для Jupyter Notebook
############################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

%matplotlib inline  # Если вы используете Jupyter Notebook

# ---------------------------
# Функция генерации синтетических данных
# ---------------------------
def generate_synthetic_data(num_samples=2000, ddos_ratio=0.3, random_state=42):
    np.random.seed(random_state)
    
    num_ddos = int(num_samples * ddos_ratio)
    num_normal = num_samples - num_ddos
    
    normal_src_ip_count = np.random.randint(1, 10, size=num_normal)
    ddos_src_ip_count   = np.random.randint(10, 100, size=num_ddos)
    
    normal_avg_packet_size = np.random.randint(200, 800, size=num_normal)
    ddos_avg_packet_size   = np.random.randint(100, 1500, size=num_ddos)
    
    normal_packet_rate = np.random.randint(10, 100, size=num_normal)
    ddos_packet_rate   = np.random.randint(1000, 5000, size=num_ddos)
    
    possible_protocols = [1, 6, 17]
    normal_protocol = np.random.choice(possible_protocols, size=num_normal)
    ddos_protocol   = np.random.choice(possible_protocols, size=num_ddos)
    
    normal_data = pd.DataFrame({
        'src_ip_count': normal_src_ip_count,
        'avg_packet_size': normal_avg_packet_size,
        'packet_rate': normal_packet_rate,
        'protocol_type': normal_protocol,
        'label': np.zeros(num_normal, dtype=int)
    })
    
    ddos_data = pd.DataFrame({
        'src_ip_count': ddos_src_ip_count,
        'avg_packet_size': ddos_avg_packet_size,
        'packet_rate': ddos_packet_rate,
        'protocol_type': ddos_protocol,
        'label': np.ones(num_ddos, dtype=int)
    })
    
    data = pd.concat([normal_data, ddos_data], axis=0).sample(frac=1, random_state=random_state).reset_index(drop=True)
    return data

# ---------------------------
# Предобработка
# ---------------------------
def preprocess_data(data):
    X = data.drop('label', axis=1)
    y = data['label'].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    return X_train, X_test, y_train, y_test, scaler

# ---------------------------
# Построение и обучение модели
# ---------------------------
def build_and_train_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    input_shape = X_train.shape[1]
    
    model = keras.Sequential([
        layers.Dense(32, activation='relu', input_shape=(input_shape,)),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    
    return model, history

# ---------------------------
# Функция для отображения истории обучения
# ---------------------------
def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Accuracy')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Loss')
    
    plt.show()

# ---------------------------
# Оценка модели
# ---------------------------
def evaluate_model(model, X_test, y_test):
    y_pred_probs = model.predict(X_test)
    y_pred = (y_pred_probs > 0.5).astype(int).ravel()
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    
    print("Accuracy:", acc)
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", report)
    
    return y_pred

# ---------------------------
# Генерация команды для Mikrotik
# ---------------------------
def block_ip_mikrotik(ip_address):
    cmd = f'/ip firewall filter add chain=input src-address={ip_address} action=drop comment="Blocked by ML"'
    print("[Mikrotik CMD] ", cmd)

# ---------------------------
# Главный запуск
# ---------------------------
def main():
    # 1) Генерация (или загрузка) данных
    data = generate_synthetic_data(num_samples=2000, ddos_ratio=0.3, random_state=42)
    
    # 2) Предобработка
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    # 3) Построение и обучение модели
    model, history = build_and_train_model(X_train, y_train, X_test, y_test, epochs=10, batch_size=32)
    
    # 4) Графики обучения
    plot_history(history)
    
    # 5) Оценка и метрики
    y_pred = evaluate_model(model, X_test, y_test)
    
    # 6) Пример применения "блокировки"
    #   (в реальности надо взять IP-адреса из логов, сопоставленных с X_test)
    fake_ips = [f"10.0.0.{i}" for i in range(1, len(y_pred)+1)]
    for ip, pred_label in zip(fake_ips, y_pred):
        if pred_label == 1:  # DDoS
            block_ip_mikrotik(ip)

# Запуск
main()
