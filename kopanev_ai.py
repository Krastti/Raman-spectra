import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from scipy import interpolate
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import keras
from keras import layers, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras import models
from keras.utils import to_categorical
import joblib
import warnings
import os
import time
from tqdm import tqdm  # для прогресс-баров

warnings.filterwarnings('ignore')


class FastRamanNeuralNetwork:
    """
    ОПТИМИЗИРОВАННАЯ нейросеть для Рамановских спектров
    """

    def __init__(self,
                 target_length=500,
                 n_components=30,
                 use_smote=True,
                 cache_dir='cache',
                 random_state=42):

        self.target_length = target_length
        self.n_components = n_components
        self.use_smote = use_smote
        self.cache_dir = cache_dir
        self.random_state = random_state

        # Создаем папку для кеша
        os.makedirs(cache_dir, exist_ok=True)

        # Компоненты предобработки
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.label_encoder = LabelEncoder()
        self.smote = SMOTE(random_state=random_state) if use_smote else None

        # Модель
        self.model = None
        self.history = None
        self.classes_ = None
        self.n_classes = 0

        print(f"\n⚡ Оптимизированная нейросеть")
        print(f"   Входные признаки: {n_components} (после PCA)")
        print(f"   SMOTE: {'✓' if use_smote else '✗'}")
        print(f"   Кеш: {cache_dir}")

    def normalize_spectra_fast(self, spectra):
        """Быстрая нормализация через векторизацию"""
        print("📏 Нормализация длины...")

        x_new = np.linspace(0, 1, self.target_length)
        normalized = np.zeros((len(spectra), self.target_length))

        for i, spec in enumerate(tqdm(spectra, desc="Нормализация")):
            x_old = np.linspace(0, 1, len(spec))
            f = interpolate.interp1d(x_old, spec, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
            normalized[i] = f(x_new)

        return normalized

    def load_data_cached(self, filepath):
        """Загрузка с кешированием"""
        cache_file = os.path.join(self.cache_dir, 'preprocessed_data.npz')

        # Если есть кеш - загружаем
        if os.path.exists(cache_file):
            print(f"📦 Загрузка из кеша: {cache_file}")
            data = np.load(cache_file, allow_pickle=True)
            return data['X'], data['y']

        # Иначе обрабатываем
        print(f"\n📂 Загрузка {filepath}")
        df = pd.read_csv(filepath)

        # Определяем колонку
        intensity_col = 'Intensity_Filtered' if 'Intensity_Filtered' in df.columns else 'Intensity'

        # Группировка
        spectra = []
        classes = []

        for (x, y), group in tqdm(df.groupby(['X', 'Y']), desc="Группировка"):
            group = group.sort_values('WAVE')
            spectra.append(group[intensity_col].values)
            classes.append(group['class'].iloc[0])

        print(f"✅ Загружено {len(spectra)} спектров")

        # Нормализация
        X = self.normalize_spectra_fast(spectra)
        y = np.array(classes)

        # Сохраняем в кеш
        np.savez_compressed(cache_file, X=X, y=y)
        print(f"💾 Кеш сохранен: {cache_file}")

        return X, y

    def prepare_features(self, X_train, y_train, X_val=None, X_test=None):
        """
        Подготовка признаков ТОЛЬКО на train данных
        """
        print("\n🔧 Подготовка признаков...")

        # Кодирование меток
        y_train_enc = self.label_encoder.fit_transform(y_train)
        self.classes_ = self.label_encoder.classes_
        self.n_classes = len(self.classes_)

        # One-hot
        y_train_onehot = to_categorical(y_train_enc)

        # Масштабирование (fit только на train)
        X_train_scaled = self.scaler.fit_transform(X_train)

        # PCA (fit только на train)
        X_train_reduced = self.pca.fit_transform(X_train_scaled)

        explained = sum(self.pca.explained_variance_ratio_) * 100
        print(f"   PCA: {X_train_reduced.shape[1]} компонент ({explained:.1f}% дисперсии)")

        # SMOTE только на train!
        if self.use_smote and self.smote:
            print("⚡ Применяю SMOTE...")
            X_train_balanced, y_train_balanced_enc = self.smote.fit_resample(
                X_train_reduced, y_train_enc
            )
            y_train_balanced_onehot = to_categorical(y_train_balanced_enc)
            print(f"   После SMOTE: {len(X_train_balanced)} образцов")

            # Обработка validation/test
            X_val_prep = None
            X_test_prep = None

            if X_val is not None:
                X_val_prep = self.pca.transform(self.scaler.transform(X_val))
            if X_test is not None:
                X_test_prep = self.pca.transform(self.scaler.transform(X_test))

            return (X_train_balanced, y_train_balanced_onehot, y_train_balanced_enc,
                    X_val_prep, X_test_prep)
        else:
            X_val_prep = None
            X_test_prep = None

            if X_val is not None:
                X_val_prep = self.pca.transform(self.scaler.transform(X_val))
            if X_test is not None:
                X_test_prep = self.pca.transform(self.scaler.transform(X_test))

            return (X_train_reduced, y_train_onehot, y_train_enc,
                    X_val_prep, X_test_prep)

    def build_model(self, input_dim):
        """Оптимизированная архитектура"""

        model = models.Sequential([
            layers.Input(shape=(input_dim,)),

            # Блок 1
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Блок 2
            layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.BatchNormalization(),
            layers.Dropout(0.3),

            # Блок 3
            layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),

            layers.Dense(self.n_classes, activation='softmax')
        ])

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        return model

    def train(self, X_train, y_train_onehot, X_val=None, y_val_onehot=None,
              epochs=100, batch_size=32):
        """Оптимизированное обучение"""

        print(f"\n🚀 ОБУЧЕНИЕ")

        # Создаем модель
        self.model = self.build_model(X_train.shape[1])

        # Callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss' if X_val is not None else 'loss',
                          patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss' if X_val is not None else 'loss',
                              factor=0.5, patience=5, min_lr=1e-6)
        ]

        # Обучение
        start = time.time()

        self.history = self.model.fit(
            X_train, y_train_onehot,
            validation_data=(X_val, y_val_onehot) if X_val is not None else None,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        print(f"\n✅ Обучено за {time.time() - start:.1f} сек")
        return self.history

    def predict(self, X, return_proba=False):
        """Быстрое предсказание"""
        proba = self.model.predict(X, verbose=0)

        if return_proba:
            return proba
        return self.label_encoder.inverse_transform(np.argmax(proba, axis=1))

    def evaluate(self, X_test, y_test):
        """Оценка"""
        y_pred = self.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"\n{'=' * 50}")
        print(f"🎯 ТОЧНОСТЬ: {acc:.4f} ({acc * 100:.2f}%)")
        print(f"{'=' * 50}")
        print(f"\n📋 Отчет:\n{classification_report(y_test, y_pred)}")

        return acc

    def plot_results(self, X_test, y_test):
        """Визуализация"""
        y_pred = self.predict(X_test)

        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.classes_,
                    yticklabels=self.classes_,
                    ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')

        # История
        if self.history:
            axes[0, 1].plot(self.history.history['accuracy'], label='Train')
            if 'val_accuracy' in self.history.history:
                axes[0, 1].plot(self.history.history['val_accuracy'], label='Val')
            axes[0, 1].legend()
            axes[0, 1].grid(True)

            axes[1, 0].plot(self.history.history['loss'], label='Train')
            if 'val_loss' in self.history.history:
                axes[1, 0].plot(self.history.history['val_loss'], label='Val')
            axes[1, 0].legend()
            axes[1, 0].grid(True)

        # Уверенность
        proba = self.predict(X_test, return_proba=True)
        axes[1, 1].hist(np.max(proba, axis=1), bins=20)
        axes[1, 1].set_title('Confidence Distribution')
        axes[1, 1].grid(True)

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath='raman_nn'):
        """Сохранение"""
        # Сохраняем веса и параметры
        self.model.save(f'{filepath}.h5')

        # Сохраняем препроцессоры
        joblib.dump({
            'scaler': self.scaler,
            'pca': self.pca,
            'encoder': self.label_encoder,
            'classes': self.classes_,
            'n_components': self.n_components
        }, f'{filepath}_preprocessors.pkl')

        print(f"💾 Модель сохранена: {filepath}.h5")


# ============================================
# БЫСТРЫЙ ЗАПУСК
# ============================================

def main():
    filepath = "filtered_data.csv"

    print("=" * 60)
    print("⚡ БЫСТРАЯ НЕЙРОСЕТЬ ДЛЯ РАМАНОВСКИХ СПЕКТРОВ")
    print("=" * 60)

    # 1. Загружаем данные (с кешем)
    nn = FastRamanNeuralNetwork(
        target_length=500,
        n_components=30,
        use_smote=True,
        cache_dir='nn_cache'
    )

    X, y = nn.load_data_cached(filepath)

    # 2. Разделяем
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\n📊 Размеры: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # 3. Подготовка признаков
    (X_train_prep, y_train_onehot, y_train_enc,
     X_val_prep, X_test_prep) = nn.prepare_features(X_train, y_train, X_val, X_test)

    # 4. Обучение
    y_val_onehot = to_categorical(nn.label_encoder.transform(y_val)) if X_val_prep is not None else None

    nn.train(
        X_train_prep, y_train_onehot,
        X_val_prep, y_val_onehot,
        epochs=100,
        batch_size=32
    )

    # 5. Оценка
    nn.evaluate(X_test_prep, y_test)
    nn.plot_results(X_test_prep, y_test)
    nn.save_model('optimized_raman_nn')

    # 6. Сравнение с XGBoost
    try:
        from xgboost import XGBClassifier

        print("\n" + "=" * 50)
        print("📊 СРАВНЕНИЕ С XGBOOST")
        print("=" * 50)

        xgb = XGBClassifier(n_estimators=200, max_depth=6, random_state=42)
        xgb.fit(X_train_prep, y_train_enc)
        xgb_pred = nn.label_encoder.inverse_transform(xgb.predict(X_test_prep))
        xgb_acc = accuracy_score(y_test, xgb_pred)

        nn_acc = accuracy_score(y_test, nn.predict(X_test_prep))

        print(f"XGBoost:  {xgb_acc:.4f}")
        print(f"NeuralNet: {nn_acc:.4f}")
        print(f"Improvement: {((nn_acc / xgb_acc) - 1) * 100:+.1f}%")

    except Exception as e:
        print(f"XGBoost сравнение: {e}")


if __name__ == "__main__":
    main()