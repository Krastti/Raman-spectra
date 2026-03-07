import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from scipy import interpolate
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
import os
import time

warnings.filterwarnings('ignore')


class FastRamanClassifier:
    """
    СУПЕР-БЫСТРЫЙ классификатор для Рамановских спектров
    Оптимизация занимает СЕКУНДЫ, а не часы!
    """

    def __init__(self,
                 algorithm='xgboost',  # xgboost, rf, svm, knn
                 target_length=500,
                 n_components=30,
                 use_smote=True,
                 random_state=42):

        self.algorithm = algorithm
        self.target_length = target_length
        self.n_components = n_components
        self.use_smote = use_smote
        self.random_state = random_state

        self.model = None
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=n_components)
        self.label_encoder = LabelEncoder()
        self.smote = SMOTE(random_state=random_state) if use_smote else None

        self.training_time = 0
        print(f"\n⚡ Инициализация: {algorithm.upper()} (SMOTE={'✓' if use_smote else '✗'})")

    def normalize_spectra(self, spectra):
        """Быстрая нормализация"""
        print(f"📏 Нормализация до {self.target_length} точек...")
        normalized = []
        x_new = np.linspace(0, 1, self.target_length)

        for spec in spectra:
            x_old = np.linspace(0, 1, len(spec))
            f = interpolate.interp1d(x_old, spec, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')
            normalized.append(f(x_new))

        return np.array(normalized)

    def load_data(self, filepath):
        """Быстрая загрузка"""
        print(f"\n📂 Загрузка {filepath}")

        # Читаем CSV
        df = pd.read_csv(filepath)

        # Определяем колонку с интенсивностью
        if 'Intensity_Filtered' in df.columns:
            intensity_col = 'Intensity_Filtered'
        else:
            intensity_col = 'Intensity'

        # Группировка
        spectra = []
        classes = []

        for (x, y), group in df.groupby(['X', 'Y']):
            group = group.sort_values('WAVE')
            spectra.append(group[intensity_col].values)
            classes.append(group['class'].iloc[0])

        print(f"✅ Загружено {len(spectra)} спектров")

        if len(spectra) == 0:
            return np.array([]), np.array([])

        X = self.normalize_spectra(spectra)
        y = np.array(classes)

        return X, y

    def prepare_features(self, X, y):
        """Быстрая подготовка"""
        print("\n🔧 Подготовка признаков...")

        y_encoded = self.label_encoder.fit_transform(y)

        # Масштабирование
        X_scaled = self.scaler.fit_transform(X)

        # PCA
        X_reduced = self.pca.fit_transform(X_scaled)
        explained = sum(self.pca.explained_variance_ratio_) * 100
        print(f"   PCA: {X_reduced.shape[1]} компонент ({explained:.1f}% дисперсии)")

        # SMOTE
        if self.use_smote and self.smote:
            print("⚡ Применяю SMOTE...")
            X_balanced, y_balanced = self.smote.fit_resample(X_reduced, y_encoded)
            print(f"   После SMOTE: {len(X_balanced)} образцов")
            return X_balanced, y_balanced

        return X_reduced, y_encoded

    def get_fast_model(self):
        """Мгновенная модель с оптимальными параметрами (без GridSearch)"""

        if self.algorithm == 'knn':
            return KNeighborsClassifier(
                n_neighbors=7,
                weights='distance',
                metric='cosine',
                n_jobs=-1
            )

        elif self.algorithm == 'svm':
            return SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=self.random_state
            )

        elif self.algorithm == 'rf':
            return RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=5,
                class_weight='balanced',
                n_jobs=-1,
                random_state=self.random_state
            )

        elif self.algorithm == 'xgboost':
            return XGBClassifier(
                n_estimators=200,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )

        else:
            raise ValueError(f"Неизвестный алгоритм: {self.algorithm}")

    def train(self, X, y):
        """Мгновенное обучение"""
        start_time = time.time()
        print(f"\n🚀 Обучение {self.algorithm.upper()}...")

        self.model = self.get_fast_model()
        self.model.fit(X, y)

        self.training_time = time.time() - start_time
        print(f"✅ Обучено за {self.training_time:.2f} секунд")

        # Важность признаков
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = self.model.feature_importances_

        return self.model

    def predict(self, X):
        """Быстрое предсказание"""
        return self.label_encoder.inverse_transform(self.model.predict(X))

    def predict_proba(self, X):
        """Вероятности"""
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test):
        """Быстрая оценка"""
        y_pred = self.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)

        print(f"\n{'=' * 50}")
        print(f"🎯 ТОЧНОСТЬ: {accuracy:.4f} ({accuracy * 100:.2f}%)")
        print(f"{'=' * 50}")

        print(f"\n📋 Отчет по классам:")
        print(classification_report(y_test, y_pred))

        return accuracy

    def plot_results(self, X_test, y_test):
        """Визуализация"""
        y_pred = self.predict(X_test)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Матрица ошибок
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.label_encoder.classes_,
                    yticklabels=self.label_encoder.classes_,
                    ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')

        # Важность признаков
        if hasattr(self, 'feature_importance'):
            axes[1].bar(range(min(20, len(self.feature_importance))),
                        self.feature_importance[:20])
            axes[1].set_title('Feature Importance (Top 20)')
            axes[1].set_xlabel('Feature Index')
            axes[1].set_ylabel('Importance')
        else:
            axes[1].text(0.5, 0.5, 'No feature importance',
                         ha='center', va='center')

        plt.tight_layout()
        plt.show()

    def save_model(self, filepath=None):
        """Сохранение"""
        if filepath is None:
            filepath = f"fast_model_{self.algorithm}.pkl"

        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'pca': self.pca,
            'encoder': self.label_encoder,
            'algorithm': self.algorithm
        }, filepath)
        print(f"💾 Модель сохранена: {filepath}")


# ============================================
# МГНОВЕННОЕ СРАВНЕНИЕ ВСЕХ АЛГОРИТМОВ
# ============================================

def fast_comparison(filepath):
    """Сравнивает все алгоритмы за МИНУТЫ вместо часов"""

    print("\n" + "=" * 60)
    print("⚡ СУПЕР-БЫСТРОЕ СРАВНЕНИЕ АЛГОРИТМОВ")
    print("=" * 60)

    algorithms = ['xgboost', 'rf', 'svm', 'knn']
    results = {}
    times = {}

    # Загружаем данные 1 раз
    print("\n📊 Загрузка данных...")
    base = FastRamanClassifier()
    X, y = base.load_data(filepath)

    if len(X) == 0:
        print("❌ Нет данных!")
        return

    # Разделяем на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    print(f"\n📊 Размер выборок: train={len(X_train)}, test={len(X_test)}")

    for algo in algorithms:
        print("\n" + "-" * 40)
        print(f"🧪 Тестирую: {algo.upper()}")
        print("-" * 40)

        start_time = time.time()

        # Создаем и обучаем
        clf = FastRamanClassifier(algorithm=algo, use_smote=True)
        X_train_prep, y_train_enc = clf.prepare_features(X_train, y_train)
        X_test_prep = clf.pca.transform(clf.scaler.transform(X_test))

        clf.train(X_train_prep, y_train_enc)

        # Оцениваем
        acc = clf.evaluate(X_test_prep, y_test)

        total_time = time.time() - start_time

        results[algo] = acc
        times[algo] = total_time

        print(f"⏱️  Время: {total_time:.1f} сек")

    # Финальное сравнение
    print("\n" + "=" * 60)
    print("🏆 РЕЗУЛЬТАТЫ (отсортированы по точности)")
    print("=" * 60)

    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    for algo, acc in sorted_results:
        print(f"{algo.upper():8}: {acc:.4f} ({acc * 100:.1f}%) ⏱️ {times[algo]:.1f} сек")

    best_algo = sorted_results[0][0]
    print(f"\n🎉 ЛУЧШИЙ: {best_algo.upper()} с точностью {results[best_algo]:.4f}")

    return results


# ============================================
# ОПТИМИЗАЦИЯ ТОЛЬКО ЛУЧШЕГО АЛГОРИТМА
# ============================================

def quick_optimize(filepath, algorithm='xgboost'):
    """Быстрая оптимизация одного алгоритма"""

    print(f"\n⚡ БЫСТРАЯ ОПТИМИЗАЦИЯ {algorithm.upper()}")

    # Загружаем
    X, y = FastRamanClassifier().load_data(filepath)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Пробуем разные параметры (только 2-3 варианта!)
    results = []

    if algorithm == 'xgboost':
        param_combinations = [
            {'n_estimators': 100, 'max_depth': 5, 'lr': 0.1},
            {'n_estimators': 200, 'max_depth': 7, 'lr': 0.1},
            {'n_estimators': 200, 'max_depth': 5, 'lr': 0.05},
        ]

        for params in param_combinations:
            print(f"\n📊 Тест: {params}")

            clf = FastRamanClassifier(algorithm='xgboost', use_smote=True)
            X_train_prep, y_train_enc = clf.prepare_features(X_train, y_train)
            X_test_prep = clf.pca.transform(clf.scaler.transform(X_test))

            # Создаем модель с текущими параметрами
            clf.model = XGBClassifier(
                n_estimators=params['n_estimators'],
                max_depth=params['max_depth'],
                learning_rate=params['lr'],
                random_state=42,
                use_label_encoder=False,
                eval_metric='logloss',
                verbosity=0
            )
            clf.model.fit(X_train_prep, y_train_enc)

            acc = clf.evaluate(X_test_prep, y_test)
            results.append((params, acc))

    # Лучший результат
    best_params, best_acc = max(results, key=lambda x: x[1])
    print(f"\n🎯 ЛУЧШИЕ ПАРАМЕТРЫ: {best_params}")
    print(f"🎯 ТОЧНОСТЬ: {best_acc:.4f} ({best_acc * 100:.2f}%)")

    return best_params


# ============================================
# ГОТОВЫЙ ПРОСТОЙ ЗАПУСК
# ============================================

if __name__ == "__main__":
    filepath = "filtered_data.csv"

    print("\n" + "=" * 60)
    print("⚡ БЫСТРЫЙ ML ДЛЯ РАМАНОВСКИХ СПЕКТРОВ")
    print("=" * 60)

    print("\n1. Супер-быстрое сравнение всех алгоритмов (2-3 минуты)")
    print("2. Быстрая оптимизация XGBoost (1 минута)")
    print("3. Просто обучить XGBoost (30 секунд)")
    print("4. Просто обучить Random Forest (30 секунд)")

    choice = input("\nТвой выбор (1-4): ").strip()

    if choice == '1':
        fast_comparison(filepath)

    elif choice == '2':
        quick_optimize(filepath, algorithm='xgboost')

    elif choice == '3':
        print("\n🚀 ОБУЧЕНИЕ XGBOOST...")
        clf = FastRamanClassifier(algorithm='xgboost', use_smote=True)
        X, y = clf.load_data(filepath)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_train_prep, y_train_enc = clf.prepare_features(X_train, y_train)
        X_test_prep = clf.pca.transform(clf.scaler.transform(X_test))
        clf.train(X_train_prep, y_train_enc)
        clf.evaluate(X_test_prep, y_test)
        clf.plot_results(X_test_prep, y_test)
        clf.save_model()

    elif choice == '4':
        print("\n🚀 ОБУЧЕНИЕ RANDOM FOREST...")
        clf = FastRamanClassifier(algorithm='rf', use_smote=True)
        X, y = clf.load_data(filepath)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        X_train_prep, y_train_enc = clf.prepare_features(X_train, y_train)
        X_test_prep = clf.pca.transform(clf.scaler.transform(X_test))
        clf.train(X_train_prep, y_train_enc)
        clf.evaluate(X_test_prep, y_test)
        clf.plot_results(X_test_prep, y_test)
        clf.save_model()

    else:
        print("❌ Неверный выбор!")