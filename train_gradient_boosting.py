# train.py
import os
import argparse
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import random

from data_loader import TxtDataset

# ==========================================
# ФИКСАЦИЯ СЛУЧАЙНОСТИ (SEED)
# ==========================================
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

def main():
    parser = argparse.ArgumentParser()
    default_root = os.path.join(os.getcwd(), "Хакatон_cleaned")
    parser.add_argument("--data-root", default=default_root)
    parser.add_argument("--n-estimators", type=int, default=500)      # ← вместо epochs
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--max-depth", type=int, default=6)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--npy-x", default=None)
    parser.add_argument("--npy-y", default=None)
    parser.add_argument("--npy-files", default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)  # ← для GBM
    args = parser.parse_args()

    if not os.path.exists(args.data_root):
        raise SystemExit(f"Data root not found: {args.data_root}")

    # ==========================================
    # 1. ЗАГРУЗКА ДАННЫХ
    # ==========================================
    t0 = time.time()
    
    if args.npy_x and os.path.exists(args.npy_x):
        print(f"Loading X from {args.npy_x}")
        X = np.load(args.npy_x)
        y = np.load(args.npy_y) if args.npy_y else np.load(os.path.join(os.path.dirname(args.npy_x), "y.npy"))
        files_per_row = np.load(args.npy_files) if args.npy_files else None
    else:
        cache_dir = os.path.join(args.data_root, ".cache")
        cache_X = os.path.join(cache_dir, "X.npy")
        cache_y = os.path.join(cache_dir, "y.npy")
        cache_files = os.path.join(cache_dir, "files.npy")

        if os.path.exists(cache_X) and os.path.exists(cache_y):
            print(f"Loading from cache: {cache_X}")
            X = np.load(cache_X)
            y = np.load(cache_y)
            files_per_row = np.load(cache_files) if os.path.exists(cache_files) else None
        else:
            print("Parsing .txt files (one-time)...")
            ds = TxtDataset(args.data_root)
            X, y, files_per_row = ds.get_all(return_files=True)
            
            if X.size == 0:
                raise SystemExit("No samples found!")
            
            os.makedirs(cache_dir, exist_ok=True)
            np.save(cache_X, X)
            np.save(cache_y, y)
            np.save(cache_files, files_per_row)
            print(f"Saved cache to {cache_dir}")

    print(f"⏱ Загрузка данных: {time.time() - t0:.2f} сек")
    print(f"📊 Всего образцов: {len(X):,}")

    # ==========================================
    # 2. ОБРЕЗКА ДАННЫХ
    # ==========================================
    if args.max_samples and args.max_samples > 0 and len(X) > args.max_samples:
        print(f"✂️ Обрезка данных: {len(X):,} → {args.max_samples:,} образцов")
        X = X[:args.max_samples]
        y = y[:args.max_samples]
        if files_per_row is not None:
            files_per_row = files_per_row[:args.max_samples]

    # ==========================================
    # 3. СПЛИТ ПО ФАЙЛАМ
    # ==========================================
    t1 = time.time()
    
    if files_per_row is not None:
        df = pd.DataFrame({
            'file': files_per_row,
            'label': y,
            'idx': np.arange(len(y))
        })
        
        file_labels = df.groupby('file')['label'].first()
        file_list = file_labels.index.tolist()
        labels_list = file_labels.values.tolist()
        
        train_files, val_files = train_test_split(
            file_list, test_size=0.2, random_state=42, stratify=labels_list
        )
        
        train_mask = df['file'].isin(train_files).values
        val_mask = df['file'].isin(val_files).values
        
        train_idx = df.loc[train_mask, 'idx'].values
        val_idx = df.loc[val_mask, 'idx'].values
        
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        
        print(f"📁 Сплит по файлам: {len(train_files)} файлов → {len(X_train)} образцов")
        print(f"📁 Сплит по файлам: {len(val_files)} файлов → {len(X_val)} образцов")
    else:
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print("⚠️ Сплит по строкам (нет mapping файлов)")

    print(f"⏱ Сплит данных: {time.time() - t1:.2f} сек")

    # ==========================================
    # 4. НОРМАЛИЗАЦИЯ
    # ==========================================
    t2 = time.time()
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.joblib"))

    print(f"⏱ Нормализация: {time.time() - t2:.2f} сек")

    # ==========================================
    # 5. МОДЕЛЬ: Gradient Boosting
    # ==========================================
    print(f"\n🚀 Обучение Gradient Boosting: {args.n_estimators} деревьев")
    print("="*80)

    # Веса для несбалансированных классов (через sample_weight)
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    sample_weights = np.array([class_weights[label] for label in y_train])

    model = GradientBoostingClassifier(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        random_state=SEED,
        validation_fraction=0.1,  # для early stopping
        n_iter_no_change=args.early_stopping_rounds,
        verbose=1
    )

    # ==========================================
    # 6. ОБУЧЕНИЕ
    # ==========================================
    t3 = time.time()
    model.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"⏱ Обучение завершено: {time.time() - t3:.2f} сек")

    # ==========================================
    # 7. СОХРАНЕНИЕ И ОЦЕНКА
    # ==========================================
    # Сохраняем лучшую модель (GBM сам выбирает итерацию при early stopping)
    joblib.dump(model, os.path.join(args.out_dir, "best_model.joblib"))
    print(f"💾 Модель сохранена в {args.out_dir}/best_model.joblib")

    # Предсказания на валидации
    y_pred = model.predict(X_val)
    
    print("\n📊 Classification Report на валидации:")
    class_names = ['control', 'endo', 'exo']
    print(classification_report(y_val, y_pred, target_names=class_names))

    # Опционально: важность признаков
    if hasattr(model, 'feature_importances_'):
        print("\n🔍 Важность признаков:")
        for i, imp in enumerate(model.feature_importances_):
            print(f"  Признак {i}: {imp:.4f}")

if __name__ == "__main__":
    main()