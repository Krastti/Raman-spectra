import os
import argparse
import time
import json
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier
import joblib
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ==========================================
# СЕТКИ ПАРАМЕТРОВ ДЛЯ GRID SEARCH
# ==========================================

PARAM_GRIDS = {
    'small': {
        'n_estimators': [100, 150],
        'learning_rate': [0.05, 0.1],
        'max_depth': [3, 4],
        'subsample': [0.8, 0.9]
    },
    'medium': {
        'n_estimators': [100, 150, 200],
        'learning_rate': [0.03, 0.05, 0.1],
        'max_depth': [3, 4, 5],
        'subsample': [0.7, 0.8, 0.9]
    },
    'large': {
        'n_estimators': [100, 200, 300, 500],
        'learning_rate': [0.01, 0.03, 0.05, 0.1],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.6, 0.7, 0.8, 0.9]
    }
}


def main():
    parser = argparse.ArgumentParser()
    default_root = os.path.join(os.getcwd(), "Хакatон_cleaned")

    # === Данные ===
    parser.add_argument("--data-root", default=default_root)
    parser.add_argument("--npy-x", default=None)
    parser.add_argument("--npy-y", default=None)
    parser.add_argument("--npy-files", default=None)
    parser.add_argument("--max-samples", type=int, default=0)

    # === Гиперпараметры модели ===
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--max-depth", type=int, default=4)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--early-stopping-rounds", type=int, default=20)

    # === Grid Search ===
    parser.add_argument("--use-grid-search", action="store_true",
                        help="Включить Grid Search для подбора параметров")
    parser.add_argument("--param-grid", type=str, default="medium",
                        choices=["small", "medium", "large"],
                        help="Размер сетки параметров")
    parser.add_argument("--cv-folds", type=int, default=5,
                        help="Количество фолдов для кросс-валидации")
    parser.add_argument("--scoring", type=str, default="f1_macro",
                        choices=["accuracy", "f1_macro", "f1_weighted", "roc_auc_ovr"],
                        help="Метрика для оптимизации")
    parser.add_argument("--n-jobs", type=int, default=-1,
                        help="Количество ядер CPU (-1 = все доступные)")

    # === Вывод ===
    parser.add_argument("--out-dir", default="artifacts")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    if not os.path.exists(args.data_root):
        raise SystemExit(f"Папка с данными не обнаружена: {args.data_root}")

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
            from data_loader import TxtDataset
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
    if args.max_samples and 0 < args.max_samples < len(X):
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
        df = DataFrame({
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
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    os.makedirs(args.out_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.joblib"))

    print(f"⏱ Нормализация: {time.time() - t2:.2f} сек")

    # ==========================================
    # 5. ВЕСА КЛАССОВ
    # ==========================================
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1)
    class_weights = class_weights / class_weights.sum() * len(class_counts)
    sample_weights = np.array([class_weights[label] for label in y_train])

    # ==========================================
    # 6. МОДЕЛЬ И ОБУЧЕНИЕ
    # ==========================================
    t3 = time.time()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.use_grid_search:
        # ==================== GRID SEARCH ====================
        print(f"\n🔍 Grid Search: {args.param_grid} сетка, {args.cv_folds}-fold CV")
        print("=" * 80)

        param_grid = PARAM_GRIDS[args.param_grid]

        base_model = GradientBoostingClassifier(
            random_state=SEED,
            validation_fraction=0.1,
            n_iter_no_change=args.early_stopping_rounds,
            verbose=1 if args.verbose else 0
        )

        cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=SEED)

        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring=args.scoring,
            n_jobs=args.n_jobs,
            verbose=2 if args.verbose else 1,
            return_train_score=True
        )

        grid_search.fit(X_train_scaled, y_train, sample_weight=sample_weights)

        # Сохранение лучших параметров
        best_params = grid_search.best_params_
        with open(os.path.join(args.out_dir, "best_params.json"), "w", encoding="utf-8") as f:
            json.dump(best_params, f, indent=2, ensure_ascii=False)

        print(f"\n✅ Лучшие параметры:")
        for k, v in best_params.items():
            print(f"   {k}: {v}")

        print(f"\n📊 Лучшая метрика ({args.scoring}): {grid_search.best_score_:.4f}")

        model = grid_search.best_estimator_

        # Лог результатов Grid Search
        results_df = DataFrame(grid_search.cv_results_)
        results_df.to_csv(os.path.join(args.out_dir, "grid_search_results.csv"), index=False)
        print(f"💾 Результаты Grid Search сохранены в {args.out_dir}/grid_search_results.csv")

    else:
        # ==================== ОБЫЧНОЕ ОБУЧЕНИЕ ====================
        print(f"\n🚀 Обучение Gradient Boosting: {args.n_estimators} деревьев")
        print("=" * 80)

        model = GradientBoostingClassifier(
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            max_depth=args.max_depth,
            subsample=args.subsample,
            random_state=SEED,
            validation_fraction=0.1,
            n_iter_no_change=args.early_stopping_rounds,
            verbose=1
        )

        model.fit(X_train_scaled, y_train, sample_weight=sample_weights)

        # Сохранение параметров
        params = {
            'n_estimators': args.n_estimators,
            'learning_rate': args.learning_rate,
            'max_depth': args.max_depth,
            'subsample': args.subsample
        }
        with open(os.path.join(args.out_dir, "model_params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)

    print(f"⏱ Обучение завершено: {time.time() - t3:.2f} сек")

    # ==========================================
    # 7. СОХРАНЕНИЕ И ОЦЕНКА
    # ==========================================
    joblib.dump(model, os.path.join(args.out_dir, "best_model.joblib"))
    print(f"💾 Модель сохранена в {args.out_dir}/best_model.joblib")

    # Предсказания на валидации
    y_pred = model.predict(X_val_scaled)

    print("\n📊 Classification Report на валидации:")
    class_names = ['control', 'endo', 'exo']
    print(classification_report(y_val, y_pred, target_names=class_names))

    # Важность признаков
    if hasattr(model, 'feature_importances_'):
        print("\n🔍 Важность признаков (топ-10):")
        indices = np.argsort(model.feature_importances_)[::-1][:10]
        for i in indices:
            print(f"   Признак {i}: {model.feature_importances_[i]:.4f}")

        # Сохранение важности признаков
        np.save(os.path.join(args.out_dir, "feature_importances.npy"), model.feature_importances_)

    # ==========================================
    # 8. ИТОГОВЫЙ ОТЧЁТ
    # ==========================================
    report = {
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'classes': class_names,
        'accuracy': float((y_pred == y_val).sum() / len(y_val)),
        'grid_search_used': args.use_grid_search,
        'best_params': model.get_params() if args.use_grid_search else params
    }

    with open(os.path.join(args.out_dir, "final_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"💾 Итоговый отчёт сохранён в {args.out_dir}/final_report.json")


if __name__ == "__main__":
    main()