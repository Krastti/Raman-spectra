import os
import sys
import glob
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, colorchooser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from scipy.signal import savgol_filter, medfilt
from scipy.ndimage import uniform_filter1d
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
from datetime import datetime
import copy

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

import warnings
warnings.filterwarnings('ignore')

# Совместимость с NumPy 2.0+
if hasattr(np, 'trapezoid'):
    def numpy_trapz(y, x):
        return np.trapezoid(y, x)
else:
    def numpy_trapz(y, x):
        return np.trapz(y, x)

CONFIG = {
    'smooth_window': 11,
    'smooth_polyorder': 3,
    'baseline_window': 100,
    'normalize_method': 'area',
    'wave_range': (2400, 3300),
    'n_estimators': 200,
    'max_depth': 15,
    'test_size': 0.2,
    'cv_folds': 5,
    'random_state': 42,
    'dpi': 150,
}

# Цвета по умолчанию для классов
DEFAULT_COLORS = {
    'control': '#2ecc71',
    'endo': '#3498db',
    'exo': '#e74c3c',
    'unknown': '#95a5a6'
}


def load_spectrum(filepath):
    """Загрузка спектра из txt-файла"""
    try:
        if not os.path.exists(filepath):
            return None, "Файл не существует"
        if os.path.getsize(filepath) == 0:
            return None, "Файл пустой"

        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        data_lines = [l.strip() for l in lines if l.strip() and not l.strip().startswith('#')]

        if len(data_lines) < 5:
            return None, f"Мало данных: {len(data_lines)} строк"

        sep = '\t' if '\t' in data_lines[0] else r'\s+'

        df = pd.read_csv(
            filepath, sep=sep, comment='#', header=None,
            names=['X', 'Y', 'Wave', 'Intensity'],
            engine='python', on_bad_lines='skip',
            encoding='utf-8', skip_blank_lines=True
        )

        df = df.dropna(subset=['Wave', 'Intensity'])
        df['Wave'] = pd.to_numeric(df['Wave'], errors='coerce')
        df['Intensity'] = pd.to_numeric(df['Intensity'], errors='coerce')
        df = df.dropna(subset=['Wave', 'Intensity'])
        df = df.sort_values('Wave').reset_index(drop=True)

        if len(df) < 20:
            return None, f"Мало точек: {len(df)}"

        return df, None
    except Exception as e:
        return None, str(e)


def extract_label_from_path(filepath):
    """Извлечение метки из пути"""
    path_lower = filepath.lower()
    if 'control' in path_lower or 'контр' in path_lower:
        return 'control'
    elif 'endo' in path_lower or 'эндо' in path_lower:
        return 'endo'
    elif 'exo' in path_lower or 'экзо' in path_lower:
        return 'exo'
    return 'unknown'


# ПРЕДОБРАБОТКА
def preprocess_spectrum(df, config=CONFIG, apply_smooth=False, apply_baseline=False, apply_normalize=False):
    """Предобработка спектра"""
    wave = df['Wave'].values.astype(float).copy()
    intensity = df['Intensity'].values.astype(float).copy()

    if config.get('wave_range'):
        w_min, w_max = config['wave_range']
        mask = (wave >= w_min) & (wave <= w_max)
        wave = wave[mask]
        intensity = intensity[mask]

    if len(wave) < 10:
        return wave, intensity

    if apply_smooth:
        intensity = medfilt(intensity, kernel_size=3)
        window = min(config['smooth_window'], len(intensity) // 2 * 2 + 1)
        if window % 2 == 0:
            window += 1
        if window >= 3:
            intensity = savgol_filter(intensity, window, config['smooth_polyorder'])

    if apply_baseline:
        bl_window = min(config['baseline_window'], len(intensity) // 4)
        if bl_window > 1:
            baseline = uniform_filter1d(intensity, size=bl_window, mode='nearest')
            bl_smooth = min(51, len(baseline) // 10 * 2 + 1)
            if bl_smooth % 2 == 0:
                bl_smooth += 1
            if bl_smooth >= 3:
                baseline = savgol_filter(baseline, bl_smooth, 3)
            intensity = intensity - baseline + np.min(baseline)
            intensity = np.clip(intensity, 0, None)

    if apply_normalize:
        method = config['normalize_method']
        if method == 'area':
            area = numpy_trapz(intensity, wave)
            if area > 0:
                intensity = intensity / area
        elif method == 'max':
            intensity = intensity / (np.max(intensity) + 1e-10)

    return wave, intensity


def interpolate_to_common_wave(spectra, config=CONFIG):
    """Интерполяция на общую сетку"""
    try:
        if not spectra:
            return None, []

        all_waves = np.concatenate([df['Wave'].values for df in spectra])
        w_min, w_max = np.percentile(all_waves, [1, 99])

        if config.get('wave_range'):
            w_min = max(w_min, config['wave_range'][0])
            w_max = min(w_max, config['wave_range'][1])

        common_wave = np.arange(w_min, w_max, 1.0)

        interpolated = []
        for df in spectra:
            wave = df['Wave'].values
            intensity = df['Intensity'].values
            intensity_interp = np.interp(common_wave, wave, intensity)
            interpolated.append(intensity_interp)

        return common_wave, np.array(interpolated)
    except Exception as e:
        print(f"⚠ Ошибка интерполяции: {e}")
        return None, []


# ОБУЧЕНИЕ
def train_classification_model(X, y, config=CONFIG):
    """Обучение модели"""
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    class_names = le.classes_

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_enc, test_size=config['test_size'],
        stratify=y_enc, random_state=config['random_state']
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        random_state=config['random_state'],
        n_jobs=-1,
        class_weight='balanced'
    )

    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_test_s)

    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'class_names': class_names.tolist(),
    }

    cv = StratifiedKFold(n_splits=config['cv_folds'], shuffle=True, random_state=config['random_state'])
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=cv, scoring='f1_weighted')
    metrics['cv_f1_mean'] = cv_scores.mean()
    metrics['cv_f1_std'] = cv_scores.std()

    importances = model.feature_importances_

    return {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'metrics': metrics,
        'importances': importances,
        'y_test': y_test,
        'y_pred': y_pred,
    }


# GUI ПРИЛОЖЕНИЕ
class RamanAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🔬 Рамановский Анализатор v7.0")
        self.root.geometry("1600x1000")

        # Данные
        self.spectra = []
        self.original_spectra = []
        self.labels = []
        self.filepaths = []
        self.processed_spectra = []
        self.custom_colors = {}
        self.wave_axis = None
        self.X_matrix = None
        self.ml_results = None
        self.current_index = 0

        # Настройки
        self.show_all = tk.BooleanVar(value=False)
        self.apply_smooth = tk.BooleanVar(value=False)
        self.apply_baseline = tk.BooleanVar(value=False)
        self.apply_normalize = tk.BooleanVar(value=False)

        self.log_messages = []

        self.create_ui()
        self.show_welcome()
        self.log("✅ Запущено | Обработка выключена")

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        full_msg = f"[{timestamp}] {message}"
        self.log_messages.append(full_msg)
        print(full_msg)
        self.statusbar.config(text=message)

    def create_ui(self):
        panel = ttk.Frame(self.root, width=400)
        panel.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        panel.pack_propagate(False)

        # Кнопки
        ttk.Button(panel, text="📁 Открыть файл", command=self.load_file).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="📂 Открыть папку", command=self.load_folder).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="🔄 Предобработать", command=self.preprocess_all).pack(fill=tk.X, pady=3)
        ttk.Button(panel, text="🤖 Обучить модель", command=self.train_model).pack(fill=tk.X, pady=3)

        # Сброс
        ttk.Button(panel, text="🔁 Сбросить всё", command=self.reset_all,
                   style='Accent.TButton').pack(fill=tk.X, pady=3)

        ttk.Button(panel, text="🗑️ Очистить", command=self.clear_all).pack(fill=tk.X, pady=3)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Режим отображения
        ttk.Label(panel, text="📊 Режим", font=('Arial', 11, 'bold')).pack(pady=3)
        ttk.Radiobutton(panel, text="Все спектры", variable=self.show_all, value=True, command=self.update_plot).pack(
            anchor=tk.W)
        ttk.Radiobutton(panel, text="По одному", variable=self.show_all, value=False, command=self.update_plot).pack(
            anchor=tk.W)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Выбор файла
        ttk.Label(panel, text="📋 Файл", font=('Arial', 11, 'bold')).pack(pady=3)
        self.file_var = tk.StringVar()
        self.file_combo = ttk.Combobox(panel, textvariable=self.file_var, state='readonly', height=15)
        self.file_combo.pack(fill=tk.X, pady=3)
        self.file_combo.bind('<<ComboboxSelected>>', lambda e: self.on_file_select())

        nav_frame = ttk.Frame(panel)
        nav_frame.pack(fill=tk.X, pady=3)
        ttk.Button(nav_frame, text="⏮️", command=self.prev_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(nav_frame, text="⏭️", command=self.next_file).pack(side=tk.LEFT, padx=2)

        self.index_label = ttk.Label(panel, text="Файл: 0/0", font=('Consolas', 10))
        self.index_label.pack(pady=3)

        # Выбор цвета
        color_frame = ttk.Frame(panel)
        color_frame.pack(fill=tk.X, pady=5)
        ttk.Label(color_frame, text="🎨 Цвет:").pack(side=tk.LEFT)
        ttk.Button(color_frame, text="Выбрать", command=self.choose_color).pack(side=tk.LEFT, padx=5)
        self.color_preview = tk.Canvas(color_frame, width=30, height=20, bg='#95a5a6')
        self.color_preview.pack(side=tk.LEFT, padx=5)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Обработка
        ttk.Label(panel, text="⚙️ Обработка", font=('Arial', 11, 'bold')).pack(pady=3)
        ttk.Checkbutton(panel, text="☐ Сглаживание", variable=self.apply_smooth).pack(anchor=tk.W)
        ttk.Checkbutton(panel, text="☐ Базовая линия", variable=self.apply_baseline).pack(anchor=tk.W)
        ttk.Checkbutton(panel, text="☐ Нормировка", variable=self.apply_normalize).pack(anchor=tk.W)

        ttk.Label(panel, text="→ Нажмите 'Предобработать' для применения",
                  font=('Arial', 9), foreground='gray').pack(anchor=tk.W, pady=5)

        ttk.Separator(panel, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        # Статистика
        ttk.Label(panel, text="📈 Статистика", font=('Arial', 11, 'bold')).pack(pady=3)
        self.stats_label = ttk.Label(panel, text="Файлов: 0", justify=tk.LEFT, font=('Consolas', 10))
        self.stats_label.pack(anchor=tk.W, pady=3)

        # График
        graph_frame = ttk.Frame(self.root)
        graph_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, graph_frame)
        toolbar.update()

        # Статус
        self.statusbar = ttk.Label(self.root, text="Готов", relief=tk.SUNKEN, anchor=tk.W)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    def show_welcome(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5,
                     'Загрузите файлы .txt\n\nФормат: X Y Wave Intensity\n\n Можно менять цвет каждого спектра\n Кнопка сброса возвращает к оригиналу',
                     ha='center', va='center', fontsize=13, transform=self.ax.transAxes,
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        self.ax.set_xlim(0, 1)
        self.ax.set_ylim(0, 1)
        self.ax.axis('off')
        self.canvas.draw()

    def load_file(self):
        files = filedialog.askopenfilenames(filetypes=[("Text", "*.txt")])
        if files:
            self.add_files(list(files))

    def load_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            files = glob.glob(os.path.join(folder, '**', '*.txt'), recursive=True)
            if files:
                self.add_files(files)

    def add_files(self, files):
        loaded = 0
        start_index = len(self.spectra)

        for filepath in files:
            df, error = load_spectrum(filepath)
            if df is not None:
                label = extract_label_from_path(filepath)

                # Сохраняем оригинальные данные для сброса
                self.original_spectra.append(copy.deepcopy(df))
                self.spectra.append(df)
                self.labels.append(label)
                self.filepaths.append(filepath)

                # Устанавливаем цвет по умолчанию для класса
                self.custom_colors[len(self.spectra) - 1] = DEFAULT_COLORS.get(label, '#95a5a6')

                loaded += 1
                self.log(f"✅ {os.path.basename(filepath)} ({len(df)} точек)")
            else:
                self.log(f"❌ {os.path.basename(filepath)}: {error}")

        self.file_combo['values'] = [f"{i + 1}. {self.labels[i]}: {os.path.basename(f)}" for i, f in
                                     enumerate(self.filepaths)]

        if loaded > 0:
            self.current_index = start_index
            self.file_combo.current(self.current_index)
            self.update_stats()
            self.update_plot()

        self.log(f"📊 Загружено: {loaded}")

    def clear_all(self):
        """Полная очистка всех данных"""
        self.spectra = []
        self.original_spectra = []
        self.labels = []
        self.filepaths = []
        self.processed_spectra = []
        self.custom_colors = {}
        self.wave_axis = None
        self.X_matrix = None
        self.ml_results = None
        self.current_index = 0
        self.file_combo['values'] = []
        self.file_combo.set('')
        self.stats_label.config(text="Файлов: 0")
        self.index_label.config(text="Файл: 0/0")
        self.show_welcome()
        self.log("🗑️ Все данные удалены")

    def reset_all(self):
        """
        СБРОС ВСЕХ ИЗМЕНЕНИЙ
        Возвращает спектры к оригинальному состоянию
        """
        if not self.spectra:
            messagebox.showinfo("Сброс", "Нет данных для сброса")
            return

        result = messagebox.askyesno(
            "Сброс всех изменений",
            f"Вернуть все {len(self.spectra)} спектров к исходному состоянию?\n\n"
            "Будет сброшено:\n"
            "• Предобработка (сглаживание, базовая линия, нормировка)\n"
            "• Пользовательские цвета\n"
            "• Результаты ML модели"
        )

        if not result:
            return

        # Восстанавливаем оригинальные данные
        self.spectra = [copy.deepcopy(df) for df in self.original_spectra]
        self.processed_spectra = []
        self.wave_axis = None
        self.X_matrix = None
        self.ml_results = None

        # Сбрасываем цвета к умолчанию
        self.custom_colors = {}
        for i, label in enumerate(self.labels):
            self.custom_colors[i] = DEFAULT_COLORS.get(label, '#95a5a6')

        # Сбрасываем настройки обработки
        self.apply_smooth.set(False)
        self.apply_baseline.set(False)
        self.apply_normalize.set(False)

        self.current_index = 0
        if self.file_combo['values']:
            self.file_combo.current(0)

        self.update_stats()
        self.update_plot()

        self.log("🔁 Все изменения сброшены к оригиналу")
        messagebox.showinfo("Сброс", "✅ Все спектры возвращены к исходному состоянию")

    def choose_color(self):
        """Выбор цвета для текущего спектра"""
        if not self.spectra:
            messagebox.showwarning("Внимание", "Нет загруженных спектров")
            return

        # Текущий цвет
        current_color = self.custom_colors.get(self.current_index,
                                               DEFAULT_COLORS.get(self.labels[self.current_index], '#95a5a6'))

        # Диалог выбора цвета
        color = colorchooser.askcolor(color=current_color, title="Выберите цвет спектра")

        if color[1]:  # Если цвет выбран
            self.custom_colors[self.current_index] = color[1]
            self.color_preview.config(bg=color[1])
            self.update_plot()
            self.log(f"🎨 Цвет спектра {self.current_index + 1} изменён на {color[1]}")

    def on_file_select(self):
        idx = self.file_combo.current()
        if idx >= 0 and idx < len(self.spectra):
            self.current_index = idx
            # Обновляем превью цвета
            color = self.custom_colors.get(self.current_index,
                                           DEFAULT_COLORS.get(self.labels[self.current_index], '#95a5a6'))
            self.color_preview.config(bg=color)
            self.update_plot()

    def prev_file(self):
        if len(self.spectra) > 0:
            self.current_index = (self.current_index - 1) % len(self.spectra)
            self.file_combo.current(self.current_index)
            color = self.custom_colors.get(self.current_index,
                                           DEFAULT_COLORS.get(self.labels[self.current_index], '#95a5a6'))
            self.color_preview.config(bg=color)
            self.update_plot()

    def next_file(self):
        if len(self.spectra) > 0:
            self.current_index = (self.current_index + 1) % len(self.spectra)
            self.file_combo.current(self.current_index)
            color = self.custom_colors.get(self.current_index,
                                           DEFAULT_COLORS.get(self.labels[self.current_index], '#95a5a6'))
            self.color_preview.config(bg=color)
            self.update_plot()

    def preprocess_all(self):
        if not self.spectra:
            messagebox.showwarning("Внимание", "Нет данных")
            return

        self.log("Обработка...")
        self.processed_spectra = []

        try:
            for i, df in enumerate(self.spectra):
                wave, intensity = preprocess_spectrum(
                    df, config=CONFIG,
                    apply_smooth=self.apply_smooth.get(),
                    apply_baseline=self.apply_baseline.get(),
                    apply_normalize=self.apply_normalize.get()
                )
                self.processed_spectra.append((wave, intensity))

            self.wave_axis, self.X_matrix = interpolate_to_common_wave(self.spectra, CONFIG)

            self.log(f"✅ Обработано: {len(self.processed_spectra)}")
            self.update_plot()

        except Exception as e:
            self.log(f"❌ Ошибка: {str(e)}")
            messagebox.showerror("Ошибка", f"Ошибка обработки:\n{str(e)}")
            self.processed_spectra = []

    def update_plot(self):
        self.ax.clear()

        if not self.spectra:
            self.show_welcome()
            return

        if self.current_index >= len(self.spectra):
            self.current_index = 0

        if self.show_all.get():
            for i in range(len(self.spectra)):
                if self.processed_spectra and i < len(self.processed_spectra):
                    wave, intensity = self.processed_spectra[i]
                    title_extra = " (обр.)"
                else:
                    df = self.spectra[i]
                    wave = df['Wave'].values
                    intensity = df['Intensity'].values
                    title_extra = " (ориг.)"

                label = self.labels[i]
                # Используем пользовательский цвет или цвет по умолчанию
                color = self.custom_colors.get(i, DEFAULT_COLORS.get(label, '#95a5a6'))
                self.ax.plot(wave, intensity, color=color, alpha=0.7, linewidth=1.0,
                             label=f"{label} #{i + 1}{title_extra}")

            self.ax.set_title(f'Все спектры ({len(self.spectra)})', fontsize=14, pad=15)
            self.ax.legend(fontsize=8, loc='best', ncol=2)
        else:
            df = self.spectra[self.current_index]

            if self.processed_spectra and self.current_index < len(self.processed_spectra):
                wave, intensity = self.processed_spectra[self.current_index]
                title_extra = " (обработанный)"
            else:
                wave = df['Wave'].values
                intensity = df['Intensity'].values
                title_extra = " (оригинал)"

            label = self.labels[self.current_index]
            # Используем пользовательский цвет или цвет по умолчанию
            color = self.custom_colors.get(self.current_index, DEFAULT_COLORS.get(label, '#95a5a6'))
            self.ax.plot(wave, intensity, color=color, linewidth=1.5,
                         label=f"{label}: {os.path.basename(self.filepaths[self.current_index])}")
            self.ax.set_title(f'Спектр {self.current_index + 1}/{len(self.spectra)}: {label}{title_extra}', fontsize=14,
                              pad=15)
            self.ax.legend(fontsize=10, loc='best')

        self.ax.set_xlabel('Wavenumber (cm⁻¹)', fontsize=12)
        self.ax.set_ylabel('Intensity (a.u.)', fontsize=12)
        self.ax.grid(True, alpha=0.3, linestyle='--')

        self.index_label.config(text=f"Файл: {self.current_index + 1}/{len(self.spectra)}")
        self.fig.tight_layout()
        self.canvas.draw()

    def update_stats(self):
        n = len(self.spectra)
        if n == 0:
            self.stats_label.config(text="Файлов: 0")
            return

        total = sum(len(df) for df in self.spectra)
        label_counts = pd.Series(self.labels).value_counts()

        stats = f"Файлов: {n}\nТочек: {total:,}\n\n"
        for label, count in label_counts.items():
            stats += f"{label}: {count}\n"

        self.stats_label.config(text=stats)

    def train_model(self):
        if self.X_matrix is None or len(self.labels) < 10:
            messagebox.showwarning("Внимание", "Нужно ≥10 спектров")
            return

        valid_idx = [i for i, l in enumerate(self.labels) if l != 'unknown']
        if len(valid_idx) < 10:
            messagebox.showwarning("Внимание", "Нужно ≥10 размеченных")
            return

        X = self.X_matrix[valid_idx]
        y = [self.labels[i] for i in valid_idx]

        self.log("Обучение...")

        try:
            self.ml_results = train_classification_model(X, y, CONFIG)

            msg = (f"✅ Модель обучена!\n\n"
                   f"Accuracy: {self.ml_results['metrics']['accuracy']:.4f}\n"
                   f"F1: {self.ml_results['metrics']['f1_weighted']:.4f}\n"
                   f"CV F1: {self.ml_results['metrics']['cv_f1_mean']:.4f}")

            messagebox.showinfo("Результат", msg)
            self.log(f"✅ Accuracy: {self.ml_results['metrics']['accuracy']:.4f}")

        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка: {str(e)}")
            self.log(f"❌ {str(e)}")


def main():
    print("=" * 60)
    print("🔬 Рамановский Анализатор v7.0")
    print("=" * 60)
    print("🎨 Можно менять цвет каждого спектра")
    print("🔁 Кнопка сброса возвращает к оригиналу")
    print("=" * 60)

    root = tk.Tk()
    app = RamanAnalyzerApp(root)

    print("✅ GUI запущен")
    print("=" * 60)

    root.mainloop()


if __name__ == '__main__':
    main()
