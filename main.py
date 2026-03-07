import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import numpy as np
import joblib
from scipy import interpolate
import os
import subprocess
import sys
import warnings
import random
from datetime import datetime

warnings.filterwarnings('ignore')

from Cleaning_module.remove_rays import CosmicRayRemover
from Cleaning_module.baseline_correction import FluorescenceCorrector
from Cleaning_module.smoothing import SpectrumSmoother
from Cleaning_module.normalization import SpectrumNormalizer


class RamanDesignerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("RAMAN - CLASSIFIER")
        self.root.geometry("480x620")
        self.root.resizable(False, False)

        # Цветовая палитра и переменные
        self.setup_colors()
        self.model = None
        self.classes = ['control', 'endo', 'exo']
        self.model_path = "raman_model.joblib"
        
        # Настройка стилей
        self.setup_styles()

        # Заглушки для архитектурных признаков (не влияют на предсказание)
        self.dummy_x = 0.0
        self.dummy_y = 0.0

        # Сначала создаём интерфейс
        self.create_widgets()
        self.center_window()
        self.apply_styles()
        
        # Потом загружаем модель
        self.load_model()

    def setup_colors(self):
        """Цветовая палитра в стиле минимализм"""
        self.colors = {
            'bg': '#F5F5F7',
            'surface': '#FFFFFF',
            'primary': '#2D3436',
            'secondary': '#636E72',
            'accent': '#6C5CE7',
            'accent_soft': '#A29BFE',
            'success': '#00B894',
            'border': '#DFE6E9',
            'result_bg': '#F8F9FA',
            'viz_button': '#00B894',
            'viz_button_hover': '#00A884',
            'error': '#E17055'
        }

    def setup_styles(self):
        """Настройка шрифтов и стилей"""
        self.fonts = {
            'title': ('Helvetica', 36, 'normal'),
            'subtitle': ('Helvetica', 11, 'normal'),
            'label': ('Helvetica', 11, 'normal'),
            'entry': ('Helvetica', 12, 'normal'),
            'result_label': ('Helvetica', 10, 'normal'),
            'result': ('Helvetica', 36, 'bold'),
            'small': ('Helvetica', 9, 'normal'),
            'viz_button': ('Helvetica', 10, 'bold')
        }

    def center_window(self):
        """Центрирование окна"""
        self.root.update_idletasks()
        x = (self.root.winfo_screenwidth() // 2) - (480 // 2)
        y = (self.root.winfo_screenheight() // 2) - (620 // 2)
        self.root.geometry(f'+{x}+{y}')

    def load_model(self):
        """Загрузка модели .joblib"""
        try:
            if os.path.exists(self.model_path):
                self.model = joblib.load(self.model_path)
                self.status_label.config(text="модель загружена", fg=self.colors['success'])
                print(f"Модель загружена: {self.model_path}")
            else:
                self.model = None
                self.status_label.config(text="модель не найдена", fg=self.colors['error'])
                print(f"Файл модели не найден: {self.model_path}")
        except Exception as e:
            self.model = None
            self.status_label.config(text="ошибка загрузки", fg=self.colors['error'])
            print(f"Ошибка загрузки модели: {e}")

    def predict_with_model(self, full_spectrum: np.ndarray = None, wave: float = None, intensity: float = None) -> tuple[str, np.ndarray]:
        """
        Предсказание класса.
        
        Приоритет: 
        - Если передан full_spectrum → используем его как признаки
        - Иначе → используем [dummy_x, dummy_y, wave, intensity] (старый режим)
        
        Возвращает: (класс_как_строка, массив_вероятностей)
        """
        if self.model is None:
            pred_class = random.choice(self.classes)
            probs = np.array([0.33, 0.33, 0.34])
            return pred_class, probs

        try:
            # Подготовка признаков
            if full_spectrum is not None:
                # Режим полного спектра
                features = full_spectrum.reshape(1, -1)
            else:
                # Старый режим: 4 признака
                features = np.array([[
                    self.dummy_x,
                    self.dummy_y,
                    wave if wave is not None else 0.0,
                    intensity if intensity is not None else 0.0
                ]])

            # Предсказание метки
            raw_pred = self.model.predict(features)[0]

            # Получаем классы модели
            model_classes_raw = None
            if hasattr(self.model, 'classes_'):
                model_classes_raw = list(self.model.classes_)

            # Получаем вероятности
            probs_model = None
            if hasattr(self.model, 'predict_proba'):
                probs_model = self.model.predict_proba(features)[0]

            # Выравнивание вероятностей по self.classes
            probs_aligned = np.zeros(len(self.classes), dtype=float)

            if probs_model is not None and model_classes_raw is not None:
                for i, cls_val in enumerate(model_classes_raw):
                    try:
                        cls_str = str(cls_val).strip().lower()
                    except Exception:
                        cls_str = None

                    target_idx = None
                    if cls_str and cls_str in self.classes:
                        target_idx = self.classes.index(cls_str)
                    else:
                        try:
                            ci = int(float(cls_val))
                            if 0 <= ci < len(self.classes):
                                target_idx = ci
                        except Exception:
                            target_idx = None

                    if target_idx is not None:
                        probs_aligned[target_idx] = float(probs_model[i])

                if probs_aligned.sum() == 0:
                    probs_aligned = probs_model[:len(probs_aligned)] if len(probs_model) >= len(probs_aligned) else np.ones(len(self.classes)) / len(self.classes)
            else:
                # Fallback без predict_proba
                pred_idx = None
                try:
                    rp = str(raw_pred).strip().lower()
                    if rp in self.classes:
                        pred_idx = self.classes.index(rp)
                    else:
                        rpi = int(float(raw_pred))
                        if 0 <= rpi < len(self.classes):
                            pred_idx = rpi
                except Exception:
                    pred_idx = None

                if pred_idx is None:
                    probs_aligned = np.ones(len(self.classes)) / len(self.classes)
                else:
                    probs_aligned = np.ones(len(self.classes)) * 0.01
                    probs_aligned[pred_idx] = 0.98

            # Итоговый класс
            best_idx = int(np.nanargmax(probs_aligned))
            pred_class = self.classes[best_idx]

            return pred_class, probs_aligned

        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return random.choice(self.classes), np.array([0.33, 0.33, 0.34])

    def preprocess_full_spectrum(self, wave_axis: np.ndarray, intensity: np.ndarray, n_points: int = 200) -> np.ndarray:
        """
        Полная предобработка спектра: интерполяция на единую сетку + пайплайн очистки.
        Возвращает нормализованный вектор длиной n_points.
        """
        import math
        try:
            # 1. Интерполяция на единую сетку [400, 1800] см⁻¹
            wave_axis = np.asarray(wave_axis, dtype=float).ravel()
            intensity = np.asarray(intensity, dtype=float).ravel()
            
            # Убираем невалидные значения
            mask = ~(np.isnan(wave_axis) | np.isnan(intensity) | np.isinf(wave_axis) | np.isinf(intensity))
            wave_axis, intensity = wave_axis[mask], intensity[mask]
            
            if wave_axis.size < 10:
                raise ValueError("Слишком мало точек в спектре")
            
            # Целевая сетка
            target_wave = np.linspace(400.0, 1800.0, n_points)
            
            # Интерполяция (линейная, с экстраполяцией = 0)
            f = interpolate.interp1d(wave_axis, intensity, 
                                    kind='linear', 
                                    bounds_error=False, 
                                    fill_value=0.0)
            spectrum = f(target_wave).astype(float)
            spectrum = np.nan_to_num(spectrum, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 2. Удаление космических лучей
            try:
                remover = CosmicRayRemover()
                spectrum, _ = remover.remove_cosmic_rays(target_wave, spectrum, method='interpolation')
            except Exception:
                pass  # Оставляем как есть при ошибке
            
            # 3. Коррекция базовой линии
            try:
                corrector = FluorescenceCorrector()
                _, spectrum = corrector.correct_baseline(target_wave, spectrum, method='poly', degree=5)
                if spectrum is None or len(spectrum) == 0:
                    raise ValueError("Baseline correction failed")
            except Exception:
                pass
            
            # Сдвигаем, чтобы не было отрицательных значений
            if np.min(spectrum) < 0:
                spectrum = spectrum - np.min(spectrum) + 1e-3
            
            # 4. Сглаживание
            try:
                smoother = SpectrumSmoother(method='savgol')
                # Подбираем window_length под n_points (должно быть нечётным и < длины)
                wl = min(11, n_points // 10)
                if wl % 2 == 0: wl += 1
                if wl >= 3:
                    spectrum, _ = smoother.smooth(target_wave, spectrum, window_length=wl, polyorder=3)
            except Exception:
                pass
            
            # 5. Нормализация (L2 / vector norm)
            try:
                normalizer = SpectrumNormalizer(method='vector')
                spectrum, _ = normalizer.normalize(target_wave, spectrum)
                if spectrum is None or len(spectrum) == 0:
                    raise ValueError("Normalization failed")
            except Exception:
                # Fallback: ручная L2-нормализация
                norm = np.linalg.norm(spectrum)
                if norm > 1e-10:
                    spectrum = spectrum / norm
            
            return spectrum.astype(float).ravel()
            
        except Exception as e:
            print(f"Предобработка спектра не удалась: {e}")
            # Возвращаем "безопасный" нулевой вектор
            return np.zeros(n_points, dtype=float)

    def create_widgets(self):
        """Создание всех элементов интерфейса"""
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True)
        self.create_header(main_container)
        self.create_main_card(main_container)
        self.create_footer(main_container)

    def create_header(self, parent):
        """Создание стильного хедера"""
        header = tk.Frame(parent, bg=self.colors['accent'], height=100)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        title = tk.Label(header, text="RAMAN", font=self.fonts['title'], bg=self.colors['accent'], fg='white')
        title.place(x=30, y=20)

        subtitle = tk.Label(header, text="spectra classifier", font=self.fonts['subtitle'], bg=self.colors['accent'], fg='white')
        subtitle.place(x=35, y=65)

        # Кнопка визуализации
        self.viz_button = tk.Button(
            header, text="ВИЗУАЛИЗАЦИЯ", font=self.fonts['viz_button'],
            bg=self.colors['viz_button'], fg='white',
            activebackground=self.colors['viz_button_hover'], activeforeground='white',
            bd=0, cursor='hand2', command=self.open_visualization, padx=7, pady=5
        )
        self.viz_button.place(x=320, y=30)
        self.viz_button.bind('<Enter>', self.on_viz_hover)
        self.viz_button.bind('<Leave>', self.on_viz_leave)

    def on_viz_hover(self, event):
        self.viz_button.config(bg=self.colors['viz_button_hover'])

    def on_viz_leave(self, event):
        self.viz_button.config(bg=self.colors['viz_button'])

    def open_visualization(self):
        """Запустить внешнее приложение визуализации (raman_analyzer.py)."""
        try:
            script_path = os.path.join(os.path.dirname(__file__), 'raman_analyzer.py')
            if not os.path.exists(script_path):
                messagebox.showerror("Ошибка", f"Файл визуализации не найден:\n{script_path}")
                return

            subprocess.Popen([sys.executable, script_path], cwd=os.path.dirname(script_path))

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось запустить визуализацию:\n{e}")

    def create_main_card(self, parent):
        """Создание центральной карточки"""
        card = tk.Frame(parent, bg=self.colors['surface'],
                       highlightbackground=self.colors['border'], highlightthickness=1, bd=0)
        card.place(x=40, y=130, width=400, height=420)

        card_title = tk.Label(card, text="Введите параметры спектра",
                             font=('Helvetica', 11, 'normal'), bg=self.colors['surface'], fg=self.colors['secondary'])
        card_title.place(x=30, y=20)

        separator = tk.Frame(card, bg=self.colors['border'], height=1, width=340)
        separator.place(x=30, y=45)

        # Загрузка файла спектра
        upload_label = tk.Label(card, text="Загрузите .txt файл спектра",
                    font=('Helvetica', 10, 'normal'), bg=self.colors['surface'],
                    fg=self.colors['secondary'])
        upload_label.place(x=30, y=70)

        # Кнопка загрузки
        self.upload_button = tk.Button(card, text="Загрузить .txt", font=('Helvetica', 11, 'normal'),
                           bg=self.colors['accent'], fg='white', bd=0, cursor='hand2',
                           command=self.load_txt_file)
        self.upload_button.place(x=30, y=100, width=140, height=36)

        # Показ выбранного файла
        self.file_label = tk.Label(card, text="Файл не выбран", font=('Helvetica', 9, 'normal'),
                       bg=self.colors['surface'], fg=self.colors['secondary'])
        self.file_label.place(x=180, y=108)

        # Кнопка предсказания
        button = tk.Button(card, text="ПРЕДСКАЗАТЬ", font=('Helvetica', 11, 'bold'),
                          bg=self.colors['accent'], fg='white',
                          activebackground=self.colors['accent_soft'], activeforeground='white',
                          bd=0, cursor='hand2', command=self.predict_class)
        button.place(x=30, y=240, width=340, height=45)

        # Область результата
        self.create_result_area(card)

    def create_result_area(self, parent):
        """Создание области результата"""
        result_title = tk.Label(parent, text="РЕЗУЛЬТАТ", font=('Helvetica', 11, 'normal'),
                               bg=self.colors['surface'], fg=self.colors['secondary'])
        result_title.place(x=30, y=310)

        result_container = tk.Frame(parent, bg='#F0F0F0',
                                   highlightbackground=self.colors['border'], highlightthickness=1, bd=0)
        result_container.place(x=30, y=335, width=340, height=70)

        self.result_value = tk.Label(result_container, text="—", font=('Helvetica', 28, 'bold'),
                                    bg='#F0F0F0', fg=self.colors['accent'])
        self.result_value.place(relx=0.5, rely=0.5, anchor='center')

        # Индикатор уверенности
        self.confidence_label = tk.Label(result_container, text="", font=('Helvetica', 9),
                                        bg='#F0F0F0', fg=self.colors['secondary'])
        self.confidence_label.place(x=10, y=5)

        self.time_indicator = tk.Label(result_container, text="", font=('Helvetica', 8),
                                      bg='#F0F0F0', fg=self.colors['secondary'])
        self.time_indicator.place(x=290, y=50)

    def create_footer(self, parent):
        """Создание футера"""
        footer = tk.Frame(parent, bg=self.colors['bg'], height=40)
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = tk.Label(footer, text="ожидание ввода", font=('Helvetica', 9),
                                    bg=self.colors['bg'], fg=self.colors['secondary'])
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)

        
        team = tk.Label(footer, text="Экзаменационная группа", font=('Helvetica', 9),
                       bg=self.colors['bg'], fg=self.colors['secondary'])
        team.pack(side=tk.RIGHT, padx=20, pady=10)

    def apply_styles(self):
        """Применение дополнительных стилей"""
        self.upload_button.bind('<Return>', lambda e: self.load_txt_file())

    def load_txt_file(self):
        """Открыть диалог выбора .txt и загрузить спектр (волна, интенсивность)."""
        path = filedialog.askopenfilename(title='Выберите .txt файл', filetypes=[('Text files', '*.txt'), ('All files', '*.*')])
        if not path:
            return
        try:
            data = np.loadtxt(path)
            if data.ndim == 1:
                # Попробуем интерпретировать как два столбца, если возможно
                if data.size >= 2 and data.size % 2 == 0:
                    data = data.reshape(-1, 2)
                else:
                    # Одномерный массив: если >=2 — используем первые два значения как точку
                    if data.size >= 2:
                        wave_axis = np.array([float(data[0])])
                        intensity = np.array([float(data[1])])
                    else:
                        raise ValueError('Файл не содержит достаточных данных')
            if data.ndim == 2:
                if data.shape[1] >= 2:
                    wave_axis = data[:, 0].astype(float)
                    intensity = data[:, 1].astype(float)
                else:
                    intensity = data[:, 0].astype(float)
                    wave_axis = np.arange(len(intensity)).astype(float)

            self.loaded_wave_axis = wave_axis
            self.loaded_intensity = intensity
            fname = os.path.basename(path)
            self.file_label.config(text=fname)
            self.status_label.config(text=f'файл загружен: {fname}', fg=self.colors['success'])

        except Exception as e:
            self.show_error(f'Не удалось загрузить файл: {e}')

    def predict_class(self):
        """Предсказание класса с использованием всего спектра"""
        if getattr(self, 'loaded_wave_axis', None) is None or getattr(self, 'loaded_intensity', None) is None:
            self.show_error("Загрузите .txt файл спектра")
            return

        try:
            # Предобработка ВСЕГО спектра
            cleaned_spectrum = self.preprocess_full_spectrum(
                self.loaded_wave_axis, 
                self.loaded_intensity,
                n_points=200  # Можно настроить под вашу модель
            )
            
            # Предсказание через модель (передаём весь вектор)
            pred_class, probs = self.predict_with_model(full_spectrum=cleaned_spectrum)
            
            # Обновляем результат
            self.result_value.config(text=pred_class.upper())
            
            # Цвет в зависимости от класса
            colors = {'control': '#00B894', 'endo': '#E17055', 'exo': '#6C5CE7'}
            self.result_value.config(fg=colors.get(pred_class, self.colors['accent']))
            
            # Отображаем уверенность
            try:
                if hasattr(self.model, 'classes_'):
                    model_cls = [str(c).strip().lower() for c in list(self.model.classes_)]
                    p_idx = model_cls.index(pred_class) if pred_class in model_cls else self.classes.index(pred_class)
                else:
                    p_idx = self.classes.index(pred_class) if pred_class in self.classes else 0
                confidence = float(probs[p_idx]) * 100
            except Exception:
                confidence = 0.0
            self.confidence_label.config(text=f"")
            
            # Время и статус
            current_time = datetime.now().strftime("%H:%M")
            self.time_indicator.config(text=current_time)
            self.status_label.config(text=f"предсказано: {pred_class}", fg=self.colors['success'])
            
            # Сброс статуса
            self.root.after(3000, self.reset_status)

        except Exception as e:
            self.show_error(f"Ошибка: {str(e)[:40]}")
            print(f"Критическая ошибка: {e}")

    def show_error(self, message):
        """Показ ошибки"""
        self.status_label.config(text=f"Ошибка: {message}", fg=self.colors['error'])
        self.result_value.config(text="—", fg=self.colors['accent'])
        self.confidence_label.config(text="")
        self.root.after(2000, self.reset_status)

    def reset_status(self):
        """Сброс статуса"""
        if self.model is None:
            self.status_label.config(text="модель не загружена", fg=self.colors['error'])
        else:
            self.status_label.config(text="ожидание ввода", fg=self.colors['secondary'])


def main():
    """Запуск приложения"""
    root = tk.Tk()
    app = RamanDesignerGUI(root)
    root.mainloop()


if __name__ == "__main__":

    main()


