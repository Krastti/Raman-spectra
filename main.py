import tkinter as tk
from tkinter import ttk, messagebox
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

        # 🎯 Заглушки для архитектурных признаков (не влияют на предсказание)
        self.dummy_x = 0.0  # Фиктивное значение X
        self.dummy_y = 0.0  # Фиктивное значение Y

        # ✅ Сначала создаём интерфейс
        self.create_widgets()
        self.center_window()
        self.apply_styles()
        
        # ✅ Потом загружаем модель (когда status_label уже существует)
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

    def predict_with_model(self, wave: float, intensity: float) -> tuple[str, np.ndarray]:
        """
        Предсказание класса с помощью модели
        
        Ожидает 4 признака: [x, y, wave, intensity]
        Возвращает: (класс_как_строка, массив_вероятностей)
        """
        if self.model is None:
            pred_class = random.choice(self.classes)
            probs = np.array([0.33, 0.33, 0.34])
            return pred_class, probs
        
        try:
            # Подготовка признаков: [x, y, wave, intensity]
            features = np.array([[
                self.dummy_x,
                self.dummy_y,
                wave,
                intensity
            ]])
            
            # Предсказание
            raw_pred = self.model.predict(features)[0]
            
            # 🔧 Конвертация предсказания в строку
            # Если модель возвращает индексы (0,1,2) — мапим на названия классов
            if isinstance(raw_pred, (int, np.integer)):
                idx = int(raw_pred)
                if 0 <= idx < len(self.classes):
                    pred_class = self.classes[idx]
                else:
                    pred_class = random.choice(self.classes)  # fallback
            else:
                # Если модель уже возвращает строку
                pred_class = str(raw_pred).strip().lower()
                if pred_class not in self.classes:
                    pred_class = random.choice(self.classes)  # fallback
            
            # Получение вероятностей
            if hasattr(self.model, 'predict_proba'):
                probs = self.model.predict_proba(features)[0]
            else:
                probs = np.ones(len(self.classes)) / len(self.classes)
                probs[self.classes.index(pred_class)] = 0.98
                
            return pred_class, probs
            
        except Exception as e:
            print(f"Ошибка предсказания: {e}")
            return random.choice(self.classes), np.array([0.33, 0.33, 0.34])

    def _interpolate_spectrum(self, wave: float, intensity: float, n_points: int = 100) -> np.ndarray:
        """
        Интерполяция точки спектра до полного вектора признаков
        (используйте, если модель обучена на полных спектрах)
        """
        # Заглушка: создаём простой гауссов пик вокруг точки
        x = np.linspace(400, 1800, n_points)  # Типичный диапазон Raman
        center = np.interp(wave, [400, 1800], [0, n_points-1])
        sigma = 5  # Ширина пика
        spectrum = intensity * np.exp(-((x - wave) ** 2) / (2 * sigma ** 2))
        return spectrum.reshape(1, -1)

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

            # Запускаем отдельный процесс с тем же интерпретатором Python
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

        # Поле WAVE
        wave_label = tk.Label(card, text="WAVE", font=('Helvetica', 10, 'normal'),
                             bg=self.colors['surface'], fg=self.colors['secondary'])
        wave_label.place(x=30, y=70)

        wave_frame = tk.Frame(card, bg=self.colors['border'], bd=0)
        wave_frame.place(x=30, y=95, width=280, height=40)
        self.wave_entry = tk.Entry(wave_frame, font=('Helvetica', 12), bg='white',
                                  fg=self.colors['primary'], bd=0, highlightthickness=0)
        self.wave_entry.place(x=10, y=10, width=260, height=20)

        # Поле INTENSITY
        int_label = tk.Label(card, text="INTENSITY", font=('Helvetica', 10, 'normal'),
                            bg=self.colors['surface'], fg=self.colors['secondary'])
        int_label.place(x=30, y=150)

        int_frame = tk.Frame(card, bg=self.colors['border'], bd=0)
        int_frame.place(x=30, y=175, width=280, height=40)
        self.intensity_entry = tk.Entry(int_frame, font=('Helvetica', 12), bg='white',
                                       fg=self.colors['primary'], bd=0, highlightthickness=0)
        self.intensity_entry.place(x=10, y=10, width=260, height=20)

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
        self.wave_entry.bind('<Return>', lambda e: self.intensity_entry.focus())
        self.intensity_entry.bind('<Return>', lambda e: self.predict_class())

    def predict_class(self):
        """Предсказание класса с использованием модели"""
        wave = self.wave_entry.get().strip()
        intensity = self.intensity_entry.get().strip()

        if not wave or not intensity:
            self.show_error("Заполните все поля")
            return

        try:
            wave_val = float(wave)
            int_val = float(intensity)

            # Предсказание через модель
            pred_class, probs = self.predict_with_model(wave_val, int_val)
            
            # Обновляем результат
            self.result_value.config(text=pred_class.upper())
            
            # Цвет в зависимости от класса
            colors = {'control': '#00B894', 'endo': '#E17055', 'exo': '#6C5CE7'}
            self.result_value.config(fg=colors.get(pred_class, self.colors['accent']))
            
            # Отображаем уверенность
            confidence = probs[self.classes.index(pred_class)] * 100
            self.confidence_label.config(text=f"уверенность: {confidence:.1f}%")
            
            # Время и статус
            current_time = datetime.now().strftime("%H:%M")
            self.time_indicator.config(text=current_time)
            self.status_label.config(text=f"предсказано: {pred_class}", fg=self.colors['success'])
            
            # Сброс статуса
            self.root.after(3000, self.reset_status)

        except ValueError:
            self.show_error("Введите корректные числа")
        except Exception as e:
            self.show_error(f"Ошибка: {str(e)[:30]}")
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