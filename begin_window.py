import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import joblib
from scipy import interpolate
import os
import warnings
import random
from datetime import datetime

warnings.filterwarnings('ignore')


class RamanDesignerGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ RAMAN • CLASSIFIER")
        self.root.geometry("480x620")

        # Отключаем изменение размера для аккуратности
        self.root.resizable(False, False)

        # Цветовая палитра (современная, минималистичная)
        self.setup_colors()

        # Переменные
        self.model = None
        self.classes = ['control', 'endo', 'exo']

        # Настройка стилей
        self.setup_styles()

        # Создание интерфейса
        self.create_widgets()

        # Центрирование окна
        self.center_window()

        # Применение стилей
        self.apply_styles()

    def setup_colors(self):
        """Цветовая палитра в стиле минимализм"""
        self.colors = {
            'bg': '#F5F5F7',  # Светло-серый фон
            'surface': '#FFFFFF',  # Белая поверхность
            'primary': '#2D3436',  # Темно-серый для текста
            'secondary': '#636E72',  # Серый для подписей
            'accent': '#6C5CE7',  # Фиолетовый акцент
            'accent_soft': '#A29BFE',  # Светло-фиолетовый
            'success': '#00B894',  # Мятный для результата
            'border': '#DFE6E9',  # Цвет рамок
            'result_bg': '#F8F9FA',  # Светло-серый фон для результата
            'viz_button': '#00B894',  # Бирюзовый для кнопки визуализации
            'viz_button_hover': '#00A884'  # Темнее при наведении
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

    def create_widgets(self):
        """Создание всех элементов интерфейса"""

        # ========== ГЛАВНЫЙ КОНТЕЙНЕР ==========
        main_container = tk.Frame(self.root, bg=self.colors['bg'])
        main_container.pack(fill=tk.BOTH, expand=True)

        # ========== ХЕДЕР ==========
        self.create_header(main_container)

        # ========== ОСНОВНАЯ КАРТОЧКА ==========
        self.create_main_card(main_container)

        # ========== ФУТЕР ==========
        self.create_footer(main_container)

    def create_header(self, parent):
        """Создание стильного хедера с кнопкой визуализации"""

        header = tk.Frame(parent, bg=self.colors['accent'], height=100)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        # Заголовок
        title = tk.Label(
            header,
            text="RAMAN",
            font=self.fonts['title'],
            bg=self.colors['accent'],
            fg='white'
        )
        title.place(x=30, y=20)

        # Подзаголовок
        subtitle = tk.Label(
            header,
            text="spectra classifier",
            font=self.fonts['subtitle'],
            bg=self.colors['accent'],
            fg='white'
        )
        subtitle.place(x=35, y=65)

        # ===== КНОПКА ВИЗУАЛИЗАЦИИ =====
        self.viz_button = tk.Button(
            header,
            text="ВИЗУАЛИЗАЦИЯ",
            font=self.fonts['viz_button'],
            bg=self.colors['viz_button'],
            fg='white',
            activebackground=self.colors['viz_button_hover'],
            activeforeground='white',
            bd=0,
            cursor='hand2',
            command=self.open_visualization,
            padx=7,
            pady=5
        )
        self.viz_button.place(x=320, y=30)

        # Добавляем эффект наведения
        self.viz_button.bind('<Enter>', self.on_viz_hover)
        self.viz_button.bind('<Leave>', self.on_viz_leave)

    def on_viz_hover(self, event):
        """Эффект при наведении на кнопку визуализации"""
        self.viz_button.config(bg=self.colors['viz_button_hover'])

    def on_viz_leave(self, event):
        """Эффект при уходе с кнопки визуализации"""
        self.viz_button.config(bg=self.colors['viz_button'])

    def open_visualization(self):
        """Открытие окна визуализации (заглушка)"""

        # Создаем новое окно
        viz_window = tk.Toplevel(self.root)
        viz_window.title("📊 Визуализация спектра")
        viz_window.geometry("500x400")
        viz_window.configure(bg=self.colors['surface'])

        # Центрируем окно
        viz_window.update_idletasks()
        x = (viz_window.winfo_screenwidth() // 2) - (500 // 2)
        y = (viz_window.winfo_screenheight() // 2) - (400 // 2)
        viz_window.geometry(f'+{x}+{y}')

        # Заголовок
        title = tk.Label(
            viz_window,
            text="Визуализация спектральных данных",
            font=('Helvetica', 14, 'bold'),
            bg=self.colors['surface'],
            fg=self.colors['primary']
        )
        title.pack(pady=20)

        # Получаем текущие значения для визуализации
        wave = self.wave_entry.get().strip()
        intensity = self.intensity_entry.get().strip()

        # Создаем рамку для информации
        info_frame = tk.Frame(viz_window, bg=self.colors['surface'], highlightbackground=self.colors['border'],
                              highlightthickness=1)
        info_frame.pack(pady=10, padx=40, fill=tk.X)

        if wave and intensity:
            try:
                wave_val = float(wave)
                int_val = float(intensity)

                info_text = f"Текущая точка спектра:\nWave = {wave_val:.2f} cm⁻¹\nIntensity = {int_val:.2f}"

                # Заглушка для графика
                canvas = tk.Canvas(info_frame, width=400, height=150, bg='#F8F9FA', highlightthickness=0)
                canvas.pack(pady=10)

                # Рисуем простую заглушку графика
                canvas.create_line(50, 120, 350, 120, fill=self.colors['border'], width=2)  # Ось X
                canvas.create_line(50, 20, 50, 120, fill=self.colors['border'], width=2)  # Ось Y

                # Рисуем простой пик
                points = [50, 120, 150, 40, 250, 80, 350, 100]
                canvas.create_line(points, fill=self.colors['accent'], width=3, smooth=True)

                # Отмечаем текущую точку
                x_pos = 50 + (wave_val % 300)  # Просто для демонстрации
                y_pos = 120 - (int_val % 100)
                canvas.create_oval(x_pos - 5, y_pos - 5, x_pos + 5, y_pos + 5, fill=self.colors['viz_button'],
                                   outline='')

            except:
                info_text = "Некорректные данные для визуализации"
        else:
            info_text = "Введите данные спектра для визуализации"

        info_label = tk.Label(
            info_frame,
            text=info_text,
            font=('Helvetica', 11),
            bg=self.colors['surface'],
            fg=self.colors['secondary'],
            justify=tk.CENTER
        )
        info_label.pack(pady=10)

        # Заглушка сообщение
        message_label = tk.Label(
            viz_window,
            text="🔧 Функция визуализации в разработке\nЗдесь будет отображаться спектр с выделенными пиками",
            font=('Helvetica', 10, 'italic'),
            bg=self.colors['surface'],
            fg=self.colors['secondary']
        )
        message_label.pack(pady=20)

        # Кнопка закрытия
        close_button = tk.Button(
            viz_window,
            text="ЗАКРЫТЬ",
            font=('Helvetica', 10, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            activebackground=self.colors['accent_soft'],
            activeforeground='white',
            bd=0,
            cursor='hand2',
            command=viz_window.destroy,
            padx=30,
            pady=8
        )
        close_button.pack(pady=20)

    def create_main_card(self, parent):
        """Создание центральной карточки с полями ввода"""

        # Карточка
        card = tk.Frame(
            parent,
            bg=self.colors['surface'],
            highlightbackground=self.colors['border'],
            highlightthickness=1,
            bd=0
        )
        card.place(x=40, y=130, width=400, height=420)

        # Заголовок карточки
        card_title = tk.Label(
            card,
            text="Введите параметры спектра",
            font=('Helvetica', 11, 'normal'),
            bg=self.colors['surface'],
            fg=self.colors['secondary']
        )
        card_title.place(x=30, y=20)

        # Линия-разделитель
        separator = tk.Frame(
            card,
            bg=self.colors['border'],
            height=1,
            width=340
        )
        separator.place(x=30, y=45)

        # ===== ПОЛЕ WAVE =====
        wave_label = tk.Label(
            card,
            text="WAVE",
            font=('Helvetica', 10, 'normal'),
            bg=self.colors['surface'],
            fg=self.colors['secondary']
        )
        wave_label.place(x=30, y=70)

        # Поле ввода Wave
        wave_frame = tk.Frame(
            card,
            bg=self.colors['border'],
            bd=0
        )
        wave_frame.place(x=30, y=95, width=280, height=40)

        self.wave_entry = tk.Entry(
            wave_frame,
            font=('Helvetica', 12),
            bg='white',
            fg=self.colors['primary'],
            bd=0,
            highlightthickness=0
        )
        self.wave_entry.place(x=10, y=10, width=260, height=20)

        # ===== ПОЛЕ INTENSITY =====
        int_label = tk.Label(
            card,
            text="INTENSITY",
            font=('Helvetica', 10, 'normal'),
            bg=self.colors['surface'],
            fg=self.colors['secondary']
        )
        int_label.place(x=30, y=150)

        # Поле ввода Intensity
        int_frame = tk.Frame(
            card,
            bg=self.colors['border'],
            bd=0
        )
        int_frame.place(x=30, y=175, width=280, height=40)

        self.intensity_entry = tk.Entry(
            int_frame,
            font=('Helvetica', 12),
            bg='white',
            fg=self.colors['primary'],
            bd=0,
            highlightthickness=0
        )
        self.intensity_entry.place(x=10, y=10, width=260, height=20)

        # ===== КНОПКА =====
        button = tk.Button(
            card,
            text="ПРЕДСКАЗАТЬ",
            font=('Helvetica', 11, 'bold'),
            bg=self.colors['accent'],
            fg='white',
            activebackground=self.colors['accent_soft'],
            activeforeground='white',
            bd=0,
            cursor='hand2',
            command=self.predict_class
        )
        button.place(x=30, y=240, width=340, height=45)

        # ===== ОБЛАСТЬ РЕЗУЛЬТАТА =====
        self.create_result_area(card)

    def create_result_area(self, parent):
        """Создание области результата с хорошей видимостью"""

        # Заголовок результата
        result_title = tk.Label(
            parent,
            text="РЕЗУЛЬТАТ",
            font=('Helvetica', 11, 'normal'),
            bg=self.colors['surface'],
            fg=self.colors['secondary']
        )
        result_title.place(x=30, y=310)

        # Контейнер для результата с фоном
        result_container = tk.Frame(
            parent,
            bg='#F0F0F0',  # Светло-серый фон
            highlightbackground=self.colors['border'],
            highlightthickness=1,
            bd=0
        )
        result_container.place(x=30, y=335, width=340, height=70)

        # Значение результата (крупное и жирное)
        self.result_value = tk.Label(
            result_container,
            text="—",
            font=('Helvetica', 28, 'bold'),
            bg='#F0F0F0',
            fg=self.colors['accent']
        )
        self.result_value.place(relx=0.5, rely=0.5, anchor='center')

        # Индикатор времени (маленький в углу)
        self.time_indicator = tk.Label(
            result_container,
            text="",
            font=('Helvetica', 8),
            bg='#F0F0F0',
            fg=self.colors['secondary']
        )
        self.time_indicator.place(x=290, y=50)

    def create_footer(self, parent):
        """Создание футера"""

        footer = tk.Frame(parent, bg=self.colors['bg'], height=40)
        footer.pack(side=tk.BOTTOM, fill=tk.X)

        # Статус
        self.status_label = tk.Label(
            footer,
            text="⚡ ожидание ввода",
            font=('Helvetica', 9),
            bg=self.colors['bg'],
            fg=self.colors['secondary']
        )
        self.status_label.pack(side=tk.LEFT, padx=20, pady=10)

        # Комада
        team = tk.Label(
            footer,
            text="Экзаменационная группа",
            font=('Helvetica', 9),
            bg=self.colors['bg'],
            fg=self.colors['secondary']
        )
        team.pack(side=tk.RIGHT, padx=20, pady=10)

    def apply_styles(self):
        """Применение дополнительных стилей"""
        self.wave_entry.bind('<Return>', lambda e: self.intensity_entry.focus())
        self.intensity_entry.bind('<Return>', lambda e: self.predict_class())

    def predict_class(self):
        """Предсказание класса"""

        # Получаем значения
        wave = self.wave_entry.get().strip()
        intensity = self.intensity_entry.get().strip()

        # Валидация
        if not wave or not intensity:
            self.show_error("Заполните все поля")
            return

        try:
            wave_val = float(wave)
            int_val = float(intensity)

            # нужен get_class()
            # пока для демонстрации - случайный класс
            pred_class = random.choice(self.classes)

            # Обновляем результат с ярким цветом
            self.result_value.config(text=pred_class.upper())

            # Меняем цвет в зависимости от класса
            colors = {
                'control': '#00B894',
                'endo': '#E17055',
                'exo': '#6C5CE7'
            }
            self.result_value.config(fg=colors.get(pred_class, self.colors['accent']))

            # Обновляем статус
            current_time = datetime.now().strftime("%H:%M")
            self.time_indicator.config(text=current_time)
            self.status_label.config(
                text=f"✓ предсказано: {pred_class}",
                fg=self.colors['success']
            )

            # Сброс статуса через 3 секунды
            self.root.after(3000, self.reset_status)

        except ValueError:
            self.show_error("Введите корректные числа")

    def show_error(self, message):
        """Показ ошибки"""
        self.status_label.config(text=f"⚠ {message}", fg='#E17055')
        self.result_value.config(text="—", fg=self.colors['accent'])
        self.root.after(2000, self.reset_status)

    def reset_status(self):
        """Сброс статуса"""
        self.status_label.config(text="⚡ ожидание ввода", fg=self.colors['secondary'])


def main():
    """Запуск приложения"""
    root = tk.Tk()
    app = RamanDesignerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
