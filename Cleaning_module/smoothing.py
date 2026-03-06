import numpy as np
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Optional


class SpectrumSmoother:
    """
    Класс для сглаживания Рамановских спектров.

    Сглаживание удаляет высокочастотный случайный шум, сохраняя форму пиков.
    """

    def __init__(self, method: str = 'savgol'):
        """
        Инициализация сглаживателя.

        Parameters
        ----------
        method : str
            Метод сглаживания: 'savgol' (Савицкого-Голея) или 'gaussian'.
        """
        self.method = method
        self.smoothed_spectrum = None
        self.params_used = {}

    def smooth(self,
               wave: np.ndarray,
               intensity: np.ndarray,
               **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Применение сглаживания к спектру.

        Parameters
        ----------
        wave : np.ndarray
            Массив волновых чисел.
        intensity : np.ndarray
            Массив интенсивностей.
        **kwargs
            Параметры для выбранного метода.

        Returns
        -------
        smoothed : np.ndarray
            Сглаженный спектр.
        info : dict
            Информация о применённых параметрах.
        """
        intensity = np.asarray(intensity, dtype=np.float64)

        if self.method == 'savgol':
            smoothed, info = self._savgol_smooth(intensity, **kwargs)
        elif self.method == 'gaussian':
            smoothed, info = self._gaussian_smooth(intensity, **kwargs)
        else:
            raise ValueError(f"Неизвестный метод: {method}. "
                             f"Доступные: 'savgol', 'gaussian'")

        self.smoothed_spectrum = smoothed
        self.params_used = info

        return smoothed, info

    def _savgol_smooth(self,
                       intensity: np.ndarray,
                       window_length: int = 11,
                       polyorder: int = 3,
                       deriv: int = 0) -> Tuple[np.ndarray, Dict]:
        """
        Сглаживание фильтром Савицкого-Голея.

        Parameters
        ----------
        intensity : np.ndarray
            Интенсивность спектра.
        window_length : int
            Длина окна фильтра (должна быть нечётным числом).
            Типичные значения: 5-15.
        polyorder : int
            Степень полинома для аппроксимации.
            Типичные значения: 2-4.
        deriv : int
            Порядок производной (0 = сглаживание).

        Returns
        -------
        smoothed : np.ndarray
            Сглаженный спектр.
        info : dict
            Параметры сглаживания.
        """
        # Проверка параметров
        if window_length % 2 == 0:
            window_length += 1
            print(f"⚠️ window_length увеличен до {window_length} (должен быть нечётным)")

        if window_length > len(intensity):
            window_length = len(intensity) if len(intensity) % 2 == 1 else len(intensity) - 1
            print(f"⚠️ window_length уменьшен до {window_length}")

        if polyorder >= window_length:
            polyorder = window_length - 1
            print(f"⚠️ polyorder уменьшен до {polyorder}")

        # Применение фильтра
        smoothed = savgol_filter(intensity, window_length, polyorder, deriv=deriv)

        info = {
            'method': 'savgol',
            'window_length': window_length,
            'polyorder': polyorder,
            'deriv': deriv
        }

        return smoothed, info

    def _gaussian_smooth(self,
                         intensity: np.ndarray,
                         sigma: float = 1.0) -> Tuple[np.ndarray, Dict]:
        """
        Гауссово сглаживание.

        Parameters
        ----------
        intensity : np.ndarray
            Интенсивность спектра.
        sigma : float
            Стандартное отклонение гауссиана.

        Returns
        -------
        smoothed : np.ndarray
            Сглаженный спектр.
        info : dict
            Параметры сглаживания.
        """
        smoothed = gaussian_filter1d(intensity, sigma=sigma)

        info = {
            'method': 'gaussian',
            'sigma': sigma
        }

        return smoothed, info

    def optimize_savgol_parameters(self,
                                   wave: np.ndarray,
                                   intensity: np.ndarray,
                                   window_range: list = None,
                                   polyorder_range: list = None) -> Dict:
        """
        Оптимизация параметров фильтра Савицкого-Голея.

        Parameters
        ----------
        wave : np.ndarray
            Волновые числа.
        intensity : np.ndarray
            Интенсивность.
        window_range : list
            Диапазон длин окон для поиска.
        polyorder_range : list
            Диапазон степеней полинома.

        Returns
        -------
        best_params : dict
            Лучшие параметры и метрики.
        """
        if window_range is None:
            window_range = [5, 7, 9, 11, 13, 15]
        if polyorder_range is None:
            polyorder_range = [2, 3, 4]

        best_score = float('inf')
        best_params = {'window_length': window_range[0], 'polyorder': polyorder_range[0]}
        results = []

        for window in window_range:
            if window % 2 == 0:
                window += 1
            if window >= len(intensity):
                continue

            for poly in polyorder_range:
                if poly >= window:
                    continue

                smoothed, _ = self._savgol_smooth(intensity, window_length=window, polyorder=poly)

                # Оценка качества: баланс между сглаживанием и сохранением пиков
                noise_reduction = np.std(intensity - smoothed) / np.std(intensity)
                peak_preservation = np.corrcoef(intensity, smoothed)[0, 1]

                # Счёт: меньше = лучше (компромисс между шумом и сохранением сигнала)
                score = noise_reduction * (1 - peak_preservation) * 1000

                results.append({
                    'window_length': window,
                    'polyorder': poly,
                    'score': score,
                    'noise_reduction': noise_reduction,
                    'peak_preservation': peak_preservation
                })

                if score < best_score:
                    best_score = score
                    best_params = {
                        'window_length': window,
                        'polyorder': poly,
                        'score': score,
                        'noise_reduction': noise_reduction,
                        'peak_preservation': peak_preservation
                    }

        best_params['all_results'] = results
        return best_params

    def visualize_smoothing(self,
                            wave: np.ndarray,
                            intensity_raw: np.ndarray,
                            intensity_smooth: np.ndarray,
                            figsize: Tuple[int, int] = (12, 6)) -> plt.Figure:
        """
        Визуализация результатов сглаживания.

        Parameters
        ----------
        wave : np.ndarray
            Волновые числа.
        intensity_raw : np.ndarray
            Исходная интенсивность.
        intensity_smooth : np.ndarray
            Сглаженная интенсивность.
        figsize : tuple
            Размер фигуры.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Фигура с визуализацией.
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Сглаживание Рамановского спектра', fontsize=14, fontweight='bold')

        # График 1: Сравнение спектров
        axes[0].plot(wave, intensity_raw, 'b-', linewidth=0.5, alpha=0.7, label='Исходный')
        axes[0].plot(wave, intensity_smooth, 'r-', linewidth=1, label='Сглаженный')
        axes[0].set_xlabel('Волновое число (cm⁻¹)')
        axes[0].set_ylabel('Интенсивность')
        axes[0].set_title('Сравнение спектров')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # График 2: Разница (удалённый шум)
        difference = intensity_raw - intensity_smooth
        axes[1].plot(wave, difference, 'green', linewidth=0.5)
        axes[1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[1].set_xlabel('Волновое число (cm⁻¹)')
        axes[1].set_ylabel('Разница')
        axes[1].set_title('Удалённый шум')
        axes[1].grid(True, alpha=0.3)

        # График 3: Статистика
        axes[2].axis('off')
        stats = {
            'Метод': self.params_used.get('method', 'N/A'),
            'Параметры': str({k: v for k, v in self.params_used.items() if k != 'method'}),
            'std исходный': f'{np.std(intensity_raw):.2f}',
            'std сглаженный': f'{np.std(intensity_smooth):.2f}',
            'SNR улучшение': f'{np.std(intensity_raw) / np.std(difference):.2f}x'
        }
        text = '\n'.join([f'{k}: {v}' for k, v in stats.items()])
        axes[2].text(0.1, 0.5, text, transform=axes[2].transAxes,
                     fontsize=11, verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[2].set_title('Статистика сглаживания', fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig


def batch_smoothing(spectra_list: list,
                    wave_key: str = 'wave',
                    intensity_key: str = 'intensity',
                    method: str = 'savgol',
                    **kwargs) -> Tuple[list, list]:
    """
    Пакетное сглаживание нескольких спектров.

    Parameters
    ----------
    spectra_list : list of dict
        Список словарей со спектрами.
    wave_key : str
        Ключ для волновых чисел.
    intensity_key : str
        Ключ для интенсивности.
    method : str
        Метод сглаживания.
    **kwargs
        Параметры для сглаживания.

    Returns
    -------
    smoothed_spectra : list of dict
        Сглаженные спектры.
    smoothing_info : list of dict
        Информация о сглаживании для каждого спектра.
    """
    smoother = SpectrumSmoother(method=method)
    smoothed_spectra = []
    smoothing_info = []

    for i, spectrum in enumerate(spectra_list):
        wave = spectrum[wave_key]
        intensity = spectrum[intensity_key]

        smoothed, info = smoother.smooth(wave, intensity, **kwargs)

        smoothed_spectrum = spectrum.copy()
        smoothed_spectrum[f'{intensity_key}_smoothed'] = smoothed

        info['spectrum_idx'] = i
        smoothed_spectra.append(smoothed_spectrum)
        smoothing_info.append(info)

    return smoothed_spectra, smoothing_info


# Пример использования
if __name__ == "__main__":
    # Генерация тестового спектра
    np.random.seed(42)
    wave = np.linspace(200, 2000, 1000)
    signal = 1000 + 500 * np.exp(-((wave - 1000) / 50) ** 2)
    noise = np.random.normal(0, 100, len(wave))
    intensity = signal + noise

    # Сглаживание
    smoother = SpectrumSmoother(method='savgol')
    intensity_smooth, info = smoother.smooth(wave, intensity, window_length=11, polyorder=3)

    print("✅ Сглаживание завершено!")
    print(f"Метод: {info['method']}")
    print(f"window_length: {info['window_length']}")
    print(f"polyorder: {info['polyorder']}")

    # Визуализация
    fig = smoother.visualize_smoothing(wave, intensity, intensity_smooth)
    plt.show()