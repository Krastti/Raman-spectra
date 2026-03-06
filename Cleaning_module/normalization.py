import numpy as np
from typing import Tuple, Dict, Optional, Union
import warnings


class SpectrumNormalizer:
    """
    Класс для нормировки Рамановских спектров.

    Нормировка устраняет вариации абсолютной интенсивности, вызванные
    разной концентрацией образца, фокусировкой лазера или мощностью.

    Methods
    -------
    normalize(wave, intensity, **kwargs)
        Применение нормировки к спектру.
    visualize_normalization(wave, intensity_raw, intensity_normalized)
        Визуализация результатов нормировки.
    """

    def __init__(self, method: str = 'vector'):
        """
        Инициализация нормализатора.

        Parameters
        ----------
        method : str
            Метод нормировки:
            - 'vector' : Vector Normalization (L2 norm) - рекомендуется для ML
            - 'minmax' : Min-Max Scaling [0, 1]
            - 'snv' : Standard Normal Variate
            - 'auc' : Area Under Curve normalization
            - 'standard' : Z-score standardization
        """
        self.method = method.lower()
        self.scaling_factor = None
        self.params_used = {}

        # Проверка метода
        valid_methods = ['vector', 'minmax', 'snv', 'auc', 'standard']
        if self.method not in valid_methods:
            raise ValueError(f"Неизвестный метод: {method}. "
                             f"Доступные: {', '.join(valid_methods)}")

    def normalize(self,
                  wave: np.ndarray,
                  intensity: np.ndarray,
                  **kwargs) -> Tuple[np.ndarray, Dict]:
        """
        Применение нормировки к спектру.

        Parameters
        ----------
        wave : np.ndarray
            Массив волновых чисел.
        intensity : np.ndarray
            Массив интенсивностей.
        **kwargs
            Дополнительные параметры для выбранного метода.

        Returns
        -------
        normalized : np.ndarray
            Нормированный спектр.
        info : dict
            Информация о применённых параметрах.
        """
        # Преобразование в numpy array с float64
        intensity = np.asarray(intensity, dtype=np.float64).flatten()
        wave = np.asarray(wave, dtype=np.float64).flatten()

        # Проверка длин массивов
        if len(wave) != len(intensity):
            raise ValueError(f"Длины массивов не совпадают: "
                             f"wave={len(wave)}, intensity={len(intensity)}")

        # Проверка на пустой или нулевой спектр
        if len(intensity) == 0:
            raise ValueError("Пустой спектр!")

        if np.all(intensity == 0):
            warnings.warn("Все значения интенсивности равны нулю!")
            return intensity.copy(), {'method': self.method, 'scaling_factor': 1.0}

        # Выбор метода нормировки
        if self.method == 'vector':
            normalized, info = self._vector_normalize(intensity, **kwargs)
        elif self.method == 'minmax':
            normalized, info = self._minmax_normalize(intensity, **kwargs)
        elif self.method == 'snv':
            normalized, info = self._snv_normalize(intensity, **kwargs)
        elif self.method == 'auc':
            normalized, info = self._auc_normalize(intensity, **kwargs)
        elif self.method == 'standard':
            normalized, info = self._standard_normalize(intensity, **kwargs)
        else:
            raise ValueError(f"Неизвестный метод: {self.method}")

        # Сохранение параметров
        self.scaling_factor = info.get('scaling_factor', 1.0)
        self.params_used = info

        return normalized, info

    def _vector_normalize(self,
                          intensity: np.ndarray,
                          epsilon: float = 1e-10) -> Tuple[np.ndarray, Dict]:
        """
        Vector Normalization (L2 norm).

        Приводит спектр к единичной длине. Рекомендуется для ML.
        Формула: normalized = intensity / ||intensity||_2

        Parameters
        ----------
        intensity : np.ndarray
            Интенсивность спектра.
        epsilon : float
            Малое число для избежания деления на ноль.

        Returns
        -------
        normalized : np.ndarray
            Нормированный спектр.
        info : dict
            Параметры нормировки.
        """
        # Вычисление L2 нормы
        norm = np.linalg.norm(intensity)

        if norm < epsilon:
            warnings.warn(f"L2 норма ({norm}) близка к нулю! "
                          f"Возвращается исходный спектр.")
            normalized = intensity.copy()
            scaling_factor = 1.0
        else:
            normalized = intensity / norm
            scaling_factor = norm

        info = {
            'method': 'vector',
            'norm': float(norm),
            'scaling_factor': float(scaling_factor),
            'min': float(np.min(normalized)),
            'max': float(np.max(normalized)),
            'mean': float(np.mean(normalized)),
            'std': float(np.std(normalized))
        }

        return normalized, info

    def _minmax_normalize(self,
                          intensity: np.ndarray,
                          feature_range: Tuple[float, float] = (0.0, 1.0),
                          epsilon: float = 1e-10) -> Tuple[np.ndarray, Dict]:
        """
        Min-Max Scaling.

        Приводит спектр к диапазону [0, 1] или другому указанному диапазону.
        Формула: normalized = (intensity - min) / (max - min)

        Parameters
        ----------
        intensity : np.ndarray
            Интенсивность спектра.
        feature_range : tuple
            Диапазон значений (min, max).
        epsilon : float
            Малое число для избежания деления на ноль.

        Returns
        -------
        normalized : np.ndarray
            Нормированный спектр.
        info : dict
            Параметры нормировки.
        """
        min_val = np.min(intensity)
        max_val = np.max(intensity)
        range_val = max_val - min_val

        if range_val < epsilon:
            warnings.warn(f"Диапазон ({range_val}) близок к нулю! "
                          f"Возвращается исходный спектр.")
            normalized = intensity.copy()
            scaling_factor = 1.0
        else:
            scaling_factor = range_val
            normalized = (intensity - min_val) / scaling_factor
            normalized = normalized * (feature_range[1] - feature_range[0]) + feature_range[0]

        info = {
            'method': 'minmax',
            'original_min': float(min_val),
            'original_max': float(max_val),
            'original_range': float(range_val),
            'scaling_factor': float(scaling_factor),
            'feature_range': feature_range,
            'min': float(np.min(normalized)),
            'max': float(np.max(normalized))
        }

        return normalized, info

    def _snv_normalize(self,
                       intensity: np.ndarray,
                       epsilon: float = 1e-10) -> Tuple[np.ndarray, Dict]:
        """
        Standard Normal Variate (SNV).

        Центрирование и масштабирование каждого спектра.
        Устраняет эффекты рассеяния света.
        Формула: normalized = (intensity - mean) / std

        Parameters
        ----------
        intensity : np.ndarray
            Интенсивность спектра.
        epsilon : float
            Малое число для избежания деления на ноль.

        Returns
        -------
        normalized : np.ndarray
            Нормированный спектр.
        info : dict
            Параметры нормировки.
        """
        mean = np.mean(intensity)
        std = np.std(intensity)

        if std < epsilon:
            warnings.warn(f"Стандартное отклонение ({std}) близко к нулю!")
            normalized = intensity - mean
            scaling_factor = 1.0
        else:
            normalized = (intensity - mean) / std
            scaling_factor = std

        info = {
            'method': 'snv',
            'original_mean': float(mean),
            'original_std': float(std),
            'scaling_factor': float(scaling_factor),
            'min': float(np.min(normalized)),
            'max': float(np.max(normalized)),
            'mean': float(np.mean(normalized)),
            'std': float(np.std(normalized))
        }

        return normalized, info

    def _auc_normalize(self,
                       intensity: np.ndarray,
                       epsilon: float = 1e-10) -> Tuple[np.ndarray, Dict]:
        """
        Area Under Curve (AUC) normalization.

        Нормировка на общую площадь под спектром.
        Формула: normalized = intensity / AUC

        Parameters
        ----------
        intensity : np.ndarray
            Интенсивность спектра.
        epsilon : float
            Малое число для избежания деления на ноль.

        Returns
        -------
        normalized : np.ndarray
            Нормированный спектр.
        info : dict
            Параметры нормировки.
        """
        # Используем только положительные значения для площади
        intensity_positive = np.maximum(intensity, 0)
        auc = np.trapz(intensity_positive)

        if auc < epsilon:
            warnings.warn(f"Площадь под кривой ({auc}) близка к нулю!")
            normalized = intensity.copy()
            scaling_factor = 1.0
        else:
            normalized = intensity / auc
            scaling_factor = auc

        info = {
            'method': 'auc',
            'auc': float(auc),
            'scaling_factor': float(scaling_factor),
            'min': float(np.min(normalized)),
            'max': float(np.max(normalized))
        }

        return normalized, info

    def _standard_normalize(self,
                            intensity: np.ndarray,
                            epsilon: float = 1e-10) -> Tuple[np.ndarray, Dict]:
        """
        Z-score standardization.

        Приводит спектр к нулевому среднему и единичному стандартному отклонению.
        Формула: normalized = (intensity - mean) / std

        Parameters
        ----------
        intensity : np.ndarray
            Интенсивность спектра.
        epsilon : float
            Малое число для избежания деления на ноль.

        Returns
        -------
        normalized : np.ndarray
            Нормированный спектр.
        info : dict
            Параметры нормировки.
        """
        mean = np.mean(intensity)
        std = np.std(intensity)

        if std < epsilon:
            warnings.warn(f"Стандартное отклонение ({std}) близко к нулю!")
            normalized = intensity - mean
            scaling_factor = 1.0
        else:
            normalized = (intensity - mean) / std
            scaling_factor = std

        info = {
            'method': 'standard',
            'original_mean': float(mean),
            'original_std': float(std),
            'scaling_factor': float(scaling_factor),
            'min': float(np.min(normalized)),
            'max': float(np.max(normalized)),
            'mean': float(np.mean(normalized)),
            'std': float(np.std(normalized))
        }

        return normalized, info

    def inverse_transform(self,
                          normalized_intensity: np.ndarray) -> np.ndarray:
        """
        Обратное преобразование к исходному масштабу.

        Parameters
        ----------
        normalized_intensity : np.ndarray
            Нормированный спектр.

        Returns
        -------
        original_intensity : np.ndarray
            Спектр в исходном масштабе.
        """
        if self.scaling_factor is None:
            warnings.warn("scaling_factor не установлен! "
                          "Возвращается нормированный спектр.")
            return normalized_intensity

        normalized_intensity = np.asarray(normalized_intensity, dtype=np.float64)

        if self.method == 'vector':
            return normalized_intensity * self.scaling_factor
        elif self.method == 'minmax':
            original_min = self.params_used.get('original_min', 0)
            return normalized_intensity * self.scaling_factor + original_min
        elif self.method == 'snv' or self.method == 'standard':
            original_mean = self.params_used.get('original_mean', 0)
            return normalized_intensity * self.scaling_factor + original_mean
        elif self.method == 'auc':
            return normalized_intensity * self.scaling_factor
        else:
            return normalized_intensity

    def visualize_normalization(self,
                                wave: np.ndarray,
                                intensity_raw: np.ndarray,
                                intensity_normalized: np.ndarray,
                                figsize: Tuple[int, int] = (14, 6)) -> None:
        """
        Визуализация результатов нормировки.

        Parameters
        ----------
        wave : np.ndarray
            Волновые числа.
        intensity_raw : np.ndarray
            Исходная интенсивность.
        intensity_normalized : np.ndarray
            Нормированная интенсивность.
        figsize : tuple
            Размер фигуры.
        """
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=figsize)
        fig.suptitle(f'Нормировка спектра ({self.method.upper()})',
                     fontsize=14, fontweight='bold')

        # График 1: Сравнение спектров
        axes[0].plot(wave, intensity_raw, 'b-', linewidth=0.5, alpha=0.7, label='Исходный')
        axes[0].plot(wave, intensity_normalized, 'r-', linewidth=1, label='Нормированный')
        axes[0].set_xlabel('Волновое число (cm⁻¹)')
        axes[0].set_ylabel('Интенсивность')
        axes[0].set_title('Сравнение спектров')
        axes[0].legend(fontsize=8)
        axes[0].grid(True, alpha=0.3)

        # График 2: Статистика
        axes[1].axis('off')

        stats = {
            'Метод': self.params_used.get('method', 'N/A'),
            'Исходный min': f'{np.min(intensity_raw):.2f}',
            'Исходный max': f'{np.max(intensity_raw):.2f}',
            'Исходный mean': f'{np.mean(intensity_raw):.2f}',
            'Исходный std': f'{np.std(intensity_raw):.2f}',
            'Нормированный min': f'{np.min(intensity_normalized):.6f}',
            'Нормированный max': f'{np.max(intensity_normalized):.6f}',
            'Нормированный mean': f'{np.mean(intensity_normalized):.6f}',
            'Нормированный std': f'{np.std(intensity_normalized):.6f}',
            'Scaling factor': f'{self.params_used.get("scaling_factor", "N/A"):.6f}'
        }

        text = '\n'.join([f'{k}: {v}' for k, v in stats.items()])
        axes[1].text(0.1, 0.5, text, transform=axes[1].transAxes,
                     fontsize=10, verticalalignment='center',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1].set_title('Статистика нормировки', fontsize=12, fontweight='bold')

        plt.tight_layout()
        plt.show()


def batch_normalization(spectra_list: list,
                        wave_key: str = 'wave',
                        intensity_key: str = 'intensity',
                        method: str = 'vector',
                        **kwargs) -> Tuple[list, list]:
    """
    Пакетная нормировка нескольких спектров.

    Parameters
    ----------
    spectra_list : list of dict
        Список словарей со спектрами.
    wave_key : str
        Ключ для волновых чисел.
    intensity_key : str
        Ключ для интенсивности.
    method : str
        Метод нормировки.
    **kwargs
        Параметры для нормировки.

    Returns
    -------
    normalized_spectra : list of dict
        Нормированные спектры.
    normalization_info : list of dict
        Информация о нормировке для каждого спектра.
    """
    normalizer = SpectrumNormalizer(method=method)
    normalized_spectra = []
    normalization_info = []

    for i, spectrum in enumerate(spectra_list):
        wave = spectrum[wave_key]
        intensity = spectrum[intensity_key]

        normalized, info = normalizer.normalize(wave, intensity, **kwargs)

        normalized_spectrum = spectrum.copy()
        normalized_spectrum[f'{intensity_key}_normalized'] = normalized

        info['spectrum_idx'] = i
        normalized_spectra.append(normalized_spectrum)
        normalization_info.append(info)

    return normalized_spectra, normalization_info


# Тестирование модуля
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=" * 60)
    print("🧪 ТЕСТИРОВАНИЕ МОДУЛЯ НОРМИРОВКИ")
    print("=" * 60)

    # Генерация тестового спектра
    np.random.seed(42)
    wave = np.linspace(200, 2000, 1000)
    intensity = 1000 + 500 * np.exp(-((wave - 1000) / 50) ** 2) + np.random.normal(0, 50, len(wave))

    # Тестирование разных методов
    methods = ['vector', 'minmax', 'snv', 'auc', 'standard']

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()

    # Исходный спектр
    axes[0].plot(wave, intensity, 'b-', linewidth=0.5)
    axes[0].set_xlabel('Wavenumber (cm⁻¹)')
    axes[0].set_ylabel('Intensity')
    axes[0].set_title('Исходный спектр')
    axes[0].grid(True, alpha=0.3)

    print("\n📊 Результаты нормировки:")
    print("-" * 60)

    for idx, method in enumerate(methods):
        normalizer = SpectrumNormalizer(method=method)
        normalized, info = normalizer.normalize(wave, intensity)

        axes[idx + 1].plot(wave, normalized, 'r-', linewidth=0.5)
        axes[idx + 1].set_xlabel('Wavenumber (cm⁻¹)')
        axes[idx + 1].set_ylabel('Intensity')
        axes[idx + 1].set_title(f'{method.upper()}\nmin={info["min"]:.4f}, max={info["max"]:.4f}')
        axes[idx + 1].grid(True, alpha=0.3)

        print(f"{method:10s}: min={info['min']:.6f}, max={info['max']:.6f}, "
              f"mean={info.get('mean', 'N/A'):.6f}, std={info.get('std', 'N/A'):.6f}")

    plt.tight_layout()
    plt.savefig('normalization_test.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Тест обратного преобразования
    print("\n🔄 Тест обратного преобразования:")
    normalizer = SpectrumNormalizer(method='vector')
    normalized, info = normalizer.normalize(wave, intensity)
    restored = normalizer.inverse_transform(normalized)

    reconstruction_error = np.mean(np.abs(intensity - restored))
    print(f"   Ошибка восстановления: {reconstruction_error:.6f}")

    print("\n✅ Тестирование завершено успешно!")
    print("=" * 60)