import numpy as np
from scipy.signal import medfilt
from scipy.ndimage import median_filter
import warnings
from typing import Tuple, Optional, Dict, List


class CosmicRayRemover:
    """
    Класс для обнаружения и удаления космических лучей из Рамановских спектров.

    Космические лучи проявляются как узкие пики с аномально высокой интенсивностью,
    которые необходимо удалить перед дальнейшим анализом.
    """

    def __init__(self,
                 window_size: int = 5,
                 threshold: float = 5.0,
                 method: str = 'median'):
        """
        Инициализация детектора космических лучей.

        Parameters
        ----------
        window_size : int
            Размер окна для медианного фильтра (должно быть нечетным числом).
            Рекомендуемое значение: 5-9.
        threshold : float
            Порог обнаружения в единицах стандартного отклонения.
            Точки с отклонением выше этого порога считаются космическими лучами.
            Рекомендуемое значение: 4.0-6.0.
        method : str
            Метод обнаружения: 'median' (медианный фильтр) или 'derivative' (производная).
        """
        self.window_size = window_size if window_size % 2 == 1 else window_size + 1
        self.threshold = threshold
        self.method = method

        if self.window_size < 3:
            self.window_size = 3
            warnings.warn("window_size увеличен до 3 (минимальное нечетное значение)")

    def detect_cosmic_rays(self, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обнаружение космических лучей в спектре.

        Parameters
        ----------
        intensity : np.ndarray
            Массив интенсивностей спектра.

        Returns
        -------
        cosmic_ray_mask : np.ndarray
            Булев массив, где True указывает на наличие космического луча.
        residuals : np.ndarray
            Остатки (разница между исходным сигналом и медианным фильтром).
        """
        intensity = np.asarray(intensity, dtype=np.float64)

        if self.method == 'median':
            return self._detect_median_method(intensity)
        elif self.method == 'derivative':
            return self._detect_derivative_method(intensity)
        else:
            raise ValueError(f"Неизвестный метод: {method}. Используйте 'median' или 'derivative'")

    def _detect_median_method(self, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обнаружение космических лучей с помощью медианного фильтра.
        """
        # Применяем медианный фильтр
        if len(intensity) < self.window_size:
            # Если сигнал слишком короткий, используем доступную длину
            window = len(intensity) if len(intensity) % 2 == 1 else len(intensity) - 1
            if window < 3:
                window = 3
            median_signal = median_filter(intensity, size=window)
        else:
            median_signal = median_filter(intensity, size=self.window_size)

        # Вычисляем остатки (разницу между исходным сигналом и медианой)
        residuals = np.abs(intensity - median_signal)

        # Вычисляем стандартное отклонение остатков
        std_residuals = np.std(residuals)
        mean_residuals = np.mean(residuals)

        # Создаем маску космических лучей
        # Точки, где отклонение превышает порог, считаются космическими лучами
        cosmic_ray_mask = residuals > (mean_residuals + self.threshold * std_residuals)

        return cosmic_ray_mask, residuals

    def _detect_derivative_method(self, intensity: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Обнаружение космических лучей с помощью анализа первой производной.
        """
        # Вычисляем первую производную
        derivative = np.diff(intensity)

        # Космические лучи создают резкие скачки в обеих направлениях
        # Поэтому смотрим на абсолютное значение производной
        abs_derivative = np.abs(derivative)

        # Применяем медианный фильтр к производной
        if len(abs_derivative) >= self.window_size:
            median_derivative = median_filter(abs_derivative, size=self.window_size)
        else:
            median_derivative = np.median(abs_derivative)

        # Находим точки с аномально высокой производной
        residuals = abs_derivative - median_derivative
        std_residuals = np.std(residuals)

        # Маска для производной (на 1 точку меньше)
        derivative_mask = residuals > (self.threshold * std_residuals)

        # Преобразуем маску для исходного сигнала
        cosmic_ray_mask = np.zeros(len(intensity), dtype=bool)
        cosmic_ray_mask[:-1] = derivative_mask
        cosmic_ray_mask[1:] |= derivative_mask  # Учитываем обе стороны скачка

        return cosmic_ray_mask, abs_derivative

    def remove_cosmic_rays(self,
                           wave: np.ndarray,
                           intensity: np.ndarray,
                           method: str = 'interpolation') -> Tuple[np.ndarray, Dict]:
        """
        Удаление космических лучей из спектра.

        Parameters
        ----------
        wave : np.ndarray
            Массив волновых чисел.
        intensity : np.ndarray
            Массив интенсивностей.
        method : str
            Метод замены: 'interpolation' (линейная интерполяция),
                         'median' (замена медианой),
                         'mean' (замена средним).

        Returns
        -------
        corrected_intensity : np.ndarray
            Исправленный массив интенсивностей.
        info : dict
            Словарь с информацией о найденных космических лучах.
        """
        wave = np.asarray(wave, dtype=np.float64)
        intensity = np.asarray(intensity, dtype=np.float64)

        if len(wave) != len(intensity):
            raise ValueError("Длины массивов wave и intensity должны совпадать")

        # Обнаруживаем космические лучи
        cosmic_ray_mask, residuals = self.detect_cosmic_rays(intensity)

        # Копируем интенсивность для коррекции
        corrected_intensity = intensity.copy()

        # Заменяем космические лучи
        if np.any(cosmic_ray_mask):
            corrected_intensity = self._replace_cosmic_rays(
                wave, intensity, cosmic_ray_mask, method
            )

        # Собираем информацию
        info = {
            'n_cosmic_rays': int(np.sum(cosmic_ray_mask)),
            'indices': np.where(cosmic_ray_mask)[0].tolist(),
            'wave_positions': wave[cosmic_ray_mask].tolist() if np.any(cosmic_ray_mask) else [],
            'max_intensity': float(np.max(intensity[cosmic_ray_mask])) if np.any(cosmic_ray_mask) else 0.0,
            'percentage': float(100.0 * np.sum(cosmic_ray_mask) / len(intensity))
        }

        return corrected_intensity, info

    def _replace_cosmic_rays(self,
                             wave: np.ndarray,
                             intensity: np.ndarray,
                             mask: np.ndarray,
                             method: str) -> np.ndarray:
        """
        Замена точек с космическими лучами.
        """
        corrected = intensity.copy()
        indices = np.where(mask)[0]

        for idx in indices:
            if method == 'interpolation':
                # Линейная интерполяция между соседними точками
                corrected[idx] = self._interpolate_point(wave, intensity, idx, mask)
            elif method == 'median':
                # Замена медианой соседних точек
                corrected[idx] = self._median_neighbor(intensity, idx, mask)
            elif method == 'mean':
                # Замена средним соседних точек
                corrected[idx] = self._mean_neighbor(intensity, idx, mask)
            else:
                raise ValueError(f"Неизвестный метод замены: {method}")

        return corrected

    def _interpolate_point(self,
                           wave: np.ndarray,
                           intensity: np.ndarray,
                           idx: int,
                           mask: np.ndarray) -> float:
        """
        Интерполяция точки между соседними "хорошими" точками.
        """
        # Ищем ближайшую хорошую точку слева
        left_idx = idx - 1
        while left_idx >= 0 and mask[left_idx]:
            left_idx -= 1

        # Ищем ближайшую хорошую точку справа
        right_idx = idx + 1
        while right_idx < len(intensity) and mask[right_idx]:
            right_idx += 1

        # Если нашли точки с обеих сторон - интерполируем
        if left_idx >= 0 and right_idx < len(intensity):
            # Линейная интерполяция
            fraction = (wave[idx] - wave[left_idx]) / (wave[right_idx] - wave[left_idx])
            return intensity[left_idx] + fraction * (intensity[right_idx] - intensity[left_idx])
        elif left_idx >= 0:
            return intensity[left_idx]
        elif right_idx < len(intensity):
            return intensity[right_idx]
        else:
            return np.median(intensity)

    def _median_neighbor(self, intensity: np.ndarray, idx: int, mask: np.ndarray) -> float:
        """
        Замена точки медианой соседних "хороших" точек.
        """
        # Берем окно вокруг точки
        start = max(0, idx - self.window_size // 2)
        end = min(len(intensity), idx + self.window_size // 2 + 1)

        # Исключаем точки с космическими лучами
        neighbors = intensity[start:end][~mask[start:end]]

        if len(neighbors) > 0:
            return np.median(neighbors)
        else:
            return np.median(intensity)

    def _mean_neighbor(self, intensity: np.ndarray, idx: int, mask: np.ndarray) -> float:
        """
        Замена точки средним соседних "хороших" точек.
        """
        start = max(0, idx - self.window_size // 2)
        end = min(len(intensity), idx + self.window_size // 2 + 1)

        neighbors = intensity[start:end][~mask[start:end]]

        if len(neighbors) > 0:
            return np.mean(neighbors)
        else:
            return np.mean(intensity)


def remove_cosmic_rays_batch(spectra_list: List[Dict],
                             wave_key: str = 'wave',
                             intensity_key: str = 'intensity',
                             **kwargs) -> Tuple[List[Dict], List[Dict]]:
    """
    Пакетная обработка нескольких спектров для удаления космических лучей.

    Parameters
    ----------
    spectra_list : list of dict
        Список словарей со спектрами. Каждый словарь должен содержать
        массивы волновых чисел и интенсивностей.
    wave_key : str
        Ключ для доступа к массиву волновых чисел.
    intensity_key : str
        Ключ для доступа к массиву интенсивностей.
    **kwargs
        Дополнительные параметры для CosmicRayRemover.

    Returns
    -------
    corrected_spectra : list of dict
        Список исправленных спектров.
    all_info : list of dict
        Список информации о найденных космических лучах для каждого спектра.
    """
    remover = CosmicRayRemover(**kwargs)
    corrected_spectra = []
    all_info = []

    for spectrum in spectra_list:
        wave = spectrum[wave_key]
        intensity = spectrum[intensity_key]

        corrected_intensity, info = remover.remove_cosmic_rays(wave, intensity)

        # Создаем копию спектра с исправленной интенсивностью
        corrected_spectrum = spectrum.copy()
        corrected_spectrum[intensity_key] = corrected_intensity

        corrected_spectra.append(corrected_spectrum)
        all_info.append(info)

    return corrected_spectra, all_info