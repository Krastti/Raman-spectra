import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.signal import savgol_filter
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict, List, Union
import warnings


class FluorescenceCorrector:
    """Класс для коррекции флуоресцентного фона в Рамановских спектрах."""

    def __init__(self):
        self.baseline = None
        self.corrected_spectrum = None
        self.method_used = None

    def correct_baseline(self,
                         wave: np.ndarray,
                         intensity: np.ndarray,
                         method: str = 'als',
                         **kwargs) -> Tuple[np.ndarray, np.ndarray]:
        """Коррекция базовой линии (флуоресцентного фона)."""
        self.method_used = method

        # Преобразуем intensity в float64
        intensity = np.asarray(intensity, dtype=np.float64)

        if method == 'als':
            baseline = self._als_baseline(intensity, **kwargs)
        elif method == 'poly':
            baseline = self._poly_baseline(wave, intensity, **kwargs)
        elif method == 'rolling':
            baseline = self._rolling_ball_baseline(intensity, **kwargs)
        elif method == 'snip':
            baseline = self._snip_baseline(intensity, **kwargs)
        elif method == 'spline':
            baseline = self._spline_baseline(wave, intensity, **kwargs)
        else:
            raise ValueError(f"Неизвестный метод: {method}")

        corrected = intensity - baseline
        self.baseline = baseline
        self.corrected_spectrum = corrected

        return baseline, corrected

    def _als_baseline(self,
                      intensity: np.ndarray,
                      lam: float = 1e6,
                      p: float = 0.01,
                      niter: int = 10) -> np.ndarray:
        """
        Asymmetric Least Squares (ALS) baseline correction.
        ✅ Исправленная версия без предупреждений
        """
        L = len(intensity)
        intensity = np.asarray(intensity, dtype=np.float64)

        # ✅ Исправление 1: явный dtype=np.float64
        # ✅ Исправление 2: формат 'csr' для эффективных операций
        D = sparse.diags([1, -2, 1], [0, -1, -2],
                         shape=(L, L - 2),
                         dtype=np.float64,
                         format='csr')

        w = np.ones(L, dtype=np.float64)

        for i in range(niter):
            # ✅ Исправление 3: W в формате CSR
            W = sparse.spdiags(w, 0, L, L, format='csr')

            # Z = W + lam * D * D.T
            Z = W + lam * (D @ D.T)

            # ✅ Исправление 4: преобразование в CSC для spsolve
            Z = Z.tocsc()

            z = spsolve(Z, w * intensity)
            w = p * (intensity > z) + (1 - p) * (intensity < z)

        return z

    def _poly_baseline(self,
                       wave: np.ndarray,
                       intensity: np.ndarray,
                       degree: int = 5,
                       mask_peaks: bool = True,
                       peak_threshold: float = 2.0) -> np.ndarray:
        """Полиномиальная аппроксимация базовой линии."""
        if mask_peaks:
            mask = self._create_peak_mask(intensity, threshold=peak_threshold)
            wave_fit = wave[mask]
            intensity_fit = intensity[mask]
        else:
            wave_fit = wave
            intensity_fit = intensity

        coeffs = np.polyfit(wave_fit, intensity_fit, degree)
        baseline = np.polyval(coeffs, wave)

        return baseline

    def _rolling_ball_baseline(self,
                               intensity: np.ndarray,
                               window_size: int = 100) -> np.ndarray:
        """Rolling ball baseline correction."""
        from scipy.ndimage import minimum_filter1d

        baseline = minimum_filter1d(intensity, size=window_size)
        baseline = savgol_filter(baseline, window_length=11, polyorder=3)

        return baseline

    def _snip_baseline(self,
                       intensity: np.ndarray,
                       niter: int = 100,
                       smoothing: bool = True) -> np.ndarray:
        """SNIP algorithm."""
        c = np.log(np.log(intensity + 1) + 1)
        baseline_log = c.copy()

        for i in range(niter):
            for j in range(1, len(c) - 1):
                avg_neighbors = (baseline_log[j - 1] + baseline_log[j + 1]) / 2
                baseline_log[j] = min(baseline_log[j], avg_neighbors)

        baseline = np.exp(np.exp(baseline_log) - 1) - 1

        if smoothing:
            baseline = savgol_filter(baseline, window_length=11, polyorder=3)

        return baseline

    def _spline_baseline(self,
                         wave: np.ndarray,
                         intensity: np.ndarray,
                         smoothing_factor: float = 0.5,
                         n_points: int = 50) -> np.ndarray:
        """Сплайновая аппроксимация."""
        mask = self._create_peak_mask(intensity, threshold=1.5)
        wave_fit = wave[mask]
        intensity_fit = intensity[mask]

        if len(wave_fit) < 10:
            wave_fit = wave
            intensity_fit = intensity

        spline = UnivariateSpline(wave_fit, intensity_fit,
                                  s=smoothing_factor * len(wave_fit))
        baseline = spline(wave)

        return baseline

    def _create_peak_mask(self,
                          intensity: np.ndarray,
                          threshold: float = 2.0) -> np.ndarray:
        """Создание маски для исключения пиков."""
        from scipy.signal import medfilt

        median = medfilt(intensity, kernel_size=5)
        residual = intensity - median
        mad = np.median(np.abs(residual - np.median(residual)))
        sigma = 1.4826 * mad

        mask = intensity < (median + threshold * sigma)

        return mask

    def visualize_correction(self,
                             wave: np.ndarray,
                             intensity: np.ndarray,
                             baseline: np.ndarray = None,
                             corrected: np.ndarray = None,
                             figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """Визуализация коррекции флуоресцентного фона."""
        if baseline is None:
            baseline = self.baseline
        if corrected is None:
            corrected = self.corrected_spectrum

        if baseline is None or corrected is None:
            raise ValueError("Базлайн и скорректированный спектр должны быть предоставлены")

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Коррекция флуоресцентного фона', fontsize=14, fontweight='bold')

        # График 1: Исходный спектр с базлайном
        axes[0, 0].plot(wave, intensity, 'b-', linewidth=0.5, label='Исходный спектр', alpha=0.7)
        axes[0, 0].plot(wave, baseline, 'r-', linewidth=2, label='Флуоресцентный фон')
        axes[0, 0].fill_between(wave, baseline, intensity, alpha=0.3, color='green',
                                label='Рамановский сигнал')
        axes[0, 0].set_xlabel('Волновое число (cm⁻¹)')
        axes[0, 0].set_ylabel('Интенсивность')
        axes[0, 0].set_title('Исходный спектр и оцененный фон')
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, alpha=0.3)

        # График 2: Скорректированный спектр
        axes[0, 1].plot(wave, corrected, 'g-', linewidth=0.5)
        axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.3)
        axes[0, 1].set_xlabel('Волновое число (cm⁻¹)')
        axes[0, 1].set_ylabel('Интенсивность')
        axes[0, 1].set_title('Спектр после коррекции')
        axes[0, 1].grid(True, alpha=0.3)

        # График 3: Распределение интенсивностей
        axes[1, 0].hist(intensity, bins=50, alpha=0.5, label='Исходный', color='blue')
        axes[1, 0].hist(corrected, bins=50, alpha=0.5, label='Скорректированный', color='green')
        axes[1, 0].axvline(x=0, color='k', linestyle='--', alpha=0.3)
        axes[1, 0].set_xlabel('Интенсивность')
        axes[1, 0].set_ylabel('Частота')
        axes[1, 0].set_title('Распределение интенсивностей')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # График 4: Статистика коррекции
        correction_stats = {
            'Метод': self.method_used or 'N/A',
            'Средний фон': f'{np.mean(baseline):.1f}',
            'std фона': f'{np.std(baseline):.1f}',
            'Отриц. значения': f'{np.sum(corrected < 0)}',
            'Min скорр.': f'{np.min(corrected):.1f}',
            'Max скорр.': f'{np.max(corrected):.1f}'
        }

        axes[1, 1].axis('off')
        text = '\n'.join([f'{k}: {v}' for k, v in correction_stats.items()])
        axes[1, 1].text(0.1, 0.5, text, transform=axes[1, 1].transAxes,
                        fontsize=11, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].set_title('Статистика коррекции', fontsize=12, fontweight='bold')

        plt.tight_layout()
        return fig


# Пример использования
if __name__ == "__main__":
    import pandas as pd

    # Загрузка данных
    file = "cortex_control_1group_633nm_center1500_obj100_power100_1s_5acc_map35x15_step2_place4_1.txt"
    df = pd.read_csv(file, sep='\t', comment='#', header=None,
                     names=['X', 'Y', 'Wave', 'Intensity'])

    # Берем первый спектр
    wave = df['Wave'].values
    intensity = df['Intensity'].values

    # Коррекция
    corrector = FluorescenceCorrector()
    baseline, corrected = corrector.correct_baseline(wave, intensity,
                                                     method='als',
                                                     lam=1e6, p=0.01)

    print("✅ Коррекция завершена без предупреждений!")
    print(f"Метод: {corrector.method_used}")
    print(f"Средний фон: {np.mean(baseline):.1f}")
    print(f"Отрицательных значений: {np.sum(corrected < 0)}")

    # Визуализация
    fig = corrector.visualize_correction(wave, intensity, baseline, corrected)
    plt.show()