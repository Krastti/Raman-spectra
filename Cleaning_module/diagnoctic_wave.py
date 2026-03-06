import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from remove_rays import CosmicRayRemover
from baseline_correction import FluorescenceCorrector
from smoothing import SpectrumSmoother
from normalization import SpectrumNormalizer

file1 = "../data/cortex_control_1group_633nm_center1500_obj100_power100_1s_5acc_map35x15_step2_place4_1.txt"
file2 = "../data/mk1/cortex_control_1group_633nm_center1500_obj100_power100_1s_5acc_map35x15_step2_place4_2.txt"
file3 = "../data/mk1/cortex_control_1group_633nm_center1500_obj100_power100_1s_5acc_map35x15_step2_place5_1.txt"

# Загрузка данных
df = pd.read_csv(file3, sep='\t', comment='#', header=None, names=['X', 'Y', 'Wave', 'Intensity'])
wave = df['Wave'].values
intensity = df['Intensity'].values

# ============================================
# ЭТАП 1: Удаление космических лучей
# ============================================
print("🔧 ЭТАП 1: Удаление космических лучей")
remover = CosmicRayRemover(window_size=5, threshold=5.0, method='median')
intensity_cr_removed, info_cr = remover.remove_cosmic_rays(
    wave=wave, intensity=intensity, method='interpolation'
)
print(f"✅ Найдено космических лучей: {info_cr['n_cosmic_rays']}")

# ============================================
# ЭТАП 2: Коррекция флуоресцентного фона
# ============================================
print("\n🔧 ЭТАП 2: Полиномиальная коррекция фона")
corrector = FluorescenceCorrector()
baseline, intensity_corrected = corrector.correct_baseline(
    wave=wave, intensity=intensity_cr_removed,
    method='poly', degree=5, mask_peaks=True, peak_threshold=2.0
)

if np.min(intensity_corrected) < 0:
    intensity_corrected = intensity_corrected - np.min(intensity_corrected) + 1

print(f"✅ Средний фон: {np.mean(baseline):.1f}")

# ============================================
# ЭТАП 3: Сглаживание
# ============================================
print("\n🔧 ЭТАП 3: Сглаживание")
smoother = SpectrumSmoother(method='savgol')
intensity_smoothed, info_smooth = smoother.smooth(
    wave=wave, intensity=intensity_corrected,
    window_length=11, polyorder=3
)
print(f"✅ Метод: {info_smooth['method']}")

# ============================================
# ЭТАП 4: Нормировка
# ============================================
print("\n🔧 ЭТАП 4: Нормировка")
normalizer = SpectrumNormalizer(method='minmax')
intensity_normalized, info_norm = normalizer.normalize(
    wave=wave, intensity=intensity_smoothed
)
print(f"✅ Метод: {info_norm['method']}")
print(f"   Min: {info_norm['min']:.6f}")
print(f"   Max: {info_norm['max']:.6f}")

# ============================================
# ЭТАП 5: ВИЗУАЛИЗАЦИЯ (ИСПРАВЛЕНО)
# ============================================
print("\n📊 ВИЗУАЛИЗАЦИЯ")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle('Полный пайплайн предобработки', fontsize=14, fontweight='bold')


# Функция для корректной отрисовки спектров без соединительных линий
def plot_spectra_no_artifacts(ax, wave, intensity, color, label, alpha=0.5):
    """
    Отрисовка спектров без соединительных линий между разными спектрами.
    """
    df_temp = pd.DataFrame({'wave': wave, 'intensity': intensity})

    # Группируем по уникальным длинам волн (каждый спектр)
    unique_waves = df_temp['wave'].nunique()
    n_spectra = len(df_temp) // unique_waves if unique_waves > 0 else 1

    for i in range(n_spectra):
        start_idx = i * unique_waves
        end_idx = (i + 1) * unique_waves
        wave_segment = wave[start_idx:end_idx]
        intensity_segment = intensity[start_idx:end_idx]
        ax.plot(wave_segment, intensity_segment, color=color,
                linewidth=0.5, alpha=alpha)


# 1. Исходный
plot_spectra_no_artifacts(axes[0, 0], wave, intensity, 'blue', 'Raw', alpha=0.5)
axes[0, 0].set_title('1. Исходный спектр')
axes[0, 0].set_xlabel('Wavenumber (cm⁻¹)')
axes[0, 0].set_ylabel('Intensity')
axes[0, 0].grid(True, alpha=0.3)

# 2. После удаления космических лучей
plot_spectra_no_artifacts(axes[0, 1], wave, intensity_cr_removed, 'green', 'CR Removed', alpha=0.5)
axes[0, 1].set_title('2. После удаления косм. лучей')
axes[0, 1].set_xlabel('Wavenumber (cm⁻¹)')
axes[0, 1].set_ylabel('Intensity')
axes[0, 1].grid(True, alpha=0.3)

# 3. После коррекции фона
plot_spectra_no_artifacts(axes[0, 2], wave, intensity_corrected, 'orange', 'Baseline Corrected', alpha=0.5)
axes[0, 2].set_title('3. После коррекции фона')
axes[0, 2].set_xlabel('Wavenumber (cm⁻¹)')
axes[0, 2].set_ylabel('Intensity')
axes[0, 2].grid(True, alpha=0.3)

# 4. После сглаживания
plot_spectra_no_artifacts(axes[1, 0], wave, intensity_smoothed, 'purple', 'Smoothed', alpha=0.5)
axes[1, 0].set_title('4. После сглаживания')
axes[1, 0].set_xlabel('Wavenumber (cm⁻¹)')
axes[1, 0].set_ylabel('Intensity')
axes[1, 0].grid(True, alpha=0.3)

# 5. После нормировки
plot_spectra_no_artifacts(axes[1, 1], wave, intensity_normalized, 'red', 'Normalized', alpha=0.5)
axes[1, 1].set_title('5. После нормировки (Min-Max)')
axes[1, 1].set_xlabel('Wavenumber (cm⁻¹)')
axes[1, 1].set_ylabel('Normalized Intensity')
axes[1, 1].grid(True, alpha=0.3)

# 6. Сравнение (две оси Y)
ax6 = axes[1, 2]
plot_spectra_no_artifacts(ax6, wave, intensity, 'blue', 'Raw', alpha=0.3)
ax6.set_xlabel('Wavenumber (cm⁻¹)')
ax6.set_ylabel('Raw Intensity', color='blue')
ax6.set_title('Сравнение (две оси Y)')
ax6.tick_params(axis='y', labelcolor='blue')

ax6_twin = ax6.twinx()
plot_spectra_no_artifacts(ax6_twin, wave, intensity_normalized, 'red', 'Normalized', alpha=0.5)
ax6_twin.set_ylabel('Normalized Intensity', color='red')
ax6_twin.tick_params(axis='y', labelcolor='red')

lines1, labels1 = ax6.get_legend_handles_labels()
lines2, labels2 = ax6_twin.get_legend_handles_labels()
ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=8)
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('preprocessing_full_fixed.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# ПАКЕТНАЯ ОБРАБОТКА
# ============================================
print("\n📁 ПАКЕТНАЯ ОБРАБОТКА")
cleaned_spectra = {}

for (x, y), group in df.groupby(['X', 'Y']):
    wave_spectrum = group['Wave'].values
    intensity_spectrum = group['Intensity'].values

    # 1. Космические лучи
    intensity_cr, info_cr = remover.remove_cosmic_rays(
        wave=wave_spectrum, intensity=intensity_spectrum, method='interpolation'
    )

    # 2. Коррекция фона
    baseline_spec, intensity_final = corrector.correct_baseline(
        wave=wave_spectrum, intensity=intensity_cr,
        method='poly', degree=5, mask_peaks=True
    )

    # 3. Сдвиг отрицательных
    if np.min(intensity_final) < 0:
        intensity_final = intensity_final - np.min(intensity_final) + 1

    # 4. Сглаживание
    intensity_final_smooth, _ = smoother.smooth(
        wave=wave_spectrum, intensity=intensity_final,
        window_length=11, polyorder=3
    )

    # 5. Нормировка
    normalizer_spec = SpectrumNormalizer(method='minmax')
    intensity_final_norm, info_norm = normalizer_spec.normalize(
        wave=wave_spectrum, intensity=intensity_final_smooth
    )

    cleaned_spectra[(x, y)] = {
        'wave': wave_spectrum,
        'intensity_raw': intensity_spectrum,
        'intensity_cr_removed': intensity_cr,
        'intensity_baseline_corrected': intensity_final,
        'intensity_smoothed': intensity_final_smooth,
        'intensity_final': intensity_final_norm,
        'baseline': baseline_spec,
        'cosmic_rays': info_cr['n_cosmic_rays'],
        'norm_min': info_norm['min'],
        'norm_max': info_norm['max']
    }

print(f"✅ Обработано спектров: {len(cleaned_spectra)}")

# ============================================
# ВИЗУАЛИЗАЦИЯ НЕСКОЛЬКИХ СПЕКТРОВ (ИСПРАВЛЕНО)
# ============================================
print("\n📊 ВИЗУАЛИЗАЦИЯ ОТДЕЛЬНЫХ СПЕКТРОВ")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Примеры обработанных спектров', fontsize=12, fontweight='bold')

for idx, ((x, y), data) in enumerate(list(cleaned_spectra.items())[:4]):
    ax = axes[idx // 2, idx % 2]

    # Каждый спектр рисуется отдельно - без соединительных линий
    ax.plot(data['wave'], data['intensity_raw'], 'b-', linewidth=0.5, alpha=0.5, label='Raw')
    ax.plot(data['wave'], data['intensity_final'], 'r-', linewidth=1, label='Processed')
    ax.plot(data['wave'], data['baseline'], 'g--', linewidth=1, alpha=0.7, label='Baseline')

    ax.set_xlabel('Wavenumber (cm⁻¹)')
    ax.set_ylabel('Intensity')
    ax.set_title(f'Координаты: ({x:.1f}, {y:.1f})\nКосм. лучи: {data["cosmic_rays"]}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('sample_spectra_final.png', dpi=150, bbox_inches='tight')
plt.show()

# ============================================
# ДОПОЛНИТЕЛЬНО: Визуализация с offset для наглядности
# ============================================
print("\n📊 ВИЗУАЛИЗАЦИЯ С OFFSET (для наглядности)")

fig, ax = plt.subplots(figsize=(12, 8))

# Показываем несколько спектров со смещением для наглядности
spectra_to_show = list(cleaned_spectra.items())[:10]
offset = 2000  # Смещение между спектрами

for idx, ((x, y), data) in enumerate(spectra_to_show):
    intensity_offset = data['intensity_final'] + idx * offset
    ax.plot(data['wave'], intensity_offset, linewidth=0.8, alpha=0.7,
            label=f'({x:.0f}, {y:.0f})' if idx < 5 else "")

ax.set_xlabel('Wavenumber (cm⁻¹)')
ax.set_ylabel('Intensity (offset for clarity)')
ax.set_title('Обработанные спектры со смещением')
ax.legend(fontsize=6, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('spectra_offset_view.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("✅ ПРЕДОБРАБОТКА ЗАВЕРШЕНА!")
print("=" * 60)
print("\n📁 Сохранённые файлы:")
print("   - preprocessing_full_fixed.png")
print("   - sample_spectra_final.png")
print("   - spectra_offset_view.png")
print("\n🎯 Готово для машинного обучения!")