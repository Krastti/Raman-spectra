def save_to_txt(filename, data, mode='w', encoding='utf-8'):
    """
    Записывает данные в текстовый файл.

    Параметры:
    ----------
    filename : str
        Имя файла (например, 'results.txt')
    data : str или list или dict
        Данные для записи
    mode : str
        Режим записи: 'w' (перезапись), 'a' (добавление)
    encoding : str
        Кодировка файла
    """
    with open(filename, mode, encoding=encoding) as f:
        if isinstance(data, dict):
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
        elif isinstance(data, list):
            for item in data:
                f.write(f"{item}\n")
        else:
            f.write(str(data))

    print(f"✓ Данные успешно сохранены в {filename}")


# ==================== ПРИМЕР ИСПОЛЬЗОВАНИЯ ====================

if __name__ == "__main__":
    # Пример 1: Запись метрик классификации (словарь)
    metrics = {
        'Accuracy': 0.92,
        'F1-score': 0.89,
        'ROC-AUC': 0.94,
        'Классы': ['Контроль', 'Эндогенная', 'Экзогенная']
    }
    save_to_txt('classification_metrics.txt', metrics)

    # Пример 2: Запись списка значимых пиков
    peaks = [
        'Пик 1: 1003 cm⁻¹ (фенилаланин)',
        'Пик 2: 1445 cm⁻¹ (CH₂ деформация)',
        'Пик 3: 1655 cm⁻¹ (амид I)'
    ]
    save_to_txt('important_peaks.txt', peaks, mode='a')  # 'a' - добавление в конец

    # Пример 3: Запись простого текста
    save_to_txt('log.txt', 'Обучение модели завершено\n', mode='a')