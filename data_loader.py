import os
import numpy as np
import torch
from torch.utils.data import Dataset
from concurrent.futures import ThreadPoolExecutor

CLASS_MAP = {"control": 0, "endo": 1, "exo": 2}


def _load_and_aggregate(fp):
    """
    Загружает файл и возвращает строго 1 вектор из 4 чисел (float32).
    """
    try:
        arr = np.loadtxt(fp)
    except Exception:
        return None
    
    if arr is None or arr.size == 0:
        return None
    
    arr = np.atleast_2d(arr)
    features = None
    
    # 1. Много строк, 4+ столбца → mean по строкам
    if arr.shape[1] >= 4:
        features = arr[:, :4].mean(axis=0)
    # 2. 1 столбец, много строк → первые 4 строки
    elif arr.shape[1] == 1 and arr.shape[0] >= 4:
        features = arr[:4, 0]
    # 3. 4+ строки, много столбцов → mean по первым 4 строкам
    elif arr.shape[0] >= 4:
        features = arr[:4, :4].mean(axis=0)
    
    # ФИЛЬТР: Если не получилось собрать 4 числа — выходим
    if features is None or features.size != 4:
        return None
    
    # Принудительная конвертация в плоский массив float32
    features = np.array(features, dtype=np.float32).flatten()
    
    if features.shape != (4,):
        return None
        
    if not np.all(np.isfinite(features)):
        return None
    
    return features


class TxtDataset(Dataset):
    def __init__(self, root_dir, classes_map=None, max_workers=None):
        self.root_dir = root_dir
        self.classes_map = classes_map or CLASS_MAP
        self.samples = []  # (features, label, file_id)
        self.file_records = []
        self.max_workers = max_workers or (os.cpu_count() or 1)
        self._load_all_files()

    def _gather_file_list(self):
        files = []
        # Проверка существования корневой папки
        if not os.path.exists(self.root_dir):
            return files
            
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            label = self.classes_map.get(class_name)
            if label is None:
                continue
            for dirpath, _, filenames in os.walk(class_dir):
                for fn in filenames:
                    if not fn.lower().endswith(".txt"):
                        continue
                    fp = os.path.join(dirpath, fn)
                    files.append((fp, label))
        return files

    def _load_all_files(self):
        file_list = self._gather_file_list()
        if not file_list:
            print(f"⚠️ Не найдено файлов в {self.root_dir}")
            return
        
        fps = [fp for fp, _ in file_list]
        labels = [lbl for _, lbl in file_list]
        
        print(f"📁 Загрузка {len(fps)} файлов...")
        
        success_count = 0
        error_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            results = list(ex.map(_load_and_aggregate, fps))
        
        for file_id, (fp, lbl, features) in enumerate(zip(fps, labels, results)):
            if features is None:
                error_count += 1
                continue
            
            # ЖЁСТКАЯ ПРОВЕРКА: тип и форма
            if isinstance(features, np.ndarray) and features.shape == (4,):
                self.file_records.append({"path": fp, "label": lbl, "id": file_id})
                self.samples.append((features, lbl, file_id))
                success_count += 1
            else:
                error_count += 1
                # Отладка: что именно пришло
                print(f"⚠️ Файл {fp}: неверная форма {type(features)} {getattr(features, 'shape', 'no shape')}")
        
        print(f"✅ Успешно загружено: {success_count} файлов")
        if error_count > 0:
            print(f"⚠️ Пропущено файлов: {error_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, _ = self.samples[idx]
        return torch.from_numpy(x), int(y)

    def get_all(self, return_files=False):
        if not self.samples:
            print("❌ Нет данных в self.samples!")
            return (
                np.empty((0, 4), dtype=np.float32), 
                np.array([], dtype=np.int64), 
                np.array([], dtype=np.int32)
            )
        
        # Извлекаем данные
        xs = [s[0] for s in self.samples]
        ys = [s[1] for s in self.samples]
        files = [s[2] for s in self.samples]
        
        # Отладка: проверяем типы перед vstack
        for i, x in enumerate(xs[:5]):
            if not isinstance(x, np.ndarray):
                print(f"⚠️ xs[{i}] это {type(x)}, конвертируем...")
                xs[i] = np.array(x, dtype=np.float32).flatten()
            if x.shape != (4,):
                print(f"⚠️ xs[{i}] имеет форму {x.shape}, пытаемся исправить...")
                xs[i] = np.array(x, dtype=np.float32).flatten()[:4]
                if xs[i].shape != (4,):
                    xs[i] = np.pad(xs[i], (0, 4-len(xs[i]))) # Дополняем нулями если мало
        
        # Финальная проверка
        shapes = [x.shape for x in xs]
        if len(set(shapes)) > 1:
            print(f"❌ Разные формы массивов: {set(shapes)}")
            return (
                np.empty((0, 4), dtype=np.float32), 
                np.array([], dtype=np.int64), 
                np.array([], dtype=np.int32)
            )
        
        try:
            X = np.vstack(xs)
            y = np.array(ys, dtype=np.int64)
            if return_files:
                return X, y, np.array(files, dtype=np.int32)
            return X, y
        except Exception as e:
            print(f"❌ Ошибка vstack: {e}")
            print(f"Первые 3 элемента xs: {xs[:3]}")
            return (
                np.empty((0, 4), dtype=np.float32), 
                np.array([], dtype=np.int64), 
                np.array([], dtype=np.int32)

            )
