import numpy as np
import pandas as pd
import neurokit2 as nk
import matplotlib.pyplot as plt
# import scipy.signal as signal
# import mne  # для работы с ЭЭГ
import librosa  # для работы с аудио/музыкой
import os


import_data_folder = os.path.join(os.getcwd(), "ImportData")


def import_data(fp1_file, fp2_file, ppg_file, music_file):
    """
    Импорт данных: 
    - ЭЭГ с каналов Fp1, Fp2
    - ФПГ
    - музыкальная дорожка в формате mp3
    
    Каждый CSV-файл с биосигналами содержит столбцы: время (сек) и данные.
    """
    # Загрузка биосигналов
    eeg_fp1 = pd.read_csv(fp1_file, names=['time', 'data'], sep=';', decimal=',')
    eeg_fp2 = pd.read_csv(fp2_file, names=['time', 'data'], sep=';', decimal=',')
    ppg = pd.read_csv(ppg_file, names=['time', 'data'], sep=';', decimal=',')
    
    # Вычитание начального времени, чтобы время начиналось с 0
    eeg_fp1['time'] = eeg_fp1['time'] - eeg_fp1['time'].min()
    eeg_fp2['time'] = eeg_fp2['time'] - eeg_fp2['time'].min()
    ppg['time'] = ppg['time'] - ppg['time'].min()
    
    # Вычисление частоты дискретизации биосигналов (усредненное по датасету)
    def calc_sampling_rate(df):
        dt = df['time'].diff().dropna()
        return 1 / dt.mean()
    biosignal_sr = calc_sampling_rate(eeg_fp1)
    
    # Загрузка музыкальной дорожки с сохранением оригинальной частоты дискретизации
    music_data, music_sr = librosa.load(music_file, sr=None)
    
    recording_time = min(eeg_fp1['time'].max(), eeg_fp2['time'].max(), 
                         ppg['time'].max(), len(music_data) / music_sr)
    
    return eeg_fp1, eeg_fp2, ppg, music_data, biosignal_sr, music_sr, recording_time

def crop_data(eeg_fp1, eeg_fp2, ppg, music_data, music_sr, recording_time, start_time=None, end_time=None):
    """
    Обрезает биосигналы и музыкальные данные по заданному временному интервалу.
    Если start_time или end_time равны None, обрезка на этой границе не происходит.
    
    Параметры:
    eeg_fp1, eeg_fp2, ppg - DataFrame с колонками 'time' и 'data'
    music_data - numpy array с аудиосигналом
    music_sr - частота дискретизации аудио (Hz)
    start_time, end_time - временные границы интервала в секундах или None
    
    Возвращает:
    усечённые DataFrame: eeg_fp1_crop, eeg_fp2_crop, ppg_crop
    и отсечённый numpy-массив music_crop
    """

    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = recording_time

    # Обрезка биосигналов
    eeg_fp1_crop = eeg_fp1[(eeg_fp1['time'] >= start_time) & (eeg_fp1['time'] <= end_time)].reset_index(drop=True)
    eeg_fp2_crop = eeg_fp2[(eeg_fp2['time'] >= start_time) & (eeg_fp2['time'] <= end_time)].reset_index(drop=True)
    ppg_crop = ppg[(ppg['time'] >= start_time) & (ppg['time'] <= end_time)].reset_index(drop=True)

    # Расчёт отсчётов для аудио
    start_sample = int(music_sr * start_time)
    end_sample = int(music_sr * end_time)
    music_crop = music_data[start_sample:end_sample]
    

    return eeg_fp1_crop, eeg_fp2_crop, ppg_crop, music_crop

def plot_raw_data(eeg_fp1, eeg_fp2, ppg, music_data, music_sr, recording_time, col, start_time=None, end_time=None):
    """
    Визуализация данных ЭЭГ (Fp1, Fp2), ФПГ и музыкальной дорожки
    в заданном временном диапазоне.
    
    start_time и end_time — временной интервал в секундах.
    """
    if start_time is None:
        start_time = 0
    if end_time is None:
        end_time = recording_time
        
    # Фильтрация данных по временному интервалу ЭЭГ и ФПГ
    eeg_fp1_segment, eeg_fp2_segment, ppg_segment, music_segment = crop_data(eeg_fp1, eeg_fp2, ppg, music_data, music_sr, recording_time, start_time, end_time)
    music_time = np.linspace(start_time, end_time, len(music_segment))
    
    plt.figure(figsize=(12, 10))

    plt.subplot(4, 1, 1)
    plt.plot(eeg_fp1_segment['time'], eeg_fp1_segment[col], label='EEG Fp1')
    plt.title('EEG Fp1')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.plot(eeg_fp2_segment['time'], eeg_fp2_segment[col], label='EEG Fp2', color='orange')
    plt.title('EEG Fp2')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.plot(ppg_segment['time'], ppg_segment[col], label='PPG', color='green')
    plt.title('PPG')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.plot(music_time, music_segment, label='Music', color='red')
    plt.title('Music Signal')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

def fourier_filter(df, column, fs, lowcut, highcut):
    """
    Применяет Фурье фильтрацию к сигналу в DataFrame, убирая частоты ниже lowcut и выше highcut.
    
    Arguments:
        df: pandas DataFrame содержащий сигнал
        column: имя столбца с сигналом
        fs: частота дискретизации (Гц)
        lowcut: нижняя граница пропускания (Гц)
        highcut: верхняя граница пропускания (Гц)
    
    Returns:
        DataFrame с отфильтрованным сигналом
    """
    signal = df[column].values
    N = len(signal)
    fft = np.fft.rfft(signal)
    freqs = np.fft.rfftfreq(N, 1/fs)
    
    # Обнуляем компоненты вне заданного диапазона
    fft[(freqs < lowcut) | (freqs > highcut)] = 0
    
    # Обратное преобразование Фурье
    filtered_signal = np.fft.irfft(fft, N)
    
    df[column + '_filtered'] = filtered_signal

def preprocess_biosignals(raw_biosignals):
    """
    Предобработка биосигналов: удаление артефактов, фильтрация, выделение метрик.
    """
    # Код для предобработки
    pass

def extract_emotional_state(biosignal_metrics):
    """
    Выделение эмоционального состояния на основе метрик биосигналов.
    """
    # Алгоритм классификации эмоций
    pass

def extract_notes(music_file):
    """
    Выделение нот из музыкальной дорожки.
    """
    # Анализ аудиофайла и извлечение нот
    pass

def generate_additional_tracks(notes, emotional_state):
    """
    Создание дополнительных музыкальных дорожек с инструментами
    по эмоциональному состоянию.
    """
    # Код генерации дорожек
    pass

# def merge_and_export_tracks(original_track, additional_tracks):
    """
    Объединение всех дорожек и экспорт итоговой композиции.
    """
    # Сведение и сохранение результата
    pass




# Пример общей структуры запуска этапов:
    
# Импортируем файлы
fp1_file = os.path.join(import_data_folder, "C_EEG.csv")
fp2_file = os.path.join(import_data_folder, "A_EEG.csv")
ppg_file = os.path.join(import_data_folder, "B_PPG.csv")
music_file = os.path.join(import_data_folder, "Экс_3.mp3")

eeg_fp1, eeg_fp2, ppg, music_data, biosignal_sr, music_sr, recording_time = import_data(fp1_file, fp2_file, ppg_file, music_file)

print('recording time = ', int(recording_time), ' [s]')
print('biosignal sampling rate = ', int(biosignal_sr), '[Hz]')
print('music sampling rate = ', int(music_sr), '[Hz]')

# Отрезаем излишки
eeg_fp1, eeg_fp2, ppg, music_data = crop_data(eeg_fp1, eeg_fp2, ppg, music_data, music_sr, recording_time, 10, 130)

# Строим графики во всем временном диапазоне
plot_raw_data(eeg_fp1, eeg_fp2, ppg, music_data, music_sr, recording_time, 'data', 10, 11)

fourier_filter(eeg_fp2, 'data', biosignal_sr, 4, 30)
fourier_filter(eeg_fp1, 'data', biosignal_sr, 4, 30)
fourier_filter(ppg, 'data', biosignal_sr, 0.1, 10)

# Строим графики во всем временном диапазоне
plot_raw_data(eeg_fp1, eeg_fp2, ppg, music_data, music_sr, recording_time, 'data_filtered', 10, 11)



# biosignal_metrics = preprocess_biosignals(raw_biosignals)
# emotional_state = extract_emotional_state(biosignal_metrics)
# notes = extract_notes(music_file)
# additional_tracks = generate_additional_tracks(notes, emotional_state)
# merge_and_export_tracks(music, additional_tracks)