import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import neurokit2 as nk

class Biosignal():
    def __init__(self, file_names):
        # Загрузка данных из файлов CSV с разделителем ';' и десятичной запятой
        self.eeg_fp1 = pd.read_csv(file_names["Fp1"], names=['time', 'raw'], sep=';', decimal=',')
        self.eeg_fp2 = pd.read_csv(file_names["Fp2"], names=['time', 'raw'], sep=';', decimal=',')
        self.ppg = pd.read_csv(file_names["PPG"], names=['time', 'raw'], sep=';', decimal=',')
        
        # Вычитание минимального значения времени, чтобы время начиналось с 0
        self.eeg_fp1['time'] -= self.eeg_fp1['time'].min()
        self.eeg_fp2['time'] -= self.eeg_fp2['time'].min()
        self.ppg['time'] -= self.ppg['time'].min()
        
        # Вычисление частоты дискретизации (средний интервал между измерениями)
        def calc_sr(df):
            return 1 / df['time'].diff().mean()
        self.eeg_sr = calc_sr(self.eeg_fp1)
        self.ppg_sr = calc_sr(self.ppg)
        
        self.recording_time = min(self.eeg_fp1['time'].max(), self.eeg_fp2['time'].max(), self.ppg['time'].max())
        
        self.ppg_metrics = None
        self.eeg_metrics = None
        self.state_metrics = None
        
            
    def plot(self, column='raw', time_start=0, time_stop=None):  # time_stop вместо duration - хорошо!
        if time_stop is None:
            time_stop = self.recording_time
    
        # Маска временного диапазона для каждого сигнала
        mask_fp1 = (self.eeg_fp1['time'] >= time_start) & (self.eeg_fp1['time'] <= time_stop)
        mask_fp2 = (self.eeg_fp2['time'] >= time_start) & (self.eeg_fp2['time'] <= time_stop)
        mask_ppg = (self.ppg['time'] >= time_start) & (self.ppg['time'] <= time_stop)
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axs[0].plot(self.eeg_fp1['time'][mask_fp1], self.eeg_fp1[column][mask_fp1], label='Fp1 EEG')
        axs[0].set_ylabel('Amplitude (V)')
        axs[0].legend()
        
        axs[1].plot(self.eeg_fp2['time'][mask_fp2], self.eeg_fp2[column][mask_fp2], label='Fp2 EEG')
        axs[1].set_ylabel('Amplitude (V)')
        axs[1].legend()
        
        axs[2].plot(self.ppg['time'][mask_ppg], self.ppg[column][mask_ppg], label='PPG')
        axs[2].set_ylabel('Amplitude')
        axs[2].set_xlabel('Time (seconds)')
        axs[2].legend()
        
        plt.tight_layout()
        plt.show()
        
    def crop(self, time_start=0, time_stop=None):
        if time_stop is None:
            time_stop = self.recording_time
        
        # Обрезка данных по времени
        self.eeg_fp1 = self.eeg_fp1[(self.eeg_fp1['time'] >= time_start) & (self.eeg_fp1['time'] <= time_stop)].copy()
        self.eeg_fp2 = self.eeg_fp2[(self.eeg_fp2['time'] >= time_start) & (self.eeg_fp2['time'] <= time_stop)].copy()
        self.ppg = self.ppg[(self.ppg['time'] >= time_start) & (self.ppg['time'] <= time_stop)].copy()
        
        # Перенормировка времени после обрезки
        self.eeg_fp1['time'] -= self.eeg_fp1['time'].min()
        self.eeg_fp2['time'] -= self.eeg_fp2['time'].min()
        self.ppg['time'] -= self.ppg['time'].min()
        
        # СБРОС ИНДЕКСОВ
        self.eeg_fp1 = self.eeg_fp1.reset_index(drop=True)
        self.eeg_fp2 = self.eeg_fp2.reset_index(drop=True)
        self.ppg = self.ppg.reset_index(drop=True)
        
        self.recording_time = min(self.eeg_fp1['time'].max(), self.eeg_fp2['time'].max(), self.ppg['time'].max())
        
    def filter_eeg(self, fmin=1, fmax=40):
        # Применение полосового фильтра к EEG сигналам → сохраняем в 'filtered'
        self.eeg_fp1['filtered'] = nk.signal_filter(self.eeg_fp1['raw'], sampling_rate=self.eeg_sr, lowcut=fmin, highcut=fmax)
        self.eeg_fp2['filtered'] = nk.signal_filter(self.eeg_fp2['raw'], sampling_rate=self.eeg_sr, lowcut=fmin, highcut=fmax)

    def filter_ppg(self, fmin=0.1, fmax=10):
        # Применение полосового фильтра к PPG сигналу → сохраняем в 'filtered'
        self.ppg['filtered'] = nk.signal_filter(self.ppg['raw'], sampling_rate=self.ppg_sr, lowcut=fmin, highcut=fmax)
        
    def clean_artefact_eeg(self, threshold=100e-6, extend_seconds=0.2):
        """
        Очистка артефактов на ЭЭГ по амплитуде с расширением временных интервалов.

        threshold: амплитудный порог для выделения артефактов (в Вольтах)
        extend_seconds: время (в секундах) расширения интервала артефакта до и после пика
        """
        sampling_rate = self.eeg_sr
        extend_samples = int(extend_seconds * sampling_rate)

        for df in [self.eeg_fp1, self.eeg_fp2]:
            # Найти индексы, где амплитуда превышает порог
            artefact_indices = df.index[df['filtered'].abs() > threshold].to_numpy()

            # Создать булеву маску для всего сигнала
            mask = np.zeros(len(df), dtype=bool)

            # Расширить маску вокруг каждого артефакта
            for idx in artefact_indices:
                start = max(idx - extend_samples, 0)
                end = min(idx + extend_samples, len(df) - 1)
                mask[start:end + 1] = True
            
            self.eeg_fp2['filtered']
            # Пометить загрязненные участки NaN
            df['cleaned'] = df['filtered'].copy()
            df.loc[mask, 'cleaned'] = pd.NA
            
        print("Fp1 очистка: ", 100 * self.eeg_fp1['cleaned'].isna().sum() / len(self.eeg_fp1), " %")
        print("Fp2 очистка: ", 100 * self.eeg_fp2['cleaned'].isna().sum() / len(self.eeg_fp2), " %")

    def clean_artefact_ppg(self):
        """
        Очистка артефактов из PPG с использованием встроенного метода neurokit2.
        """
        # neurokit2 функция для очистки PPG (удаление шумов и артефактов)
        self.ppg['cleaned'] = nk.ppg_clean(self.ppg['filtered'], sampling_rate=self.ppg_sr)
        print("PPG очистка: ", 100 * self.ppg['cleaned'].isna().sum() / len(self.ppg), " %")
        
    def calc_ppg_metrics(self, window_duration=60, step_duration=30):
        """
         Вычисление метрик PPG скользящим окном для оценки эмоционального состояния.
        
         Метрики и их связь с эмоциями:
        
         - PPG_Rate: средняя частота сердечных сокращений (ЧСС).
           Увеличение ЧСС часто связано с повышенным возбуждением (arousal) и стрессом.
        
         - HRV_MeanNN: средний интервал между ударами сердца в миллисекундах.
           Более длинные интервалы связаны с расслаблением и положительной валентностью.
        
         - HRV_SDNN: стандартное отклонение интервалов между ударами — общая вариабельность.
           Низкие значения указывают на стресс и негативные эмоции.
        
         - HRV_RMSSD: квадратичное среднее разности последовательных интервалов.
           Отражает парасимпатическую активность, высокие значения характерны для расслабления и положительных эмоций.
        
         - HRV_LF: мощность низкочастотного диапазона вариабельности ЧСС.
           Связана с общей активностью симпатической и парасимпатической систем.
        
         - HRV_HF: мощность высокочастотного диапазона вариабельности.
           Отражает парасимпатическую активность, связанную с дыхательной синусовой аритмией.
        
         - HRV_LFHF: соотношение LF/HF — индикатор симпато-парасимпатического баланса.
           Повышенные значения часто ассоциируются с высоким возбуждением и стрессом.
        
         Параметры:
         -----------
         window_duration : float
             Длина анализируемого окна в секундах.
         step_duration : float
             Шаг смещения окна в секундах.
        
         Результат:
         -----------
         self.ppg_metrics — DataFrame, строки — середины окон времени, столбцы — метрики.
         """
        # Обработка PPG с neurokit2
        signals, info = nk.ppg_process(self.ppg['cleaned'], sampling_rate=self.ppg_sr)
        all_peak_indices = info["PPG_Peaks"]
        
        # ИТЕРАЦИЯ ПО ВРЕМЕНИ, а не по индексам!
        time_start = 0
        time_end = self.ppg['time'].max()
        
        metrics_data = []
        
        # Скользящее окно по времени (исходная временная шкала)
        while time_start + window_duration <= time_end:
            time_center = time_start + window_duration / 2
            
            # Находим индексы для временного окна
            mask = (self.ppg['time'] >= time_start) & (self.ppg['time'] <= time_start + window_duration)
            
            # Проверяем наличие данных (минимум 50% непропущенных)
            if mask.sum() < 0.5 * self.ppg_sr * window_duration:
                time_start += step_duration
                continue
            
            ppg_segment_indices = self.ppg[mask].index
            window_peaks = all_peak_indices[
                (all_peak_indices >= ppg_segment_indices[0]) & (all_peak_indices <= ppg_segment_indices[-1])
            ]
            
            if len(window_peaks) < 8:
                time_start += step_duration
                continue
            
            try:
                hrv_time = nk.hrv_time(window_peaks, sampling_rate=self.ppg_sr)
                hrv_frequency = nk.hrv_frequency(window_peaks, sampling_rate=self.ppg_sr)
                
                # PPG Rate для временного окна
                ppg_rate_window = signals['PPG_Rate'][mask].mean()
                
                row_data = {
                    'time': time_center,
                    'PPG_Rate': ppg_rate_window,
                    'HRV_MeanNN': hrv_time.get('HRV_MeanNN', pd.NA),
                    'HRV_SDNN': hrv_time.get('HRV_SDNN', pd.NA),
                    'HRV_RMSSD': hrv_time.get('HRV_RMSSD', pd.NA)#,
                    # 'HRV_LF': hrv_frequency.get('HRV_LF', pd.NA),
                    # 'HRV_HF': hrv_frequency.get('HRV_HF', pd.NA),
                    # 'HRV_LFHF': hrv_frequency.get('HRV_LFHF', pd.NA)
                }
                metrics_data.append(row_data)
                
            except Exception:
                pass
            
            time_start += step_duration
        
        self.ppg_metrics = pd.DataFrame(metrics_data).astype(float)

    def plot_ppg_metrics(self):
        """
        Визуализация временных рядов метрик PPG, вычисленных скользящим окном.
    
        Ожидается, что self.ppg_metrics — DataFrame с индексом времени и столбцами:
        'PPG_Rate', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_LF', 'HRV_HF', 'HRV_LFHF'.
        """
        if not hasattr(self, 'ppg_metrics') or self.ppg_metrics.empty:
            print("Метрики не найдены. Выполните calc_ppg_metrics() сначала.")
            return
    
        # metrics = ['PPG_Rate', 'HRV_MeanNN', 'HRV_SDNN', 'HRV_RMSSD', 'HRV_LF', 'HRV_HF', 'HRV_LFHF']
        metrics = self.ppg_metrics.columns
        plt.figure(figsize=(14, 10))
    
        for i, metric in enumerate(metrics, 1):
            # if metric in self.ppg_metrics.columns:
            plt.subplot(len(metrics), 1, i)
            plt.plot(self.ppg_metrics['time'], self.ppg_metrics[metric], marker='o', linestyle='-')
            plt.title(metric)
            plt.xlabel('Time (s)')
            plt.grid(True)
    
        plt.tight_layout()
        plt.show()
        
    def calc_eeg_metrics(self, window_duration=4, step_duration=2):
        """
        Вычисление метрик ЭЭГ скользящим окном для оценки эмоционального состояния.
        
        Метрики и их связь с эмоциями:
        
        РИТМЫ МОЩНОСТИ (Delta, Theta, Alpha, Beta):
        - Fp1_Delta, Fp2_Delta: глубокое расслабление, негативные эмоции (↑)
        - Fp1_Theta, Fp2_Theta: эмоциональная вовлеченность, медитация (↑)
        - Fp1_Alpha, Fp2_Alpha: релаксация, положительный valence (↑Alpha), ↓при arousal
        - Fp1_Beta, Fp2_Beta: возбуждение, стресс, когнитивная нагрузка (↑)
        
        АСИММЕТРИЯ (Fp1 vs Fp2):
        - Alpha_Asymmetry: (Fp1_Alpha - Fp2_Alpha)/(Fp1_Alpha + Fp2_Alpha)
          Отрицательные значения = позитивный valence (Fp2>Fp1)
          
        ВОВЛЕЧЕННОСТЬ
        - engagement: (Fp1_Beta/Fp1_Alpha + Fp2_Beta/Fp2_Alpha) / 2
        
        HJORTH ПАРАМЕТРЫ:
        - Hjorth_Mobility: скорость изменений сигнала (↑ = активность)
        - Hjorth_Complexity: сложность/нерегулярность сигнала (↑ = стресс)
        
        Параметры:
        -----------
        window_duration : float, секунды (рекомендуется 2-8 сек для EEG)
        step_duration : float, секунды
        """
        bands = [(1, 4), (4, 8), (8, 13), (13, 30)]  # Delta, Theta, Alpha, Beta
        band_names = ['Delta', 'Theta', 'Alpha', 'Beta']
        
        # ИТЕРАЦИЯ ПО ВРЕМЕНИ, а не по индексам!
        time_start = 0
        time_end = min(self.eeg_fp1['time'].max(), self.eeg_fp2['time'].max())
        
        metrics_data = []
        
        # Скользящее окно по времени (исходная временная шкала)
        while time_start + window_duration <= time_end:
            time_center = time_start + window_duration / 2
            
            # Находим индексы для временного окна
            fp1_mask = (self.eeg_fp1['time'] >= time_start) & (self.eeg_fp1['time'] <= time_start + window_duration)
            fp2_mask = (self.eeg_fp2['time'] >= time_start) & (self.eeg_fp2['time'] <= time_start + window_duration)
            
            fp1_segment = self.eeg_fp1.loc[fp1_mask, 'cleaned'].to_numpy()
            fp2_segment = self.eeg_fp2.loc[fp2_mask, 'cleaned'].to_numpy()
            
            # Проверяем наличие данных (минимум 50% непропущенных)
            if len(fp1_segment) < 0.5 * self.eeg_sr * window_duration or len(fp2_segment) < 0.5 * self.eeg_sr * window_duration:
                time_start += step_duration
                continue
            
            # Заменяем NaN на 0 только для анализа
            fp1_segment = np.nan_to_num(fp1_segment)
            fp2_segment = np.nan_to_num(fp2_segment)
            
            try:
                # Мощности в диапазонах
                fp1_power = nk.signal_power(fp1_segment, frequency_band=bands, sampling_rate=self.eeg_sr)
                fp2_power = nk.signal_power(fp2_segment, frequency_band=bands, sampling_rate=self.eeg_sr)
                
                # Именуем колонки
                fp1_power.columns = [f'Fp1_{name}' for name in band_names]
                fp2_power.columns = [f'Fp2_{name}' for name in band_names]
                
                def asymmetry(c1, c2):
                    return (c1.iloc[0] - c2.iloc[0]) / (c1.iloc[0] + c2.iloc[0] + 1e-10)
                
                # Асимметрия
                asymmetries = {f'{name}_Asymmetry': asymmetry(fp1_power[f'Fp1_{name}'], fp2_power[f'Fp2_{name}']) 
                              for name in band_names}
                
                # ВОВЛЕЧЕННОСТЬ
                engagement =  (fp1_power['Fp1_Beta']/fp1_power['Fp1_Alpha'] + 
                               fp2_power['Fp2_Beta']/fp2_power['Fp2_Alpha']) / 2
                
                # Hjorth параметры
                def hjorth_params(signal):
                    signal = signal[~np.isnan(signal)]
                    if len(signal) < 10:
                        return pd.NA, pd.NA
                    activity = np.var(signal)
                    diff1 = np.diff(signal)
                    mobility = np.std(diff1) / np.sqrt(activity) if activity > 0 else pd.NA
                    diff2 = np.diff(diff1)
                    complexity = (np.std(diff2) / np.std(diff1)) / mobility if mobility > 0 else pd.NA
                    return mobility, complexity
                
                fp1_mobility, fp1_complexity = hjorth_params(fp1_segment)
                fp2_mobility, fp2_complexity = hjorth_params(fp2_segment)
                
                row = {
                    'time': time_center,
                    **{k: v.iloc[0] for k, v in fp1_power.items()},
                    **{k: v.iloc[0] for k, v in fp2_power.items()},
                    **asymmetries,
                    'Engagement': engagement,
                    'Fp1_Hjorth_Mobility': fp1_mobility,
                    'Fp1_Hjorth_Complexity': fp1_complexity,
                    'Fp2_Hjorth_Mobility': fp2_mobility,
                    'Fp2_Hjorth_Complexity': fp2_complexity,
                }
                metrics_data.append(row)
                
            except Exception:
                pass
            
            time_start += step_duration
        
        self.eeg_metrics = pd.DataFrame(metrics_data)
        
    def plot_eeg_metrics(self):
        """
        Визуализация метрик ЭЭГ.
        
        Figure 1: Мощность ритмов для каждого канала (Fp1 и Fp2) отдельно
        Figure 2: Асимметрия ритмов + Hjorth параметры
        """
        if not hasattr(self, 'eeg_metrics') or self.eeg_metrics.empty:
            print("Метрики EEG не найдены. Выполните calc_eeg_metrics() сначала.")
            return
        
        # Figure 1: Мощность ритмов по каналам
        fig1, axes1 = plt.subplots(2, 1, figsize=(14, 10))
        
        # Ритмы для отображения
        rhythms = ['Delta', 'Theta', 'Alpha', 'Beta']
        
        # График Fp1
        ax1 = axes1[0]
        for rhythm in rhythms:
            ax1.plot(self.eeg_metrics['time'], self.eeg_metrics[f'Fp1_{rhythm}'], 
                    label=rhythm, linewidth=2)
        ax1.set_title('EEG Fp1 - Мощность ритмов', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Мощность')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # График Fp2  
        ax2 = axes1[1]
        for rhythm in rhythms:
            ax2.plot(self.eeg_metrics['time'], self.eeg_metrics[f'Fp2_{rhythm}'], 
                    label=rhythm, linewidth=2)
        ax2.set_title('EEG Fp2 - Мощность ритмов', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Время (с)')
        ax2.set_ylabel('Мощность')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Figure 2: Асимметрия + Hjorth
        fig2, axes2 = plt.subplots(4, 1, figsize=(14, 12))
        
        # Асимметрия ритмов
        ax3 = axes2[0]
        for rhythm in rhythms:
            ax3.plot(self.eeg_metrics['time'], self.eeg_metrics[f'{rhythm}_Asymmetry'], 
                    label=rhythm, linewidth=2)
        ax3.set_title('Асимметрия ритмов (Fp1 - Fp2)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Асимметрия')
        ax3.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        
        # Hjorth параметры Fp1
        ax4 = axes2[1]
        ax4.plot(self.eeg_metrics['time'], self.eeg_metrics['Fp1_Hjorth_Mobility'], 
                label='Fp1 Mobility', linewidth=2, color='blue')
        ax4.plot(self.eeg_metrics['time'], self.eeg_metrics['Fp1_Hjorth_Complexity'], 
                label='Fp1 Complexity', linewidth=2, color='red')
        ax4.set_title('Hjorth параметры Fp1', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Значение')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Hjorth параметры Fp2
        ax5 = axes2[2]
        ax5.plot(self.eeg_metrics['time'], self.eeg_metrics['Fp2_Hjorth_Mobility'], 
                label='Fp2 Mobility', linewidth=2, color='blue')
        ax5.plot(self.eeg_metrics['time'], self.eeg_metrics['Fp2_Hjorth_Complexity'], 
                label='Fp2 Complexity', linewidth=2, color='red')
        ax5.set_title('Hjorth параметры Fp2', fontsize=14, fontweight='bold')
        ax5.set_xlabel('Время (с)')
        ax5.set_ylabel('Значение')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Вовлеченность
        ax6 = axes2[3]
        ax6.plot(self.eeg_metrics['time'], self.eeg_metrics['Engagement'], 
                label='Engagement', linewidth=2, color='blue')
        ax6.set_title('Вовлеченность', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Время (с)')
        ax6.set_ylabel('Значение')
        ax6.legend()
        ax6.grid(True, alpha=0.3)        
        
        plt.tight_layout()
        plt.show()
        
    def extract_numeric(self, df):
        numeric_df = pd.DataFrame()
        for col in df.columns:
            if isinstance(df[col].iloc[0], pd.Series):
                # Извлекаем первый (и единственный) элемент Series
                numeric_df[col] = df[col].apply(lambda x: x.iloc[0] if isinstance(x, pd.Series) else x)
            else:
                numeric_df[col] = df[col]
        return numeric_df
    
    def calc_state(self, time_step, model):
        
        """Интерполяция метрик на временную ось от 0 до recording_time"""
        interp_times = np.arange(0, self.recording_time + time_step/2, time_step)
        
        interp = self.ppg_metrics.set_index('time')
        interp = interp.reindex(interp.index.union(interp_times))
        interp = interp.interpolate(method='index').ffill().bfill()
        ppg_interp = interp.loc[interp_times].reset_index()
        
        interp = self.eeg_metrics.set_index('time')
        interp = interp.reindex(interp.index.union(interp_times))
        interp = interp.interpolate(method='index').ffill().bfill()
        eeg_interp = interp.loc[interp_times].reset_index()
        
        # Общая таблица
        metrics_df = pd.merge(ppg_interp, eeg_interp, on='time', how='outer')
        metrics_df = metrics_df.sort_values('time').reset_index(drop=True)
        
        def compute_scale(scale_cfg, row):
            total = scale_cfg.get('bias', 0.0)
            for metric_name, params in scale_cfg['metrics'].items():
                mean = params['mean']
                std = params['std']
                weight = params['weight']
                value = row[metric_name]
                z = (value - mean) / std
                total += weight * z
            return total
        
        valence_values = []
        arousal_values = []
        
        for _, row in metrics_df.iterrows():
            valence_values.append(compute_scale(model['valence'], row))
            arousal_values.append(compute_scale(model['arousal'], row))
        
        self.state_metrics = pd.DataFrame({
            'time': metrics_df['time'],
            'valence': valence_values,
            'arousal': arousal_values
        }).astype(float)

    def plot_state(self, time_start=0, time_stop=None):
        if time_stop is None:
            time_stop = self.recording_time
    
        df = self.state_metrics
        mask = (df['time'] >= time_start) & (df['time'] <= time_stop)
        df = df.loc[mask]
    
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
        # 1) Временные ряды возбуждения и валентности
        axes[0].plot(df['time'], df['arousal'], label='Arousal', linewidth=2, color='tab:orange')
        axes[0].plot(df['time'], df['valence'], label='Valence', linewidth=2, color='tab:blue')
        axes[0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[0].set_ylabel('Score (a.u.)')
        axes[0].set_title('Time Courses of Arousal and Valence')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
    
        # 2) Точки на плоскости (валентность по X, возбуждение по Y)
        axes[1].scatter(df['valence'], df['arousal'], c=df['time'], cmap='viridis', s=20)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].axvline(x=0, color='black', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('Valence')
        axes[1].set_ylabel('Arousal')
        axes[1].set_title('Arousal–Valence Plane (color = time)')
        axes[1].grid(True, alpha=0.3)
    
        plt.tight_layout()
        plt.show()  
        