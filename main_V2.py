import os
from biosignal import Biosignal
from sound import Sound

import_data_folder = os.path.join(os.getcwd(), "ImportData")

file_names_biosignal = {
    "Fp1": os.path.join(import_data_folder, "C_EEG.csv"),
    "Fp2": os.path.join(import_data_folder, "A_EEG.csv"),
    "PPG": os.path.join(import_data_folder, "B_PPG.csv")
    }

biosignal = Biosignal(file_names_biosignal)
biosignal.plot(column='raw')

#%%
biosignal.crop(time_start=10, time_stop=None)
biosignal.plot(column='raw')

#%%
biosignal.filter_eeg(fmin=1, fmax=40)
biosignal.filter_ppg(fmin=0.1, fmax=10)
biosignal.plot(column='raw', time_stop=20)
biosignal.plot(column='filtered', time_stop=20)

#%%
biosignal.clean_artefact_eeg(threshold=0.2, extend_seconds=0.05)
biosignal.clean_artefact_ppg()
biosignal.plot(column='filtered')#, time_start=50, time_stop=60)
biosignal.plot(column='cleaned')#, time_start=50, time_stop=60)

#%%
biosignal.calc_ppg_metrics(window_duration=30, step_duration=20)
print(biosignal.ppg_metrics.columns)
biosignal.plot_ppg_metrics()

#%%
biosignal.calc_eeg_metrics(window_duration=10, step_duration=5)
biosignal.plot_eeg_metrics()

#%%
model = {
    'valence': {
        'metrics': {
            # Индекс фронтальной альфа-асимметрии (Fp1Alpha-Fp2Alpha)/(Fp1Alpha+Fp2Alpha)
            # Среднее 0, считаем, что уже приведён к приблизительно нормальному распределению
            'Alpha_Asymmetry': {'mean': 0.0, 'std': 1.0, 'weight': 0.7},

            # Более высокая RMSSD обычно ассоциирована с лучшей регуляцией и благополучием
            # Типичный диапазон для здоровых взрослых ~40–60 мс, возьмём среднее 50, σ ~15
            'HRV_RMSSD': {'mean': 40.0, 'std': 15.0, 'weight': 0.3}
        },
        'bias': 0.0
    },

    'arousal': {
        'metrics': {
            # Пульс: спокойное состояние ~70–75 уд/мин, σ ~10
            # Более высокий пульс → большее возбуждение
            'PPG_Rate': {'mean': 90.0, 'std': 10.0, 'weight': 0.3},

            # Вовлеченность: больше беты, меньше альфы → выше кортикальное возбуждение/вовлечённость
            'Engagement': {'mean': 0.5, 'std': 0.2, 'weight': 0.7}
        },
        'bias': 0.0
    }
}

biosignal.calc_state(time_step=5, model = model)
biosignal.plot_state()

#%%
file_name_sound = os.path.join(import_data_folder, "Экс_3.mp3")
sound = Sound(file_name_sound)


