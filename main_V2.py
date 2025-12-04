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
biosignal.plot_ppg_metrics()

#%%
biosignal.calc_eeg_metrics(window_duration=10, step_duration=5)
biosignal.plot_eeg_metrics()

#%%
biosignal.calc_state(time_step=5)
# biosignal.plot_state()

#%%
file_name_sound = os.path.join(import_data_folder, "Экс_3.mp3")
sound = Sound(file_name_sound)


