import numpy as np
import pandas as pd
import os

for i in range(9,10):
    directory = f'/Users/stabatabaeim/Library/CloudStorage/OneDrive-TheUniversityofMelbourne/University/Year 2/Experiment/EyeTracker/Data/Participant {i}'
    print(directory)
    folders_2025 = [name for name in os.listdir(directory)
    if os.path.isdir(os.path.join(directory, name)) and name.startswith('2025')]
    directory = directory + "/" + folders_2025[0]
    print(directory)

    # load raw timestamps
    times = np.fromfile(directory + '/Neon Scene Camera v1 ps1.time', dtype="<u8")

    # make a DataFrame and save
    df = pd.DataFrame({'timestamp_ns': times})
    df.to_csv(directory + '/Video_Timestamp.csv', index=False)




# # convert to human-readable datetimes
# dt_index = pd.to_datetime(times, unit="ns")
# dt_aux   = pd.to_datetime(times_aux, unit="ns")

# # just take the first 5 entries
# print(dt_index[:5])
# print(dt_aux[:5])