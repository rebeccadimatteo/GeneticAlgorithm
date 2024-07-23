import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder
import os

# Assicurarsi che la cartella results esista
os.makedirs('results', exist_ok=True)

# Caricamento del dataset
data = pd.read_csv('Dataset/dataset1.csv')

# Separazione delle etichette dai dati
labels = data['DecileScore']
data = data.drop(columns=['DecileScore'])

# Codifica degli attributi categoriali
label_encoders = {}
for column in data.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    data[column] = le.fit_transform(data[column])
    label_encoders[column] = le

# Codifica delle etichette
labels = LabelEncoder().fit_transform(labels)

# Funzione per applicare FairSMOTE
def apply_fair_smote(data, labels):
    smote = SMOTE(sampling_strategy='auto')
    data_res, labels_res = smote.fit_resample(data, labels)
    return pd.DataFrame(data_res, columns=data.columns), pd.Series(labels_res)

# Funzione per applicare Reweighing
def apply_reweighing(data, labels, protected_attribute):
    data = data.copy()
    data['target'] = labels
    data['weights'] = 1.0

    # Calcolare pesi per ogni combinazione di attributi protetti e target
    for p_value in data[protected_attribute].unique():
        for t_value in data['target'].unique():
            group_mask = (data[protected_attribute] == p_value) & (data['target'] == t_value)
            group_size = group_mask.sum()
            total_size = data.shape[0]
            reweight_factor = total_size / (len(data[protected_attribute].unique()) * group_size)
            data.loc[group_mask, 'weights'] *= reweight_factor

    return data.drop(columns=['target'])

# Funzione per applicare Disparate Impact Remover
def apply_disparate_impact_remover(data, protected_attribute):
    data = data.copy()
    protected_mean = data.groupby(protected_attribute).transform('mean')
    data = (data - protected_mean) + data.mean()
    return data

# Applicare le tecniche di preprocessing
# FairSMOTE
data_smote, labels_smote = apply_fair_smote(data, labels)
data_smote['DecileScore'] = labels_smote
print("FairSMOTE Applied")

# Reweighing
reweighed_data = apply_reweighing(data, labels, 'Sex_Code_Text')
print("Reweighing Applied")

# Disparate Impact Remover
dir_data = apply_disparate_impact_remover(data, 'Sex_Code_Text')
dir_data['DecileScore'] = labels
print("Disparate Impact Remover Applied")

# Salva i risultati preprocessati in file CSV nella cartella results
data_smote.to_csv('results/preprocessed_fair_smote.csv', index=False)
print("FairSMOTE Results Saved")

reweighed_data.to_csv('results/preprocessed_reweighing.csv', index=False)
print("Reweighing Results Saved")

dir_data.to_csv('results/preprocessed_disparate_impact_remover.csv', index=False)
print("Disparate Impact Remover Results Saved")
