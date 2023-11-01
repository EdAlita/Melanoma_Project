import numpy as np
import pandas as pd

if __name__ == "__main__":

    data = pd.read_csv('./features/all/features_train_HSV_GLCM_shape_gloh.csv')

    category_mapping = {'nevus': 1, 'others': 0}
    labels = np.array(['nev','ack', 'bcc', 'bkl', 'def', 'mel', 'scc', 'vac'])

    y =  pd.Series([labels[np.where(fname[:3] == labels)][0] for fname in data['fname']])
    data['label'] = y
    data.to_csv('./features/all/features_train_HSV_GLCM_shape_gloh_MC.csv', index=False)

