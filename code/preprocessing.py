import pandas as pd
import numpy as np
import cv2

path = '../data/vali_modified.csv'


def preprocess_csv(csv_path):
    df = pd.read_csv(csv_path)
    image_path_array = df['image_path'].as_matrix()
    label_array = df['category'].as_matrix()
    x1 = df['x1'].as_matrix().astype(np.float32)
    y1 = df['y1'].as_matrix().astype(np.float32)
    x2 = df['x2'].as_matrix().astype(np.float32)
    y2 = df['y2'].as_matrix().astype(np.float32)


    for i in range(len(image_path_array)):
        path = image_path_array[i]
        img = cv2.imread(path)
        if img is None:
            continue
        h = img.shape[0]
        w = img.shape[1]

        x1[i] = x1[i] * 1.0 /w
        x2[i] = x2[i] * 1.0/ w
        y1[i] = y1[i] * 1.0/ h
        y2[i] = y2[i] * 1.0/ h

    df['x1_modified'] = pd.DataFrame(x1)
    df['y1_modified'] = pd.DataFrame(y1)
    df['x2_modified'] = pd.DataFrame(x2)
    df['y2_modified'] = pd.DataFrame(y2)

    df.to_csv(csv_path, index=False)
    print(df.head())

if __name__ == '__main__':
    paths = ['../data/vali_modified2.csv',
             '../data/test_modified2.csv',
             '../data/train_modified2.csv']

    for path in paths:
        preprocess_csv(path)
