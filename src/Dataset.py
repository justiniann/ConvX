#Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],

import pandas as pd
import os

def read_data_entry_csv():
    return pd.read_csv("..{0}res{0}Data_Entry_2017.csv".format(os.path.sep))

dataset = read_data_entry_csv()

value_counts = dataset['Finding Labels'].value_counts()

num_images = len(dataset['Finding Labels'])
num_healthy_images = value_counts['No Finding']

print("Total number of images: {}".format(num_images))
print("Total number of healthy images: {}".format(num_healthy_images))
print("Total number of unhealthy images: {}".format(num_images - num_healthy_images))
