#Image Index,Finding Labels,Follow-up #,Patient ID,Patient Age,Patient Gender,View Position,OriginalImage[Width,Height],OriginalImagePixelSpacing[x,y],

import pandas as pd
import os

def read_data_entry_csv():
    dataset = pd.read_csv("..{0}res{0}Data_Entry_2017.csv".format(os.path.sep))


