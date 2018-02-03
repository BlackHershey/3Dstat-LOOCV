# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os
import glob
import pandas as pd

for filename in glob.glob('*.csv2'):
    fileroot, file_ext = os.path.splitext(filename)
    pd.read_csv(filename,header=None,index_col=0).T.to_csv(fileroot+".csv",index=False)

# Thanks to https://stackoverflow.com/questions/4869189/how-to-transpose-a-dataset-in-a-csv-file

# FOR ONE FILE:
#import argparse
#parser = argparse.ArgumentParser(description="flip a text effect file")
#parser.add_argument("input_file", 
#    help='"sideways" text effect file e.g. Valence_Text_File_6-14_AG.csv2')
#parser.add_argument("output_file", 
#    help="proper CSV effect file e.g. Valence_Text_File_6-14_AG.csv")
#args = parser.parse_args()
#pd.read_csv(args.input_file,header=None,index_col=0).T.to_csv(args.output_file,index=False)
