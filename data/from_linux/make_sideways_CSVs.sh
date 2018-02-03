#!/bin/sh
for file in *"_Text_File_"*.csv2
do
  [transpose_effect_data.py] $file ${file%.csv2}.csv
done
exit

