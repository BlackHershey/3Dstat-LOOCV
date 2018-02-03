#!/bin/awk -f
# reads CSH file with lines like 
# set subjects = ( 1	1	2	2	3	3	6	... 86 )
# and prints it into a sideways CSV format

BEGIN { 
    FS=" ";
#    RS="\r\n"
}

{
    if (($1 == "set") && ( ($2 =="subjects") || ($2 =="measures") || ($2 =="DV") )) {
#      if ($2 == "subjects") {print "=== "FILENAME" ==="}
      out = $2
#      for(i=5; i<12; i++) {
      for(i=5; i<NF; i++) {
              out=out","$i
      }
      print out
    } 
}

# import pandas as pd
# pd.read_csv('input.csv').T.to_csv('output.csv',header=False)
# Thanks to https://stackoverflow.com/questions/4869189/how-to-transpose-a-dataset-in-a-csv-file
# or, https://pypi.python.org/pypi/transposer/0.0.3

