#!/bin/awk -f
# reads Windows-format CSV file and prints one column

BEGIN { 
    FS=","
#    RS="\r\n"
}

{
    print $1
}
