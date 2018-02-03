#!/bin/awk -f

BEGIN {
	FS="\t"; 
#	OFS=","
} 
{
	out = $1
	for (i=2;i<=NF;i++) {
		out=out","$i
	}
	print out
}
