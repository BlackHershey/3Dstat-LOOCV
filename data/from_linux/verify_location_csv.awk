#!/bin/awk -f

BEGIN {
	FS=","; 
	OFS=","
} 
{
	if (($1 != $6) || ($2 != $7)) {
		print "ERROR: line "$1,$2" mismatch: "$6,$7
	}
	print $1,$2,$6,$7  # for visual double-check
}
