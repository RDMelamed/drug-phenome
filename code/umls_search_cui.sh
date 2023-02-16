#!/bin/bash
# requires conda activate py3
while read line; do
    echo $line
    python uts-rest-api/samples/python/search-terms.py -s "$line" > x;
    gots=`grep -F "ui:" x | cut -f 2 -d ":" | tr "\n" "," | sed -r 's/\s+//g'`
    #gots=`grep -m 1 -F "ui:" x | cut -d ":"  -f 2`

    echo $line%%%$gots >> res$1
done < $1
