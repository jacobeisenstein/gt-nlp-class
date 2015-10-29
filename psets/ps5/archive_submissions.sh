#! /bin/bash

dir="data/deppars";
FOREIGN_LANGUAGE=;

# Uncomment the appropriate variable

#FOREIGN_LANGUAGE="german"
#FOREIGN_LANGUAGE="italian"
#FOREIGN_LANGUAGE="spanish"
#FOREIGN_LANGUAGE="french"
#FOREIGN_LANGUAGE="portuguese"

rm -rf submission submission.tgz

if [ -z "$FOREIGN_LANGUAGE" -a "${FOREIGN_LANGUAGE+xxx}" = "xxx" ]; then 
    echo FOREIGN_LANGUAGE is not set at all; 
    exit 1;
fi

filenames=("deliverable1a.conll" "deliverable1b.conll" "deliverable1c.conll" "deliverable1d.conll"
           "$FOREIGN_LANGUAGE.deliverable2a.conll" "$FOREIGN_LANGUAGE.deliverable2b.conll"
           "$FOREIGN_LANGUAGE.deliverable2c.conll" "$FOREIGN_LANGUAGE.deliverable2d.conll"
	   )

archive=()
count=0
while [ "x${filenames[count]}" != "x" ]
do
    filename="$dir/${filenames[count]}"
    if [ ! -f $filename ]; then
	echo "WARN: file $filename not found, not added in archive"
    else
	archive+=($filename)
    fi
    count=$(( $count + 1 ))
done

#Create directory and copy everything
mkdir -p submission/$dir
cp ${archive[*]} submission/$dir
cp -r gtparsing submission
cp -r tests submission
cp score.py submission
cp pset5.ipynb submission

tar -czvf submission.tgz submission/
