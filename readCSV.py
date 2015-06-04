import csv
import os
import shutil

fn = os.path.join(os.path.dirname(__file__), 'forms.csv')
with open(fn) as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    writers = []
    rows = []
    for row in spamreader:
        #print row
        rows.append(row)
        writers.append(row[1])
        #print writers
    countVar=1
    writers_filtered = []
    for i in range(1,1000):
        if writers.count(str(i)) >= 10 and countVar<11:
            writers_filtered.append(i)
            #print " for i : %d, count is %d " % (i, writers.count(str(i)))
            countVar = countVar+1
    print (writers_filtered)

    for row in rows:
        for i in range(0, len(writers_filtered)):
            if row[1]==str(writers_filtered[i]):
                dst = os.path.join(os.path.dirname(__file__), 'Dataset/'+row[1])
                if not (os.path.isdir(dst)):
                    os.makedirs(dst)
                src = os.path.join(os.path.dirname(__file__), 'forms/'+row[0]+'.png')
                print (src)
                shutil.copy(src, dst)