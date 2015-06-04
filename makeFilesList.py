import os

dn = os.path.join(os.path.dirname(__file__), 'Dataset')
for dirname, dirnames, filenames in os.walk(dn):
    # print path to all subdirectories first.
    for subdirname in dirnames:
        print(os.path.join(dirname, subdirname))

    # print path to all filenames.
    for filename in filenames:
        fn = os.path.join(os.path.dirname(__file__), 'list.txt')
        with open(fn, "a") as myfile:
            myfile.write(subdirname + '/' + filename + '\n')