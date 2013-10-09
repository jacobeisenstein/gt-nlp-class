import sys

with open(sys.argv[0]) as keyfile:
    with open(sys.argv[1]) as resfile:
        for keyline in keyfile:
            resline = resfile.readline()
            print keyline,resline
