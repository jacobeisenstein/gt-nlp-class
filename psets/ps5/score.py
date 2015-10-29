import sys

def is_same (first, second):
    return first == second

def accuracy (keyfile, resfile):
    num_correct = 0
    i = 0
    with open (keyfile) as kfile, open (resfile) as rfile:
        for keyline in kfile:
            resline = rfile.readline ()
            keyparts = keyline.split ()
            resparts = resline.split ()
            if len (keyparts) > 1 and len (resparts) > 1:
                num_correct += int (is_same(keyparts[6].rstrip(),resparts[6].rstrip()))
                i += 1
    print num_correct/float(i)
    return num_correct/float(i)

if __name__ == "__main__":
    keyfile = sys.argv[1]
    resfile = sys.argv[2]
    accuracy (keyfile, resfile)
