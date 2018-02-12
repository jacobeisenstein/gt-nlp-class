import tarfile
import glob
names = []
with open('manifest.txt', 'r') as files:
    for name in files:
        names.extend(glob.glob(name.strip()))

tar = tarfile.open("pset2-submission.tgz", "w:gz")
for name in names:
    tar.add(name)
    print(name)
tar.close()
