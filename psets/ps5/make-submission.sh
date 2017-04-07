# To run, type the command:
# sh make-submission.sh

SUBMIT_FILES="gtnlplib/ tests/ pset5.ipynb text-answers.md predictions/"
TARBALL_NAME="problemset5.tar.gz"
tar zvcf ${TARBALL_NAME} ${SUBMIT_FILES}
echo "===---------------------------------------------------------------==="
echo "READ THIS"
echo "===---------------------------------------------------------------==="
echo "Submit ${TARBALL_NAME} on T-Square.  DO NOT MODIFY THE FILE NAME"
echo "CHECK BEFORE SUBMITTING THAT IT CONTAINS EVERYTHING YOU NEED"
echo "Run the command sequence:"
echo "$ mkdir temp"
echo "$ cd temp"
echo "$ cp ../${TARBALL_NAME} ."
echo "$ tar xvzf ${TARBALL_NAME}"
echo "And check that your code, ipython notebook, text-answers.md, predictions files are unzipped"
echo "////////////////////////////////////////////////////////////////////////"
