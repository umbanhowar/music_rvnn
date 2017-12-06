#!/bin/bash
#
#
# Make sure it runs from current directory.
#$ -cwd
#
#

/local/bin/matlab -nodisplay -nojvm -nosplash -nodesktop -r \
      "try, generateCosDistMat_chroma('$SGE_TASK_ID'), catch e, fprintf(e.message), exit(1), end, exit(0);"
echo "matlab exit code: $?"
