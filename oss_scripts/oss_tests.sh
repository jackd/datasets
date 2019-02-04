#!/bin/bash

set -vx  # print command from file as well as evaluated command

# Instead of exiting on any failure with "set -e", we'll call set_status after
# each command and exit $STATUS at the end.
STATUS=0
function set_status() {
    local last_status=$?
    if [[ $last_status -ne 0 ]]
    then
      echo "<<<<<<FAILED>>>>>> Exit code: $last_status"
    fi
    STATUS=$(($last_status || $STATUS))
}

# Certain datasets/tests don't work with TF2
# Skip them here, and link to a GitHub issue that explains why it doesn't work
# and what the plan is to support it.
TF2_IGNORE_TESTS=""
if [[ "$TF_VERSION" == "tf2"  ]]
then
  # * lsun_test: https://github.com/tensorflow/datasets/issues/34
  TF2_IGNORE_TESTS="
  tensorflow_datasets/image/lsun_test.py
  "
fi
TF2_IGNORE=$(for test in $TF2_IGNORE_TESTS; do echo "--ignore=$test "; done)

# Run Tests
pytest $TF2_IGNORE --ignore="tensorflow_datasets/core/test_utils.py"
set_status

# Test notebooks
NOTEBOOKS="
docs/overview.ipynb
"
for notebook in $NOTEBOOKS
do
  jupyter nbconvert --ExecutePreprocessor.timeout=600 --to notebook --execute $notebook
  set_status
done

exit $STATUS
