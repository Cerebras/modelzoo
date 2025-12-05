#!/usr/bin/bash
BRANCH=$1;
OUTPUT_DIR=$2;

echo Exporting branch $BRANCH to directory $OUTPUT_DIR;

git archive --format tar $BRANCH | tar -x -C $OUTPUT_DIR;

GITTOP=$(git rev-parse --show-toplevel)
cp $GITTOP/.gitignore $OUTPUT_DIR/modelzoo/