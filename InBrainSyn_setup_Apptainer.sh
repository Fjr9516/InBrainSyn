#!/bin/sh

image="/apps/containers/TensorFlow/TensorFlow-2.5.0-NGC-21.08.sif"
home="/path/to/your/home/or/project/directory"

d="$home/pip/$(basename "$image" .sif)"
export APPTAINERENV_PYTHONUSERBASE="$d"
export APPTAINERENV_PREPEND_PATH="$d/bin:$APPTAINERENV_PREPEND_PATH"
export APPTAINERENV_PIP_CACHE_DIR=$(mktemp -dp /path/to/temporary/directory)
trap "rm -rf -- '$APPTAINERENV_PIP_CACHE_DIR'" EXIT

d="$PYTHONPATH"
#d="$d:$home/git/voxelmorph"
export APPTAINERENV_PYTHONPATH="$d"

apptainer exec --nv -e "$image" "$@"
