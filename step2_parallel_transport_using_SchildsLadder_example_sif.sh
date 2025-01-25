#!/bin/bash

# Define the Singularity image path
image="/your/path/to/pole_ladder_latest.sif"

# Define the subject variable
subj="OAS30331"

# Run the Singularity container with the necessary bind mount
singularity exec -B "$(pwd)/examples/shared_volume:/usr/src/myapp/volume" "$image" bash -c "
  # Change to the directory where SchildsLadder is located
  cd /usr/src/myapp/Ladder/Ladder/build/ || exit 1

  # Iterate over numbers from 1 to 3
  for i in {1..3}; do
    # Define the input and output file paths
    input_mask=\"/usr/src/myapp/volume/${subj}/I0_MASK.nii.gz\"
    inter_SVF=\"/usr/src/myapp/volume/${subj}/inter_SVF_half_HC.nii.gz\"
    intra_SVF=\"/usr/src/myapp/volume/${subj}/intra_SVF_half_HC\${i}.nii.gz\"
    output_transported_SVF=\"/usr/src/myapp/volume/${subj}/transported_SVF_half_\${i}.nii.gz\"

    # Run SchildsLadder command
    ./SchildsLadder -m \"\$input_mask\" -v \"\$intra_SVF\" -d \"\$inter_SVF\" -t \"\$output_transported_SVF\"
  done
"
