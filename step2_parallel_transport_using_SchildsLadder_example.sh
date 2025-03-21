# !/bin/bash

# Change to the directory where SchildsLadder is located
# cd /usr/src/myapp/Ladder/Ladder/build/ || exit 1

# Iterate over numbers from 1 to 3
subj="OAS30331"
for i in {1..3}; do
# Define the input and output file paths with {x} replaced by the current value of i
input_mask="/usr/src/myapp/volume/${subj}/I0_MASK.nii.gz"
inter_SVF="/usr/src/myapp/volume/${subj}/inter_SVF_half.nii.gz"
intra_SVF="/usr/src/myapp/volume/${subj}/intra_SVF_half_${i}.nii.gz"
output_transported_SVF="/usr/src/myapp/volume/${subj}/transported_SVF_half_${i}.nii.gz"

# Run the command with the current values of input and output file paths
./SchildsLadder -m "$input_mask" -v "$intra_SVF" -d "$inter_SVF" -t "$output_transported_SVF"
done
