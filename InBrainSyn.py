#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import voxelmorph as vxm
from voxelmorph.tf.layers import SpatialTransformer, VecInt, RescaleTransform

import src.visualize_tools as vt
from src.pipeline_AtlasGAN_registration import registrator, template_generator

csv_path = './src/oasis3_metadata.csv'

# two models:
HC_model_path = './models/HC_only/gploss_1e_4_dataset_OASIS3_single_cohort_eps300_Gconfig_ours_normreg_True_lrg0.0001_lrd0.0003_cond_True_regloss_NCC_lbdgan_0.1_lbdreg_1.0_lbdtv_0.0_lbdgp_0.0001_dsnout_False_start_0_clip_True/'
AD_model_path = './models/AD_only/gploss_1e_4_dataset_OASIS3_single_cohort_eps300_Gconfig_ours_normreg_True_lrg0.0001_lrd0.0003_cond_True_regloss_NCC_lbdgan_0.1_lbdreg_1.0_lbdtv_0.0_lbdgp_0.0001_dsnout_False_start_0_clip_True/'

# test subject
subj = 'OAS30331'
avg_img_path = './src/'
data_path = f'./examples/shared_volume/{subj}/'

# functions
def read_csv_subj(subj):
    df = pd.read_csv(csv_path)
    ages = df["subject_ages"][df["subject_ids"] == subj]
    cdrs = df["scan_cdr"][df["subject_ids"] == subj]
    return list(ages), list(cdrs)

def step1_extract_SVF(is_half = False, single_cohort = 'HC', run_Baseline = False, run_InBrainSyn = True): # single_cohort = ''/'HC'/'AD'
    print("Step 1: Extracting SVF and setting up parallel transport...")
    
    # show img and age
    ages = []
    imgs = []
    for npz_file in [i for i in sorted(os.listdir(data_path)) if i.endswith('.npz')]:
        print(f'==== {npz_file} ====')
        data = np.load(f'{data_path}{npz_file}', allow_pickle=True)
        img = data['vol']
        age = data['age']
        disease_condition = data['disease_condition']
        seg = data['synth_seg']

        imgs.append(img)
        ages.append(age)

    # load cdr
    _, cdrs = read_csv_subj(subj)

    if single_cohort == 'Inter':
        model_lst = [HC_model_path if cdr == 0 else AD_model_path for cdr in cdrs]
    elif single_cohort == 'HC':
        model_lst = [HC_model_path for i in range(len(ages))]
    else:
        model_lst = [AD_model_path for i in range(len(ages))]

    # extract templates given ages
    templates = []
    for input_age, model in zip(ages, model_lst):
        templates.append(template_generator(avg_img_path,  input_age, model))

    # extract intra-template SVFs
    for i in range(1, len(templates)):
        if run_InBrainSyn:
            moved_atlas, diff_field, vel_field, vel_half = registrator(templates[0], templates[i], HC_model_path,
                                                                   avg_img_path, n_condns=1, output_vel=True)

            if is_half==True:
                vel_field = vel_half
                save_name = 'half_'
            else:
                save_name = ''
            # save
            vel_field = vel_field.numpy().squeeze()
            vxm.py.utils.save_volfile(vel_field, f'{data_path}intra_SVF_{save_name}{single_cohort}{i}.nii.gz')
            vt.correct_vox2ras_matrix(f'{data_path}intra_SVF_{save_name}{single_cohort}{i}.nii.gz', reference_nifiti='./src/align_norm.nii.gz')

        if run_Baseline:
            # run naive baseline method
            moved_img, _, _, _ = registrator(imgs[0], templates[i], HC_model_path,
                                                                       avg_img_path, n_condns=1, output_vel=True)
            # save baseline images
            moved_img = moved_img.numpy().squeeze()
            vxm.py.utils.save_volfile(moved_img, f'{data_path}Baseline_{i}.nii.gz')
            vt.correct_vox2ras_matrix(f'{data_path}Baseline_{i}.nii.gz',
                                      reference_nifiti='./src/align_norm.nii.gz')

    # extract subject-template SVF
    moved_atlas, diff_field, vel_field, vel_half = registrator(templates[0], imgs[0], HC_model_path,
                                                           avg_img_path, n_condns=1, output_vel=True)
    if is_half == True:
        vel_field = vel_half
        save_name = 'half_'
    else:
        save_name = ''
    # save
    vel_field = vel_field.numpy().squeeze()
    vxm.py.utils.save_volfile(vel_field, f'{data_path}inter_SVF_{save_name}{single_cohort}.nii.gz')
    vt.correct_vox2ras_matrix(f'{data_path}inter_SVF_{save_name}{single_cohort}.nii.gz', reference_nifiti='./src/align_norm.nii.gz')

    # create subject map
    I0_data = f"{[i for i in sorted(os.listdir(data_path)) if i.endswith('.npz')][0]}"
    print(f'==== {I0_data} ====')
    data = np.load(f'{data_path}{I0_data}')
    seg = data['synth_seg']
    age = data['age']
    seg_mask = seg > 0
    sub_MASK = np.zeros_like(seg)
    sub_MASK[seg_mask] = 1
    # save
    vxm.py.utils.save_volfile(sub_MASK, f'{data_path}I0_MASK.nii.gz')
    vt.correct_vox2ras_matrix(f'{data_path}I0_MASK.nii.gz', reference_nifiti='./src/align_norm.nii.gz')
    print("Step 1 completed. Please run the script for Step 2 (parallel transport) before proceeding.")
    
def step3_apply_transported_SVF(prefix='', single_cohort = ''):
    # step3: apply transported SVF on subject 0
    ages = []
    imgs = []
    segs = []
    for npz_file in [i for i in sorted(os.listdir(data_path)) if i.endswith('.npz')]:
        print(f'==== {npz_file} ====')
        data = np.load(f'{data_path}{npz_file}', allow_pickle=True)
        img = data['vol']
        age = data['age']
        disease_condition = data['disease_condition']
        seg = data['synth_seg']

        imgs.append(img)
        segs.append(seg)
        ages.append(age)

    # moving img:
    moving = imgs[0]
    moving = moving[np.newaxis, ..., np.newaxis]
    moving = tf.convert_to_tensor(moving, dtype=tf.float32) # 1x208x176x160x1

    # moving seg:
    moving_seg = segs[0]
    moving_seg = moving_seg[np.newaxis, ..., np.newaxis]
    moving_seg = tf.convert_to_tensor(moving_seg, dtype=tf.float32) # 1x208x176x160x1

    transported_SVFs = sorted([i for i in os.listdir(data_path) if i.startswith(f"transported_SVF{prefix}")])

    if not transported_SVFs:
        raise FileNotFoundError("No transported SVF files found! Please run the parallel transport step first.")

    for i, svf in enumerate(transported_SVFs):
        vel = vxm.py.utils.load_volfile(f'{data_path}{svf}', add_batch_axis=True, add_feat_axis=False).transpose((0,3,2,1,4)) # 1x208x176x160x3x1
        diff_field = VecInt(method='ss', int_steps=5, name='def_field')(vel)
        if prefix == '_half':
            diff_field = RescaleTransform(2.0, name='flowup')(diff_field)
        moved_img = SpatialTransformer()([moving, diff_field])
        moved_seg = SpatialTransformer(interp_method='nearest')([moving_seg, diff_field])

        # save
        moved_img = moved_img.numpy().squeeze()
        vxm.py.utils.save_volfile(moved_img, f'{data_path}synthetic{prefix}_{single_cohort}_{i}.nii.gz')
        vt.correct_vox2ras_matrix(f'{data_path}synthetic{prefix}_{single_cohort}_{i}.nii.gz', reference_nifiti='./src/align_norm.nii.gz')

        moved_seg = moved_seg.numpy().squeeze()
        vxm.py.utils.save_volfile(moved_seg, f'{data_path}synthetic{prefix}_{single_cohort}_{i}_seg.nii.gz')
        vt.correct_vox2ras_matrix(f'{data_path}synthetic{prefix}_{single_cohort}_{i}_seg.nii.gz', reference_nifiti='./src/align_norm.nii.gz')

# Main entry point
def main():
    parser = argparse.ArgumentParser(description="Script of main steps for InBrainSyn.")
    parser.add_argument(
        "--step",
        type=int,
        choices=[1, 3],
        required=True,
        help="Specify the step to run: 1 for Step 1 (Extract SVF), 3 for Step 3 (Apply Transported SVF).",
    )
    parser.add_argument(
        "--is_half",
        action="store_true",
        help="Use half SVF field for Step 1.",
    )
    parser.add_argument(
        "--single_cohort",
        type=str,
        choices=["", "HC", "AD"],
        default="HC",
        help="Specify the cohort: '', 'HC', or 'AD'.",
    )
    args = parser.parse_args()

    if args.step == 1:
        step1_extract_SVF(is_half=args.is_half, single_cohort=args.single_cohort, run_InBrainSyn=True)
    elif args.step == 3:
        step3_apply_transported_SVF(prefix='_half' if args.is_half else '', single_cohort=args.single_cohort)

if __name__ == "__main__":
    main()