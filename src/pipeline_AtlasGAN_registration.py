#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import random
import argparse

from numpy.random import seed
from tensorflow.compat.v1 import set_random_seed

from .networks import Generator, conv_block
import tensorflow.keras.layers as KL

# my import
from . import visualize_tools as vt
import voxelmorph as vxm
from voxelmorph.tf.layers import SpatialTransformer, VecInt, RescaleTransform

# ----------------------------------------------------------------------------
# Set up CLI arguments:
# TODO: replace with a config json. CLI is unmanageably large now.
# TODO: add option for type of discriminator augmentation.

parser = argparse.ArgumentParser()

parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--dataset', type=str, default='OASIS3')
parser.add_argument('--name', type=str, default='experiment_name')
parser.add_argument('--d_train_steps', type=int, default=1)
parser.add_argument('--g_train_steps', type=int, default=1)

# TTUR for training GAN, already set the default values in consistent with appendices
parser.add_argument('--lr_g', type=float, default=1e-4)
parser.add_argument('--lr_d', type=float, default=3e-4)
parser.add_argument('--beta1_g', type=float, default=0.0)
parser.add_argument('--beta2_g', type=float, default=0.9)
parser.add_argument('--beta1_d', type=float, default=0.0)
parser.add_argument('--beta2_d', type=float, default=0.9)

parser.add_argument(
    '--unconditional', dest='conditional', default=True, action='store_false',
)
parser.add_argument(
    '--nonorm_reg', dest='norm_reg', default=True, action='store_false',
)  # Not used in the paper.
parser.add_argument(
    '--oversample', dest='oversample', default=True, action='store_false',
)
parser.add_argument(
    '--d_snout', dest='d_snout', default=False, action='store_true',
)
parser.add_argument(
    '--noclip', dest='clip_bckgnd', default=True, action='store_false',
)  # should be True, updated
parser.add_argument('--reg_loss', type=str,
                    default='NCC')  # One of {'NCC', 'NonSquareNCC'}. Not used NonSquareNCC in paper
parser.add_argument('--losswt_reg', type=float, default=1.0)
parser.add_argument('--losswt_gan', type=float, default=0.1)
parser.add_argument('--losswt_tv', type=float, default=0.00)  # Not used in the paper.
parser.add_argument('--losswt_gp', type=float,
                    default=1e-3)  # TODO: Gradient penalty for discriminator loss. Need to be adjusted according to dataset. Important!!!
parser.add_argument('--gen_config', type=str, default='ours')  # One of {'ours', 'voxelmorph'}.
parser.add_argument('--steps_per_epoch', type=int, default=1000)
parser.add_argument('--rng_seed', type=int, default=33)
parser.add_argument('--start_step', type=int,
                    default=0)  # Not used in paper. GAN training is active from the first iteration.
parser.add_argument('--resume_ckpt', type=int, default=0)  # checkopint
parser.add_argument('--g_ch', type=int, default=32)
parser.add_argument('--d_ch', type=int, default=64)
parser.add_argument('--init', type=str, default='default')  # One of {'default', 'orthogonal'}.
parser.add_argument('--lazy_reg', type=int, default=1)  # Not used in the paper.

# my arguments
parser.add_argument('--checkpoint_path', type=str,
                    default='/home/fjr/data/trained_models/Atlas-GAN/training_checkpoints/gploss_1e_4_dataset_OASIS3_eps200_Gconfig_ours_normreg_True_lrg0.0001_lrd0.0003_cond_True_regloss_NCC_lbdgan_0.1_lbdreg_1.0_lbdtv_0.0_lbdgp_0.0001_dsnout_False_start_0_clip_True/')
parser.add_argument('--save_path', type=str, default='/home/fjr/data/trained_models/Atlas-GAN/my_plot_1e-4/')

args = parser.parse_args()

# my CLI
checkpoint_path = args.checkpoint_path  # None
save_path = args.save_path  # None

# Get CLI information:
epochs = args.epochs
batch_size = args.batch_size
dataset = args.dataset
exp_name = args.name
lr_g = args.lr_g
lr_d = args.lr_d
beta1_g = args.beta1_g
beta2_g = args.beta2_g
beta1_d = args.beta1_d
beta2_d = args.beta2_d
conditional = args.conditional
reg_loss = args.reg_loss
norm_reg = args.norm_reg
oversample = args.oversample
atlas_model = args.gen_config
steps = args.steps_per_epoch
lambda_gan = args.losswt_gan
lambda_reg = args.losswt_reg
lambda_tv = args.losswt_tv
lambda_gp = args.losswt_gp
g_loss_wts = [lambda_gan, lambda_reg, lambda_tv]
start_step = args.start_step
rng_seed = args.rng_seed
resume_ckpt = args.resume_ckpt
d_snout = args.d_snout
clip_bckgnd = args.clip_bckgnd
g_ch = args.g_ch
d_ch = args.d_ch
init = args.init
lazy_reg = args.lazy_reg

# ----------------------------------------------------------------------------
# Set RNG seeds

seed(rng_seed)
set_random_seed(rng_seed)
random.seed(rng_seed)

# ----------------------------------------------------------------------------
# Initialize data generators

# Change these if working with new dataset:
if dataset == 'dHCP':
    fpath = './data/dHCP2/npz_files/T2/train/*.npz'
    avg_path = (
        './data/dHCP2/npz_files/T2/linearaverage_100T2_train.npz'
    )
    n_condns = 1
elif dataset == 'pHD':
    fpath = './data/predict-hd/npz_files/train_npz/*.npz'
    avg_path = './data/predict-hd/linearaverageof100.npz'
    n_condns = 3
elif dataset == 'OASIS3':
    main_path = '/media/fjr/My Passport/data/OASIS3/'  # /data/OASIS3/ or /proj/OASIS3_atlasGAN/ or /media/fjr/My Passport/data/OASIS3/
    fpath = main_path + 'all_npz/'
    avg_path = main_path + 'linearaverageof100.npz'
    n_condns = 3
else:
    raise ValueError('dataset expected to be dHCP, pHD or OASIS3')

# define a registration model
def Registration(
    ch=32,
    normreg=False,
    input_resolution=[160, 192, 160, 1],
    output_vel=False,
):
    image_inputs = tf.keras.layers.Input(shape=input_resolution)
    new_atlas    = tf.keras.layers.Input(shape=input_resolution)

    init = None
    vel_init = tf.keras.initializers.RandomNormal(
        mean=0.0,
        stddev=1e-5,
    )

    # Registration network. Taken from vxm:
    # Encoder:
    inp = KL.concatenate([image_inputs, new_atlas])
    d1 = conv_block(inp, ch, stride=2, instancen=normreg, init=init)
    d2 = conv_block(d1, ch, stride=2, instancen=normreg, init=init)
    d3 = conv_block(d2, ch, stride=2, instancen=normreg, init=init)
    d4 = conv_block(d3, ch, stride=2, instancen=normreg, init=init)

    # Bottleneck:
    dres = conv_block(d4, ch, instancen=normreg, init=init)

    # Decoder:
    d5 = conv_block(dres, ch, mode='up', instancen=normreg, init=init)
    d5 = KL.concatenate([d5, d3])

    d6 = conv_block(d5, ch, mode='up', instancen=normreg, init=init)
    d6 = KL.concatenate([d6, d2])

    d7 = conv_block(d6, ch, mode='up', instancen=normreg, init=init)
    d7 = KL.concatenate([d7, d1])

    d7 = conv_block(
        d7, ch, mode='const', instancen=normreg, init=init,
    )
    d7 = conv_block(
        d7, ch, mode='const', instancen=normreg, init=init,
    )
    d7 = conv_block(d7, ch//2, mode='const', activation=False, init=init)

    # Get velocity field:
    d7 = tf.pad(d7, [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]], "REFLECT")

    vel = KL.Conv3D(
        filters=3,
        kernel_size=3,
        padding='valid',
        use_bias=True,
        kernel_initializer=vel_init,
        name='vel_field',
    )(d7)

    # Get diffeomorphic displacement field:
    diff_field = VecInt(method='ss', int_steps=5, name='def_field')(vel)

    # # Get moving average of deformations:
    # diff_field_ms = MeanStream(name='mean_stream', cap=100)(diff_field)
    #
    # # compute regularizers on diff_field_half for efficiency:
    # diff_field_half = 1.0 * diff_field
    vel_field  = RescaleTransform(2.0, name='flowup_vel_field')(vel)
    diff_field = RescaleTransform(2.0, name='flowup')(diff_field)
    moved_atlas = SpatialTransformer()([new_atlas, diff_field])

    if output_vel:
        ops = [moved_atlas, diff_field, vel_field, vel]
    else:
        ops = [moved_atlas, diff_field, vel_field]

    return tf.keras.Model(
        inputs=[image_inputs, new_atlas],
        outputs=ops,
    )

def apply_half_svf(vel, moving_img):
    tf.keras.backend.clear_session()
    moving_image = moving_img[np.newaxis, ..., np.newaxis]
    vel  = vel[np.newaxis, ...]
    moving_image = tf.convert_to_tensor(moving_image, dtype=tf.float32)
    vel = tf.convert_to_tensor(vel, dtype=tf.float32)

    # Get diffeomorphic displacement field:
    diff_field = VecInt(method='ss', int_steps=5, name='def_field')(vel)
    diff_field = RescaleTransform(2.0, name='flowup')(diff_field)
    moved_img = SpatialTransformer()([moving_image, diff_field])
    return moved_img.numpy().squeeze()

def WeightLoading_Registration_Block(main_path, checkpoint_path, n_condns, output_vel=False):
    avg_path = main_path + 'linearaverageof100.npz'

    avg_img = np.load(avg_path)['vol']  # TODO: make generic fname in npz

    vol_shape = avg_img.shape  # calculate [208, 176, 160] for OASIS3 dataset

    # avg_batch = np.repeat(
    #     avg_img[np.newaxis, ...], batch_size, axis=0,
    # )[..., np.newaxis]

    # ----------------------------------------------------------------------------
    # Initialize networks

    generator = Generator(
        ch=g_ch,
        atlas_model=atlas_model,
        conditional=conditional,
        normreg=norm_reg,
        clip_bckgnd=clip_bckgnd,
        input_resolution=[*vol_shape, 1],
        initialization=init,
        n_condns=n_condns,
    )

    # ----------------------------------------------------------------------------
    # Set up Checkpoints
    checkpoint = tf.train.Checkpoint(
        generator=generator,
    )

    # restore checkpoint from the latest trained model:
    if checkpoint_path:
        checkpoint.restore(
            tf.train.latest_checkpoint(checkpoint_path)
        ).expect_partial()
    else:
        raise ValueError('Testing phase, please provide checkpoint path!')

    registration_model = Registration(
        ch=g_ch,
        normreg=norm_reg,
        input_resolution=[*vol_shape, 1],
        output_vel=output_vel
    )

    # construct weight layer names
    def get_layers_name_with_weights(generator):
        weights_layers = []
        for layer_id, layer in enumerate(generator.layers):
            if len(layer.trainable_weights) > 0 or len(layer.non_trainable_weights) > 0:
                weights_layers.append(layer.name)
        return weights_layers

    weights_layers_generator = get_layers_name_with_weights(generator)
    weights_layers_registration=get_layers_name_with_weights(registration_model)
    # load weight layer by layer, references: https://www.gcptutorials.com/post/how-to-get-weights-of-layers-in-tensorflow
    # https://stackoverflow.com/questions/43702323/how-to-load-only-specific-weights-on-keras
    start_generator = weights_layers_generator.index('conv3d_12')
    for i, layer in enumerate(weights_layers_registration):
        generator_layer = weights_layers_generator[start_generator+i]
        print(f'Loading weights for layer {layer} from generator layer {generator_layer}')
        registration_model.get_layer(layer).set_weights(generator.get_layer(generator_layer).get_weights())

    print("loading end")
    return registration_model

def template_generator(main_path,  input_age, checkpoint_path):
    input_condns = np.expand_dims(np.array((input_age / 97.1726095890411)), axis=0)
    n_condns = 1

    avg_path = main_path + 'linearaverageof100.npz'

    avg_img = np.load(avg_path)['vol']  # TODO: make generic fname in npz

    vol_shape = avg_img.shape  # calculate [216, 190, 172] for OASIS3 dataset

    avg_batch = np.repeat(
        avg_img[np.newaxis, ...], batch_size, axis=0,
    )[..., np.newaxis]

    avg_input = tf.convert_to_tensor(avg_batch, dtype=tf.float32)
    # ----------------------------------------------------------------------------
    # Initialize networks

    generator = Generator(
        ch=g_ch,
        atlas_model=atlas_model,
        conditional=conditional,
        normreg=norm_reg,
        clip_bckgnd=clip_bckgnd,
        input_resolution=[*vol_shape, 1],
        initialization=init,
        n_condns=n_condns,
    )

    # ----------------------------------------------------------------------------
    # Set up Checkpoints
    checkpoint = tf.train.Checkpoint(
        generator=generator,
    )

    # restore checkpoint from the latest trained model:
    if checkpoint_path:
        checkpoint.restore(
            tf.train.latest_checkpoint(checkpoint_path)
        ).expect_partial()
    else:
        raise ValueError('Testing phase, please provide checkpoint path!')

    # ----------------------------------------------------------------------------
    # Set up generator training loop

    @tf.function
    def get_inputs(unconditional_inputs, conditional_inputs):
        """If conditionally training, append condition tensor to network inputs."""
        if conditional:
            return unconditional_inputs + conditional_inputs
        else:
            return unconditional_inputs

    _, _, sharp_atlases, _ = generator(
            get_inputs([avg_input, avg_input], [input_condns]),
            training=False,
        )

    atlasmax = tf.reduce_max(sharp_atlases).numpy()  # find the max value
    print("atlasmax = {}".format(atlasmax))

    return tf.nn.relu(sharp_atlases.numpy().squeeze()).numpy() / atlasmax  # with normalization


def registrator(moving_image, fixed_image, checkpoint_path, main_path, n_condns, output_vel=False):
    tf.keras.backend.clear_session()
    fixed_image = fixed_image[np.newaxis, ..., np.newaxis]
    moving_image = moving_image[np.newaxis, ..., np.newaxis]

    fixed_image = tf.convert_to_tensor(fixed_image, dtype=tf.float32)
    moving_image = tf.convert_to_tensor(moving_image, dtype=tf.float32)

    registration_model = WeightLoading_Registration_Block(main_path, checkpoint_path, n_condns,  output_vel=output_vel)

    opts = registration_model([fixed_image, moving_image])

    return opts

# given two inputs [fixed_image, moving_image], save [moved_atlas, diff_field, vel_field]
def extract_and_save(fixed_image, moving_image, save_path, save_name, save_moved_nii=False, save_vel_nii=False):
    os.makedirs(save_path, exist_ok=True)

    fixed_image = tf.convert_to_tensor(fixed_image, dtype=tf.float32)
    moving_image = tf.convert_to_tensor(moving_image, dtype=tf.float32)

    registration_model = WeightLoading_Registration_Block(checkpoint_path)

    tf.keras.backend.clear_session()
    [moved_atlas, diff_field, vel_field] = registration_model([fixed_image, moving_image])

    print(f'Moved image shape = {moved_atlas.numpy().squeeze().shape}, save as {save_path}{save_name}.')

    np.savez_compressed(
        save_path+save_name+'.npz',
        moved= moved_atlas.numpy().squeeze(),
        diff = diff_field.numpy().squeeze(),
        vel  = vel_field.numpy().squeeze()
    )

    if save_moved_nii is True:
        atlasmax = tf.reduce_max(moved_atlas).numpy() # find the max value
        print("atlasmax = {}".format(atlasmax))

        template = tf.nn.relu(moved_atlas.numpy().squeeze()).numpy()/ atlasmax  # with normalization
        # template = sharp_atlases.numpy().squeeze() # without normalization
        # # use PSR transform as default affine
        # affine = np.array([[0, 0, -1, 0],  # nopep8
        #                    [1, 0, 0, 0],  # nopep8
        #                    [0, -1, 0, 0],  # nopep8
        #                    [0, 0, 0, 1]], dtype=float)  # nopep8
        # pcrs = np.append(np.array(template.shape[:3]) / 2, 1)
        # affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        # vxm.py.utils.save_volfile(template, save_path+save_name+'_moved.nii.gz', affine)
        vxm.py.utils.save_volfile(template, save_path + save_name + '_moved.nii.gz')
        vt.correct_vox2ras_matrix(save_path + save_name + '_moved.nii.gz')

    if save_vel_nii is True:
        #atlasmax = tf.reduce_max(vel_field).numpy() # find the max value
        #print("atlasmax = {}".format(atlasmax))

        #template = tf.nn.relu(vel_field.numpy().squeeze()).numpy()/ atlasmax  # with normalization
        # template = sharp_atlases.numpy().squeeze() # without normalization
        # # use PSR transform as default affine
        # affine = np.array([[0, 0, -1, 0],  # nopep8
        #                    [1, 0, 0, 0],  # nopep8
        #                    [0, -1, 0, 0],  # nopep8
        #                    [0, 0, 0, 1]], dtype=float)  # nopep8
        # pcrs = np.append(np.array(template.shape[:3]) / 2, 1)
        # affine[:3, 3] = -np.matmul(affine, pcrs)[:3]
        # vxm.py.utils.save_volfile(template, save_path+save_name+'_vel.nii.gz', affine)
        vxm.py.utils.save_volfile(vel_field.numpy().squeeze(), save_path + save_name + '_vel.nii.gz')
        vt.correct_vox2ras_matrix(save_path + save_name + '_vel.nii.gz')

def transform(vol, loc_shift, interp_method='linear', indexing='ij', fill_value=None):
    import neurite as ne
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow

    Essentially interpolates volume vol at locations determined by loc_shift.
    This is a spatial transform in the sense that at location [x] we now have the data from,
    [x + shift] so we've moved data.

    Args:
        vol (Tensor): volume with size vol_shape or [*vol_shape, C]
            where C is the number of channels
        loc_shift: shift volume [*new_vol_shape, D] or [*new_vol_shape, C, D]
            where C is the number of channels, and D is the dimentionality len(vol_shape)
            If loc_shift is [*new_vol_shape, D], it applies to all channels of vol
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'
        fill_value (default: None): value to use for points outside the domain.
            If None, the nearest neighbors will be used.

    Return:
        new interpolated volumes in the same size as loc_shift[0]

    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """

    # parse shapes.
    # location volshape, including channels if available
    loc_volshape = loc_shift.shape[:-1]
    if isinstance(loc_volshape, (tf.compat.v1.Dimension, tf.TensorShape)):
        loc_volshape = loc_volshape.as_list()

    # volume dimensions
    nb_dims = len(vol.shape) - 1
    is_channelwise = len(loc_volshape) == (nb_dims + 1)
    assert loc_shift.shape[-1] == nb_dims, \
        'Dimension check failed for ne.utils.transform(): {}D volume (shape {}) called ' \
        'with {}D transform'.format(nb_dims, vol.shape[:-1], loc_shift.shape[-1])

    # location should be mesh and delta
    mesh = ne.utils.volshape_to_meshgrid(loc_volshape, indexing=indexing)  # volume mesh
    for d, m in enumerate(mesh):
        if m.dtype != loc_shift.dtype:
            mesh[d] = tf.cast(m, loc_shift.dtype)
    loc = [mesh[d] + loc_shift[..., d] for d in range(nb_dims)]

    # if channelwise location, then append the channel as part of the location lookup
    if is_channelwise:
        loc.append(mesh[-1])

    # test single
    return ne.utils.interpn(vol, loc, interp_method=interp_method, fill_value=fill_value)