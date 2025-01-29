#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import os
import random
# import argparse
import json 

from numpy.random import seed
from tensorflow.compat.v1 import set_random_seed

from .networks import Generator, conv_block
import tensorflow.keras.layers as KL

# my import
from . import visualize_tools as vt
import voxelmorph as vxm
from voxelmorph.tf.layers import SpatialTransformer, VecInt, RescaleTransform

# ----------------------------------------------------------------------------
# Load configuration from `config.json`
config_path = os.path.join(os.path.dirname(__file__), "config.json")
with open(config_path, "r") as f:
    config = json.load(f)

epochs = config["epochs"]
batch_size = config["batch_size"]
dataset = config["dataset"]
exp_name = config["name"]
lr_g = config["lr_g"]
lr_d = config["lr_d"]
beta1_g = config["beta1_g"]
beta2_g = config["beta2_g"]
beta1_d = config["beta1_d"]
beta2_d = config["beta2_d"]
conditional = config["conditional"]
norm_reg = config["norm_reg"]
oversample = config["oversample"]
d_snout = config["d_snout"]
clip_bckgnd = config["clip_bckgnd"]
reg_loss = config["reg_loss"]
atlas_model = config["gen_config"]
steps = config["steps_per_epoch"]
rng_seed = config["rng_seed"]
start_step = config["start_step"]
resume_ckpt = config["resume_ckpt"]
g_ch = config["g_ch"]
d_ch = config["d_ch"]
init = config["init"]
lazy_reg = config["lazy_reg"]
checkpoint_path = config["checkpoint_path"]
save_path = config["save_path"]

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
