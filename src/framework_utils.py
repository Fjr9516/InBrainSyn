#!/usr/bin/env python 
# -*- coding:utf-8 -*-
import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as KL
from voxelmorph.tf.layers import SpatialTransformer, VecInt, RescaleTransform
from .networks import conv_block, Generator

import matplotlib.pyplot as plt
import seaborn as sns

# set default paras
g_ch = 32
atlas_model = 'ours'
conditional = True
norm_reg = True
clip_bckgnd = True
init = 'default'

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

def WeightLoading_Registration_Block(main_path, checkpoint_path, n_condns, output_vel=False):
    avg_path = main_path + 'linearaverageof100.npz'

    avg_img = np.load(avg_path)['vol']  # TODO: make generic fname in npz

    vol_shape = avg_img.shape  # calculate [208, 176, 160] for OASIS3 dataset

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
        ch = g_ch,
        normreg = norm_reg,
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
        registration_model.get_layer(layer).set_weights(generator.get_layer(generator_layer).get_weights())

    print("loading end")
    return registration_model

def registrator(moving_image, fixed_image, checkpoint_path, main_path, n_condns, output_vel = False):
    tf.keras.backend.clear_session()
    fixed_image = fixed_image[np.newaxis, ..., np.newaxis]
    moving_image = moving_image[np.newaxis, ..., np.newaxis]

    fixed_image = tf.convert_to_tensor(fixed_image, dtype=tf.float32)
    moving_image = tf.convert_to_tensor(moving_image, dtype=tf.float32)

    registration_model = WeightLoading_Registration_Block(main_path, checkpoint_path, n_condns,  output_vel = output_vel)

    opts = registration_model([fixed_image, moving_image])

    return opts


def template_generator(main_path,  input_age, checkpoint_path):
    input_condns  = np.expand_dims(np.array((input_age / 97.1726095890411)), axis=0)
    n_condns = 1

    avg_path = main_path + 'linearaverageof100.npz'

    avg_img = np.load(avg_path)['vol']  # TODO: make generic fname in npz

    vol_shape = avg_img.shape  # calculate [216, 190, 172] for OASIS3 dataset

    avg_batch = np.repeat(
        avg_img[np.newaxis, ...], 1, axis=0,
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

    return tf.nn.relu(sharp_atlases.numpy().squeeze()).numpy() / atlasmax  # with normalization


def apply_mask_diff(input_array, labels, segmentation1, segmentation2):
    '''
    Args:
        input_array:
        labels:
        segmentation1: morphological smaller segmentation
        segmentation2: morphological bigger segmentation

    Returns:
    '''
    def get_mask_from_labels(segmentation):
        mask = np.zeros_like(segmentation, dtype=np.uint8)
        for label in labels:
            mask[segmentation == label] = 1
        # print(f"# mask = {mask.sum()}")
        return mask
    if get_mask_from_labels(segmentation2).sum() > get_mask_from_labels(segmentation1).sum():
        dif_mask = get_mask_from_labels(segmentation2) - get_mask_from_labels(segmentation1)
    else:
        dif_mask = get_mask_from_labels(segmentation1) - get_mask_from_labels(segmentation2)

    # create a new array with the same shape as the input array
    masked_array = np.zeros_like(input_array)
    # set the masked pixels to the input values
    masked_array[dif_mask == 1] = input_array[dif_mask == 1]
    return masked_array, dif_mask


def calcu_biomarkers(labels, segmentation, wk, u0, quantile = 0):
    x, y, z, d = u0.shape

    # calculate tk_3d
    u0_norm = np.sum(u0 ** 2, axis=3, keepdims=True)
    numerator = np.sum(wk * u0, axis=3, keepdims=True)

    if quantile != 0:
        # set up a threshold
        th = np.quantile(u0_norm, quantile)
    else:
        th = 0
    tk_3d = np.divide(numerator, u0_norm, out=np.zeros_like(numerator), where=u0_norm > 0)

    # calculate w_ad
    tk_3d_broadcasted = np.broadcast_to(tk_3d, (x, y, z, d))
    w_ad = wk - tk_3d_broadcasted * u0

    mask = np.zeros_like(segmentation, dtype=np.uint8)
    if labels == "wholebrain":
        mask = np.ones_like(segmentation, dtype=np.uint8)
        mask[segmentation == 0] = 0
        mask[u0_norm.squeeze() <= th] = 0
    elif labels == "diff_ventricles":
        labels_ventricles = [4, 14, 15, 43]
        mask = apply_mask_diff(tk_3d, labels_ventricles, segmentation[0], segmentation[1])[1]
        mask[u0_norm.squeeze() <= th] = 0
    else:
        for label in labels:
            mask[segmentation == label] = 1
        mask[u0_norm.squeeze() <= th] = 0

    # calculate masked tk_3d and w_ad
    def apply_mask(input_array, mask):
        # create a new array with the same shape as the input array
        masked_array = np.zeros_like(input_array)
        # set the masked pixels to the input values
        masked_array[mask == 1] = input_array[mask == 1]
        return masked_array

    masked_tk_3d = apply_mask(tk_3d, mask)
    masked_w_ad = apply_mask(w_ad, mask)

    t_hc = np.sum(masked_tk_3d) / mask.sum()
    U, V, W = masked_w_ad[..., 0], masked_w_ad[..., 1], masked_w_ad[..., 2]
    t_ad = (np.sum(np.sqrt(U ** 2 + V ** 2 + W ** 2)) / mask.sum())
    return [t_hc, t_ad, mask.sum()]


def plot_adjusted_scores_on_regions(df, th, component, scan_scores, scan_age, scan_cdr, y_label=''):
    import statsmodels
    from statannotations.Annotator import Annotator

    plt.rcParams.update({'font.size': 14})
    subcat_palette = sns.dark_palette("#8BF", reverse=True, n_colors=6)

    # manually found outlier in CDR=1: OAS30853: 79.58 age
    df = df.drop(df[df['subject_ids'] == 'OAS30853'].index)

    df['disease_categroy'] = np.where((df.disease_condition == 0), 'CN', df.scan_cdr)

    def age_effect(df, dv, single_scan=False):
        # TODO: fix the hardcoding
        AS_paras = [0.43975772860025175,
                    0.8001156059010084,
                    0.4724442755234878,
                    0.5572996640765091]
        ADS_paras = [0.0034318404542013495,
                     -0.0034169561442378964,
                     0.0034040817385265552,
                     0.001191197431178439]
        if "wb_t" in dv:
            if "hc" in dv:
                params = AS_paras[0]
            else:
                params = ADS_paras[0]
        elif "ven_t" in dv:
            if "hc" in dv:
                params = AS_paras[1]
            else:
                params = ADS_paras[1]
        elif "hipp_t" in dv:
            if "hc" in dv:
                params = AS_paras[2]
            else:
                params = ADS_paras[2]
        else:
            if "hc" in dv:
                params = AS_paras[3]
            else:
                params = ADS_paras[3]
        if single_scan:
            oup = params * (scan_age - 60) # set offset from reference age 60
        else:
            oup = params * (df['subject_ages'] - 60) # set offset from reference age 60
        return oup

    df[f'wb_t{component}_{th[0]}_diff'] = df[f'wb_t{component}_{th[0]}'] - age_effect(df, f'wb_t{component}_{th[0]}')
    df[f'ven_t{component}_{th[1]}_diff'] = df[f'ven_t{component}_{th[1]}'] - age_effect(df, f'ven_t{component}_{th[1]}')
    df[f'hipp_t{component}_{th[2]}_diff'] = df[f'hipp_t{component}_{th[2]}'] - age_effect(df,
                                                                                          f'hipp_t{component}_{th[2]}')
    df[f'vendiff_t{component}_{th[3]}_diff'] = df[f'vendiff_t{component}_{th[3]}'] - age_effect(df,
                                                                                                f'vendiff_t{component}_{th[3]}')

    # normalize to 0 according to mean for CN group
    save_means = [df[f'wb_t{component}_{th[0]}_diff'][df['disease_categroy'] == 'CN'].mean(),
                  df[f'ven_t{component}_{th[1]}_diff'][df['disease_categroy'] == 'CN'].mean(),
                  df[f'hipp_t{component}_{th[2]}_diff'][df['disease_categroy'] == 'CN'].mean(),
                  df[f'vendiff_t{component}_{th[3]}_diff'][df['disease_categroy'] == 'CN'].mean()]

    df[f'wb_t{component}_{th[0]}_diff'] = df[f'wb_t{component}_{th[0]}_diff'] - df[f'wb_t{component}_{th[0]}_diff'][
        df['disease_categroy'] == 'CN'].mean()
    df[f'ven_t{component}_{th[1]}_diff'] = df[f'ven_t{component}_{th[1]}_diff'] - df[f'ven_t{component}_{th[1]}_diff'][
        df['disease_categroy'] == 'CN'].mean()
    df[f'hipp_t{component}_{th[2]}_diff'] = df[f'hipp_t{component}_{th[2]}_diff'] - \
                                            df[f'hipp_t{component}_{th[2]}_diff'][df['disease_categroy'] == 'CN'].mean()
    df[f'vendiff_t{component}_{th[3]}_diff'] = df[f'vendiff_t{component}_{th[3]}_diff'] - \
                                               df[f'vendiff_t{component}_{th[3]}_diff'][
                                                   df['disease_categroy'] == 'CN'].mean()

    # adj scan_scores
    if component == 'hc':
        id_score = 0
    else:
        id_score = 1
    new_scores = []
    regions_names = ['wb_t', 'ven_t', 'hipp_t', 'vendiff_t']
    for region_id, region_scores in enumerate(scan_scores):
        region_scores[id_score] = region_scores[id_score] - age_effect(df, f'{regions_names[region_id]}{component}',
                                                                       True) - save_means[region_id]
        new_scores.append(region_scores)

    # Decide which pairs of data to annotate
    plotting_parameters = {
        'data': df,
        'x': 'disease_categroy',
        'palette': subcat_palette[1:],
        'order': ['CN', "0.0", "0.5", "1.0", "2.0"]
    }

    pairs = [('CN', '0.0'),
             ('CN', '0.5'),
             ('CN', '1.0'),
             ('0.0', '0.5'),
             ('0.0', '1.0'),
             ('0.5', '1.0'),
             ]

    # Plot the data points and the linear line
    fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(18, 6.5))

    # Iterate over each subplot
    plot_titles = ["Whole Brain", "Ventricles", "Hippocampi & Amygdala", "Ventricle Edges"]
    for i, ax in enumerate(axs):
        column_name = f'wb_t{component}_{th[0]}_diff' if i == 0 else f'ven_t{component}_{th[1]}_diff' if i == 1 else f'hipp_t{component}_{th[2]}_diff' if i == 2 else f'vendiff_t{component}_{th[3]}_diff'

        # Plot with seaborn
        sns.boxplot(y=column_name, **plotting_parameters, ax=ax)

        # add a point
        order = ['CN', "0.0", "0.5", "1.0", "2.0"]
        ax.plot(order.index(str(scan_cdr)), new_scores[i][id_score], 'r*', markersize=10)

        # Set the title and axis labels for the current subplot
        if component == 'hc':
            ax.set_ylim((-40, 50))
        else:
            ax.set_ylim((-0.5, 1.0))

        ax.set_title(plot_titles[i], y=0.99, pad=-14)
        ax.set_ylabel(y_label)
        ax.set_xlabel('CDR')

        # Add annotations
        annotator = Annotator(ax, pairs, y=column_name, **plotting_parameters)
        annotator.configure(test='t-test_ind', comparisons_correction="bonferroni", verbose=False, text_format='star',
                            loc='outside').apply_and_annotate()

    # Adjust the spacing between subplots
    plt.tight_layout()

    # Show the plot
    plt.show()
    return new_scores