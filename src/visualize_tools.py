#!/usr/bin/env python
# -*- coding:utf-8 -*-
'''
everything related to visualization of 2D/3D images is here
'''
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import nibabel as nib

def create_new_cmap():
    '''

    Returns: A new colormap created according to FreesurfurLUT

    '''
    from matplotlib import cm
    from matplotlib.colors import ListedColormap
    # create colormap using FreeSurferColorLUT.txt
    with open('./src/FreeSurferColorLUT.txt') as f:
        lines = f.readlines()
    f.close()

    dic_label_rgba = {}
    for line in lines[2:]:
        line = ' '.join(line.split())
        values = line.split(" ")
        if len(line) < 6:
            continue
        dic_label_rgba[int(values[0])] = (values[1], np.array([int(i) / 256 for i in values[2:-1]] + [1]))

    viridis = cm.get_cmap('viridis', 256)
    newcolors = viridis(np.linspace(0, 1, 256))

    for label, value in dic_label_rgba.items():
        value = value[1]
        newcolors[label, :] = value

    return ListedColormap(newcolors)

def plot_3slices(fdata, cmap='gray', common_bar=False, colorbar_from_zero=False, slice_numbs=None, title=''):
    '''

    Args:
        fdata: A ndarray with RAS+ orientation
        cmap: colormap, default='gray'
        common_bar: Show 3 slices using the same colarbar, default=False
        colorbar_from_zero: Normalize colorbar from 0, default=False

    Returns:

    '''
    print(f'shape of fdata {fdata.shape}')
    # Create subplots for the three slices
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    (ax1, ax2, ax3) = axes

    if common_bar:
        # Create a common colorbar
        max_dif = np.max(np.abs(fdata))
        if colorbar_from_zero:
            norm = plt.Normalize(vmin=0, vmax=max_dif)
        else:
            norm = plt.Normalize(vmin=-max_dif, vmax=max_dif)  # specify the range of values to be mapped to colors
        cmap = plt.get_cmap(cmap)  # choose the colormap, 'RdBu_r'
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])  # Set the array that the ScalarMappable references
        cbar = plt.colorbar(sm, ax=axes.ravel().tolist(), shrink=0.6, aspect=10)
    else:
        norm = None

    if slice_numbs == None:
        midx = fdata.shape[0] // 2
        midy = fdata.shape[0] // 2
        midz = fdata.shape[0] // 2
    else:
        midx = slice_numbs[0]
        midy = slice_numbs[1]
        midz = slice_numbs[2]

    # # mask out?
    # fdata[abs(fdata)<0.5] = 0.0

    # Plot the three slices - one from each axis
    ax1.imshow(fdata[midx // 2, :, :].T, cmap=cmap, origin='lower', norm=norm)
    ax1.set_xlabel('Posterior - Anterior')
    ax1.set_ylabel('Inferior - Superior')
    ax1.set_title('Sagittal')
    ax2.imshow(fdata[:, midy, :].T, cmap=cmap, origin='lower', norm=norm)
    ax2.set_xlabel('Left - Right')
    ax2.set_ylabel('Inferior - Superior')
    ax2.set_title('Coronal')
    ax3.imshow(fdata[:, :, midz].T, cmap=cmap, origin='lower', norm=norm)
    ax3.set_xlabel('Left - Right')
    ax3.set_ylabel('Posterior - Anterior')
    ax3.set_title('Axial')
    if isinstance(cmap, str) == False and common_bar == False:
        from matplotlib.pyplot import pcolormesh
        ax1.pcolormesh(fdata[midx, :, :].T, cmap=cmap, rasterized=True, vmin=0, vmax=255)
        ax2.pcolormesh(fdata[:, midy, :].T, cmap=cmap, rasterized=True, vmin=0, vmax=255)
        ax3.pcolormesh(fdata[:, :, midz].T, cmap=cmap, rasterized=True, vmin=0, vmax=255)

    plt.suptitle(title)
    plt.show()


def plot_nifti(file_path, cmap='gray'):
    """
    Loads a NIfTI file from the specified filepath, reorients it to RAS+ orientation,
    and plots three slices - one from each of the three axes.

    Parameters
    ----------
    filepath : str
        The file path to the input NIfTI file.

    Returns
    -------
    None
    """
    # Load Nifti1Image object
    img = nib.load(file_path)

    # Transform image to RAS+ orientation
    img = nib.as_closest_canonical(img)
    img.set_qform(np.eye(4))

    # Extract image data and affine
    fdata = img.get_fdata()
    print(f'Size of fdata = {fdata.shape}')
    aff = img.affine

    # Reorder the image data according to the axis codes
    axcodes = nib.aff2axcodes(aff)
    reorder_idx = [axcodes.index(code) for code in ('R', 'A', 'S')]
    fdata = np.transpose(fdata, reorder_idx)

    plot_3slices(fdata, cmap)


def plot_nifti_difference(file_path1, file_path2, cmap='gray', colorbar_from_zero=False):
    def normalize_input(file_path):
        # Load Nifti1Image object
        img = nib.load(file_path)

        # Transform image to RAS+ orientation
        img = nib.as_closest_canonical(img)
        img.set_qform(np.eye(4))

        # Extract image data and affine
        fdata = img.get_fdata()
        print(f'Size of fdata = {fdata.shape}')
        aff = img.affine

        # Reorder the image data according to the axis codes
        axcodes = nib.aff2axcodes(aff)
        reorder_idx = [axcodes.index(code) for code in ('R', 'A', 'S')]
        fdata = np.transpose(fdata, reorder_idx)
        return fdata

    # load two data and do reduction
    fdata = normalize_input(file_path1) - normalize_input(file_path2)

    plot_3slices(fdata, cmap, True, colorbar_from_zero)

# itk image
def rescale(itkimg):
    minmaxfilter = sitk.MinimumMaximumImageFilter()
    minmaxfilter.Execute(itkimg)
    max_value = minmaxfilter.GetMaximum()
    min_value = minmaxfilter.GetMinimum()

    return sitk.IntensityWindowing(itkimg, windowMinimum=min_value, windowMaximum=max_value,
                                            outputMinimum=0.0, outputMaximum=1)

# numpy image
def crop_ndarray(ndarr_img,  uppoint=[0, 13, 13], out_size = [160, 160, 192], show = False):
    dim1 = uppoint[0]
    dim2 = uppoint[1]
    dim3 = uppoint[2]
    template = ndarr_img[dim1:(dim1 + out_size[0]), dim2:(dim2 + out_size[1]), dim3:(dim3 + out_size[2])]
    # plot
    mid_slices_moving = [np.take(template, template.shape[d] // 2, axis=d) for d in range(3)]
    mid_slices_moving[1] = np.rot90(mid_slices_moving[1], 1)
    mid_slices_moving[2] = np.rot90(mid_slices_moving[2], -1)
    slices(mid_slices_moving, cmaps=['gray'], grid=[1, 3], save=False, show=show)

def slices(slices_in,  # the 2D slices
           titles=None,  # list of titles
           suptitle = None,
           cmaps=None,  # list of colormaps
           norms=None,  # list of normalizations
           do_colorbars=False,  # option to show colorbars on each slice
           grid=False,  # option to plot the images in a grid or a single row
           width=15,  # width in in
           show=True,  # option to actually show the plot (plt.show())
           axes_off=True,
           save=False, # option to save plot
           save_path=None, # save path
           imshow_args=None):
    '''
    plot a grid of slices (2d images)
    taken from voxelmorph + small modification
    '''

    # input processing
    if type(slices_in) == np.ndarray:
        slices_in = [slices_in]
    nb_plots = len(slices_in)
    for si, slice_in in enumerate(slices_in):
        if len(slice_in.shape) != 2:
            assert len(slice_in.shape) == 3 and slice_in.shape[-1] == 3, 'each slice has to be 2d or RGB (3 channels)'
        slices_in[si] = slice_in.astype('float')

    def input_check(inputs, nb_plots, name):
        ''' change input from None/single-link '''
        assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
            'number of %s is incorrect' % name
        if inputs is None:
            inputs = [None]
        if len(inputs) == 1:
            inputs = [inputs[0] for i in range(nb_plots)]
        return inputs

    titles = input_check(titles, nb_plots, 'titles')
    cmaps = input_check(cmaps, nb_plots, 'cmaps')
    norms = input_check(norms, nb_plots, 'norms')
    imshow_args = input_check(imshow_args, nb_plots, 'imshow_args')
    for idx, ia in enumerate(imshow_args):
        imshow_args[idx] = {} if ia is None else ia

    # figure out the number of rows and columns
    if grid:
        if isinstance(grid, bool):
            rows = np.floor(np.sqrt(nb_plots)).astype(int)
            cols = np.ceil(nb_plots / rows).astype(int)
        else:
            assert isinstance(grid, (list, tuple)), \
                "grid should either be bool or [rows,cols]"
            rows, cols = grid
    else:
        rows = 1
        cols = nb_plots

    # prepare the subplot
    fig, axs = plt.subplots(rows, cols)
    if rows == 1 and cols == 1:
        axs = [axs]

    for i in range(nb_plots):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        # turn off axis
        ax.axis('off')

        # add titles
        if titles is not None and titles[i] is not None:
            ax.title.set_text(titles[i])

        # show figure
        im_ax = ax.imshow(slices_in[i], cmap=cmaps[i], interpolation="nearest", norm=norms[i], **imshow_args[i])

    # clear axes that are unnecessary
    for i in range(nb_plots, col * row):
        col = np.remainder(i, cols)
        row = np.floor(i / cols).astype(int)

        # get row and column axes
        row_axs = axs if rows == 1 else axs[row]
        ax = row_axs[col]

        if axes_off:
            ax.axis('off')

    # show the plots
    fig.set_size_inches(width, rows / cols * width)

    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")
    if show:
        plt.tight_layout()
        plt.show()

    if save:
        plt.savefig(save_path + '.png')
    return (fig, axs)

def load_nii(path, way = 'nibabel'):
    '''
    Args:
        path: path of nifti file
        way: nibabel or sitk, sitk follow the same orientation of numpy load a npz
    Returns: imf
    '''
    if way == 'nibabel':
        img = nib.load(path).get_fdata()
    elif way == 'sitk':
        load_img = sitk.ReadImage(path, sitk.sitkFloat32)
        img = sitk.GetArrayFromImage(load_img)
    else:
        raise ValueError('way expected to be nibabel or sitk')
    return(img)

def correct_vox2ras_matrix(wrong_nifit_path, save_nifit_path = None, reference_nifiti = './src/align_norm.nii.gz'):
    '''

    Args:
        wrong_nifit_path: wrong nifiti file, mainly caused by using simpleitk, it uses a
                          different loading principle, need to use it to convert back.
                          reference web page: https://itk.org/pipermail/community/2017-November/013783.html
        save_nifit_path: Can override the old one, or save as another name and/or path
        reference_nifiti: an OASIS3 data which has right orientation

    Returns:
        saved_corrected_nifit_file
    '''
    real_img = sitk.ReadImage(reference_nifiti)
    if wrong_nifit_path.endswith('.nii.gz') or wrong_nifit_path.endswith('.nii'):
        wrong_img_npy = nib.load(wrong_nifit_path).get_fdata()
    else:
        raise ValueError('Input wrong_nifit_path is not a string!')

    wrong_img = sitk.GetImageFromArray(wrong_img_npy)
    wrong_img.SetSpacing(real_img.GetSpacing())
    origin = []
    for x, y in zip(wrong_img.GetSize(), real_img.GetOrigin()):
        if y > 0:
            origin.append(1 * x / 2.0)
        else:
            origin.append(-1 * x / 2.0)
    wrong_img.SetOrigin(origin)
    wrong_img.SetDirection(real_img.GetDirection())

    if save_nifit_path == None:
        sitk.WriteImage(wrong_img, wrong_nifit_path)
    else:
        sitk.WriteImage(wrong_img, save_nifit_path)

def show_3d_mid(load_train_img_np, cmap="gray", title = ''):
    def show_slices(slices, cmap):
       """ Function to display row of image slices """
       fig, axes = plt.subplots(1, len(slices))
       for i, slice in enumerate(slices):
           axes[i].imshow(slice.T, cmap=cmap) #, origin="lower"
           if isinstance(cmap, str) == False:
               from matplotlib.pyplot import pcolormesh
               axes[i].pcolormesh(slice.T, cmap=cmap, rasterized=True, vmin=0, vmax=255)

    H,W,C = load_train_img_np.shape
    slice_0 = load_train_img_np[int(H/2), :, :]
    slice_1 = load_train_img_np[:,int(W/2), :]
    slice_2 = load_train_img_np[:, :, int(C/2)]
    show_slices([slice_0, slice_1, slice_2], cmap)
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()
