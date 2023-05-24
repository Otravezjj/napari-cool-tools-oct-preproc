"""
This module contains code for OCT data preprocessing.
"""

from typing import List
from napari.utils.notifications import show_info
from napari_cool_tools_io import viewer
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker

def generate_enface_image(vol:Image, debug=False, sin_correct=True, log_correct=True, band_pass_filter=True, CLAHE=True):
    """Generate enface image from OCT volume.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        debug (Bool): If True output intermediate vol manipulations
        filtering (Bool): If True ouput undergoes series of filters to enhance image

    Returns:
        List of napari Layers containing enface interpretation of OCT data and any selected debug layers

    """
    generate_enface_image_thread(vol=vol,debug=debug,sin_correct=sin_correct,log_correct=log_correct,band_pass_filter=band_pass_filter,CLAHE=CLAHE)
    return

@thread_worker(connect={"yielded": viewer.add_layer})
def generate_enface_image_thread(vol:Image, debug=False, sin_correct=True, log_correct=True, band_pass_filter=True, CLAHE=True)-> List[Layer]:
    """Generate enface image from OCT volume.

    Args:
        vol (Image): 3D ndarray representing structural OCT data
        debug (Bool): If True output intermediate vol manipulations
        filtering (Bool): If True ouput undergoes series of filters to enhance image

    Yields:
        List of napari Layers containing enface interpretation of OCT data and any selected debug layers

    """
    from napari_cool_tools_registration._registration_tools import a_scan_correction_func, a_scan_reg_subpix_gen, a_scan_reg_calc_settings_func
    from napari_cool_tools_img_proc._normalization import normalize_data_in_range_pt_func
    from napari_cool_tools_img_proc._denoise import diff_of_gaus_func
    from napari_cool_tools_img_proc._equalization import clahe_pt_func
    from napari_cool_tools_img_proc._luminance import adjust_log_pt_func

    show_info(f'Generate enface image thread has started')
    data = vol.data
    name = f"Enface_{vol.name}"
    layer_type = "image"

    show_info(f'Generating initial enface MIP')
    yx = data.transpose(1,2,0)
    enface_mip = yx.max(0)
    print(f"enface_mip shape: {enface_mip.shape}\n")

    if debug:
        add_kwargs = {"name": f"init_MIP_{name}"}
        layer = Layer.create(enface_mip,add_kwargs,layer_type)
        yield layer

    correct_mip = enface_mip.copy()
    correct_mip.shape = (correct_mip.shape[0],1,correct_mip.shape[1])
    correct_mip = correct_mip.transpose(2,1,0)
    print(f"correct_mip shape: {correct_mip.shape}\n")

    add_kwargs = {"name": f"corrected_MIP_{name}"}
    layer = Layer.create(correct_mip,add_kwargs,layer_type)
    
    if debug:
        yield layer

    if sin_correct:
        show_info(f'Correcting enface MIP distortion')
        correct_mip_layer = a_scan_correction_func(layer)
    else:
        correct_mip_layer = layer
    
    if debug:
        yield correct_mip_layer
    
    show_info(f'Calculating Optimal Subregions for subpixel registration')
    settings = a_scan_reg_calc_settings_func(correct_mip_layer)
    if debug:
        show_info(f"{settings['region_num']}")

    show_info(f'Completing subpixel registration of A-scans')
    outs = list(a_scan_reg_subpix_gen(correct_mip_layer,settings))
    for i,out in enumerate(outs):
        if  i == len(outs) -1:

            out.data = normalize_data_in_range_pt_func(out.data,0,1)

            if log_correct:
                out = adjust_log_pt_func(out)
            if band_pass_filter:
                out = diff_of_gaus_func(out,1.0,20.0)
            if CLAHE:
                out = clahe_pt_func(out)

            if debug:
                yield out
            else:
                out.data = out.data.squeeze()
                yield out
        else:
            if debug:
                yield out
            else:
                pass
            pass

    show_info(f'Generate enface image thread has completed')