"""
This module contains code for OCT data preprocessing.
"""

from typing import List
from napari.utils.notifications import show_info
from napari_cool_tools_io import viewer
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from napari_cool_tools_io import torch,viewer,memory_stats

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
                out = adjust_log_pt_func(out,2.5)
            if band_pass_filter:
                out = diff_of_gaus_func(out,1.0,20.0)
            if CLAHE:
                out.data = out.data.squeeze()
                out = clahe_pt_func(out)
                out.data.shape = (out.data.shape[0],1,out.data.shape[1])

            if debug:
                yield out
            else:
                out.data = out.data.squeeze()
                out.data = out.data.transpose(1,0)
                yield out
        else:
            if debug:
                yield out
            else:
                pass
            pass

    show_info(f'Generate enface image thread has completed')

def process_bscan_preset(vol:Image, ascan_corr:bool=True, Bandpass:bool=False, CLAHE:bool=False):
    """Do initial preprocessing of OCT B-scan volume.
    Args:
        vol (Image): 3D ndarray representing structural OCT data

    Returns:
        processed b-scan volume(Image)"""
    
    process_bscan_preset_thread(vol=vol,ascan_corr=ascan_corr,Bandpass=Bandpass,CLAHE=CLAHE)
    return 

@thread_worker(connect={"returned": viewer.add_layer})
def process_bscan_preset_thread(vol:Image, ascan_corr:bool=True, Bandpass:bool=False, CLAHE:bool=False)->Layer:
    """Do initial preprocessing of OCT B-scan volume.
    Args:
        vol (Image): 3D ndarray representing structural OCT data

    Returns:
        processed b-scan volume(Image)"""
    
    show_info(f"B-scan preset thread started")
    output = process_bscan_preset_func(vol=vol,ascan_corr=ascan_corr,Bandpass=Bandpass,CLAHE=CLAHE)
    torch.cuda.empty_cache()
    memory_stats()
    show_info(f"B-scan preset thread completed")
    return output

def process_bscan_preset_func(vol:Image, ascan_corr:bool=True, Bandpass:bool=False, CLAHE:bool=False)->Layer:
    """Do initial preprocessing of OCT B-scan volume.
    Args:
        vol (Image): 3D ndarray representing structural OCT data

    Returns:
        processed b-scan volume(Image)
    """
    from napari_cool_tools_img_proc._normalization import normalize_in_range_pt_func
    from napari_cool_tools_img_proc._denoise import diff_of_gaus_func
    from napari_cool_tools_img_proc._equalization import clahe_pt_func
    from napari_cool_tools_img_proc._luminance import adjust_log_pt_func
    from napari_cool_tools_vol_proc._averaging_tools import average_per_bscan
    from napari_cool_tools_registration._registration_tools import a_scan_correction_func    
    
    out = normalize_in_range_pt_func(vol,0,1)

    out = adjust_log_pt_func(out,2.5)
    
    if ascan_corr:
        out = a_scan_correction_func(out)
        torch.cuda.empty_cache()
    if Bandpass:
        out = diff_of_gaus_func(out,1.6,20)
    if CLAHE:
        out = clahe_pt_func(out,1)
        torch.cuda.empty_cache()

    out = normalize_in_range_pt_func(out,0,1)

    out = adjust_log_pt_func(out,1.5)

    out = average_per_bscan(out)

    name = f"{out.name}_proc"
    layer_type = 'image'
    add_kwargs = {"name":name}
    out_image = Layer.create(out.data,add_kwargs,layer_type)
    return out_image