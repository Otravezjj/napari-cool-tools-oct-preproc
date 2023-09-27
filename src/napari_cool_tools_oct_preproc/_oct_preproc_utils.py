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
    show_info(f'Generate enface image thread has started')
    layers = generate_enface_image_func(vol=vol,debug=debug,sin_correct=sin_correct,log_correct=log_correct,band_pass_filter=band_pass_filter,CLAHE=CLAHE)
    for layer in layers:
        yield layer
    show_info(f'Generate enface image thread has completed')


def generate_enface_image_func(vol:Image, debug=False, sin_correct=True, log_correct=True, band_pass_filter=True, CLAHE=True)-> List[Layer]:
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

    layers = []

    #show_info(f'Generate enface image thread has started')
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
        layers.append(layer)
        #yield layer

    correct_mip = enface_mip.copy()
    correct_mip.shape = (correct_mip.shape[0],1,correct_mip.shape[1])
    correct_mip = correct_mip.transpose(2,1,0)
    print(f"correct_mip shape: {correct_mip.shape}\n")

    add_kwargs = {"name": f"corrected_MIP_{name}"}
    layer = Layer.create(correct_mip,add_kwargs,layer_type)
    
    if debug:
        layers.append(layer)
        #yield layer

    if sin_correct:
        show_info(f'Correcting enface MIP distortion')
        correct_mip_layer = a_scan_correction_func(layer)
    else:
        correct_mip_layer = layer
    
    if debug:
        layers.append(correct_mip_layer)
        #yield correct_mip_layer
    
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
            if CLAHE:
                out.data = out.data.squeeze()
                out = clahe_pt_func(out)
                out.data.shape = (out.data.shape[0],1,out.data.shape[1])
            if band_pass_filter:
                out = diff_of_gaus_func(out,0.6,6.0)

            if debug:
                layers.append(out)
                #yield out
            else:
                out.data = out.data.squeeze()
                out.data = out.data.transpose(1,0)
                layers.append(out)
                #yield out
        else:
            if debug:
                layers.append(out)
                #yield out
            else:
                pass
            pass

    #show_info(f'Generate enface image thread has completed')
    return layers

def process_bscan_preset(vol:Image, ascan_corr:bool=True, Bandpass:bool=False, CLAHE:bool=False, Med:bool=False):
    """Do initial preprocessing of OCT B-scan volume.
    Args:
        vol (Image): 3D ndarray representing structural OCT data

    Returns:
        processed b-scan volume(Image)"""
    
    process_bscan_preset_thread(vol=vol,ascan_corr=ascan_corr,Bandpass=Bandpass,CLAHE=CLAHE,Med=Med)
    return 

@thread_worker(connect={"returned": viewer.add_layer})
def process_bscan_preset_thread(vol:Image, ascan_corr:bool=True, Bandpass:bool=False, CLAHE:bool=False, Med:bool=False)->Layer:
    """Do initial preprocessing of OCT B-scan volume.
    Args:
        vol (Image): 3D ndarray representing structural OCT data

    Returns:
        processed b-scan volume(Image)"""
    
    show_info(f"B-scan preset thread started")
    output = process_bscan_preset_func(vol=vol,ascan_corr=ascan_corr,Bandpass=Bandpass,CLAHE=CLAHE,Med=Med)
    torch.cuda.empty_cache()
    memory_stats()
    show_info(f"B-scan preset thread completed")
    return output

def process_bscan_preset_func(vol:Image, ascan_corr:bool=True, Bandpass:bool=False, CLAHE:bool=False, Med:bool=False)->Layer:
    """Do initial preprocessing of OCT B-scan volume.
    Args:
        vol (Image): 3D ndarray representing structural OCT data

    Returns:
        processed b-scan volume(Image)
    """
    from napari_cool_tools_img_proc._normalization import normalize_in_range_func, normalize_in_range_pt_func
    from napari_cool_tools_img_proc._denoise import diff_of_gaus_func
    from napari_cool_tools_img_proc._equalization import clahe_pt_func
    from napari_cool_tools_img_proc._luminance import adjust_log_pt_func
    from napari_cool_tools_img_proc._filters import filter_bilateral_pt_func, filter_median_pt_func
    from napari_cool_tools_vol_proc._averaging_tools import average_per_bscan
    from napari_cool_tools_registration._registration_tools import a_scan_correction_func    
    
    #out = normalize_in_range_pt_func(vol,0,1) # add flag and refactor function
    out = normalize_in_range_func(vol,0,1)

    #torch.cuda.empty_cache()

    out = adjust_log_pt_func(out,2.5)
    torch.cuda.empty_cache()
    
    if ascan_corr:
        out = a_scan_correction_func(out)
        torch.cuda.empty_cache()
    if Bandpass:
        out = diff_of_gaus_func(out,1.6,20)
        torch.cuda.empty_cache()
    if CLAHE:
        out = clahe_pt_func(out,1)
        torch.cuda.empty_cache()

    out = normalize_in_range_pt_func(out,0,1)
    torch.cuda.empty_cache()

    out = filter_bilateral_pt_func(out)
    torch.cuda.empty_cache()

    if Med:
        out = filter_median_pt_func(out)
        torch.cuda.empty_cache()

    out = adjust_log_pt_func(out,1.5)
    torch.cuda.empty_cache()

    out = average_per_bscan(out)

    name = f"{out.name}_proc"
    layer_type = 'image'
    add_kwargs = {"name":name}
    out_image = Layer.create(out.data,add_kwargs,layer_type)
    return out_image

def annotation_preset(vol:Image, ascan_corr:bool=True):
    """Do initial preprocessing of OCT B-scan and or enface to prepare them for annotation and analysis.
    Args:
        vol (Image): 3D ndarray representing structural OCT data
        ascan_corr (bool): If true volume and enface image will be corrected for sin wave scanning distortion

    Returns:
        Layers of processed b-scans, processed enface image, b-scan segmentation, enface segmentation
    """
    annotation_preset_thread(vol=vol,ascan_corr=ascan_corr)
    return

@thread_worker(connect={"yielded": viewer.add_layer})
def annotation_preset_thread(vol:Image, ascan_corr:bool=True) -> Layer:
    """Do initial preprocessing of OCT B-scan and or enface to prepare them for annotation and analysis.
    Args:
        vol (Image): 3D ndarray representing structural OCT data
        ascan_corr (bool): If true volume and enface image will be corrected for sin wave scanning distortion

    Returns:
        Layers of processed b-scans, processed enface image, b-scan segmentation, enface segmentation
    """
    show_info(f"Annotation preset thread started")
    layers = annotation_preset_func(vol=vol,ascan_corr=ascan_corr)
    for layer in layers:
        yield layer
    torch.cuda.empty_cache()
    memory_stats()
    show_info(f"B-scan preset thread completed")
    show_info(f"Annotation preset thread completed")


def annotation_preset_func(vol:Image, ascan_corr:bool=True) -> Layer:
    """Do initial preprocessing of OCT B-scan and or enface to prepare them for annotation and analysis.
    Args:
        vol (Image): 3D ndarray representing structural OCT data
        ascan_corr (bool): If true volume and enface image will be corrected for sin wave scanning distortion

    Returns:
        Layers of processed b-scans, processed enface image, b-scan segmentation, enface segmentation
    """
    from napari_cool_tools_registration._registration_tools import a_scan_correction_func
    from napari_cool_tools_segmentation._segmentation import b_scan_pix2pixHD_seg_func, enface_unet_seg_func

    layers = []

    if ascan_corr:
        init = a_scan_correction_func(vol)
    else:
        init = vol

    layers.append(init)

    out = process_bscan_preset_func(init,ascan_corr=False)
    enface_list = generate_enface_image_func(vol,sin_correct=True,band_pass_filter=False,CLAHE=False)
    enface = enface_list[0]
    bscan_seg = b_scan_pix2pixHD_seg_func(init)
    enface_seg_list = enface_unet_seg_func(enface)
    enface_seg = enface_seg_list[0]
    print(type(enface_seg))

    layers.append(out)
    layers.append(bscan_seg)
    layers.append(enface)
    layers.append(enface_seg)

    return layers