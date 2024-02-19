#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/8/23
# @Description: Black Level Correction


import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import time_cost_decorator, showimg_with_uint8

class BLC:
    """
    Black Level Correction
    
    description:
        this class is used to do black level correction, the inputs should be a bayer image
    
    step:
        1. get the black level of the image
        2. do the black level correction
        
    usage:
        blc = BLC(inputs, bayer_pattern='RGGB', white_level=1023, black_level_r=64.0, black_level_gr=64.0, black_level_gb=64.0, black_level_b=64.0)
    """
    def __init__(self, inputs: np.ndarray = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.inputs = inputs
        self.kwargs = kwargs
        self.bayer_pattern = self.kwargs.get('bayer_pattern', None)
        self.white_level = self.kwargs.get('white_level', None)
        self.black_level_r = self.kwargs.get('black_level_r', None)
        self.black_level_gr =  self.kwargs.pop('black_level_gr', None)
        self.black_level_gb = self.kwargs.pop('black_level_gb', None)
        self.black_level_b = self.kwargs.pop('black_level_b', None)
        self.__check_inputs()
        
        
    def __check_inputs(self) -> None:
        """
        check the inputs
        """ 
        assert self.inputs is not None, 'inputs is None, please check it'
        assert len(self.inputs.shape) == 2, 'inputs shape should be 2, please check it, now is {}'.format(len(self.inputs.shape))
        assert self.bayer_pattern in ['RGGB', 'BGGR', 'GRBG', 'GBRG'], 'bayer_pattern should be RGGB, BGGR, GRBG, GBRG, please check it, now is {}'.format(self.bayer_pattern)
        assert 0 < self.white_level < 65535, 'white_level should be greater than 0 and less than 65535, please check it, now is {}'.format(self.white_level)


    def run(self) -> np.ndarray:
        _dict_ = {
            'RGGB': self.__rggb_bayer_black_level_correction,
            'BGGR': self.__bggr_bayer_black_level_correction,
            'GRBG': self.__grbg_bayer_black_level_correction,
            'GBRG': self.__gbrg_bayer_black_level_correction
        }
        return _dict_.pop(self.bayer_pattern)()
        
        
    @time_cost_decorator
    def __rggb_bayer_black_level_correction(self) -> np.ndarray:
        """
        RGGB Bayer Black Level Correction
        """
        self.inputs = self.inputs.astype(np.float32)
        blc_output = self.inputs.copy()
        blc_output[0::2, 0::2] = self.inputs[0::2, 0::2] - self.black_level_r
        blc_output[0::2, 1::2] = self.inputs[0::2, 1::2] - self.black_level_gr
        blc_output[1::2, 0::2] = self.inputs[1::2, 0::2] - self.black_level_gb
        blc_output[1::2, 1::2] = self.inputs[1::2, 1::2] - self.black_level_b
        del self.inputs
        return np.clip(blc_output, 0, self.white_level).astype(np.uint16)
    
    
    @time_cost_decorator
    def __bggr_bayer_black_level_correction(self) -> np.ndarray:
        """
        BGGR Bayer Black Level Correction
        """
        self.inputs = self.inputs.astype(np.float32)
        blc_output = self.inputs.copy()
        blc_output[0::2, 0::2] = self.inputs[0::2, 0::2] - self.black_level_b
        blc_output[0::2, 1::2] = self.inputs[0::2, 1::2] - self.black_level_gb
        blc_output[1::2, 0::2] = self.inputs[1::2, 0::2] - self.black_level_gr
        blc_output[1::2, 1::2] = self.inputs[1::2, 1::2] - self.black_level_r
        del self.inputs
        return np.clip(blc_output, 0, self.white_level).astype(np.uint16)
    
    
    @time_cost_decorator
    def __grbg_bayer_black_level_correction(self) -> np.ndarray:
        """
        GRBG Bayer Black Level Correction
        """
        self.inputs = self.inputs.astype(np.float32)
        blc_output = self.inputs.copy()
        blc_output[0::2, 0::2] = self.inputs[0::2, 0::2] - self.black_level_gr
        blc_output[0::2, 1::2] = self.inputs[0::2, 1::2] - self.black_level_r
        blc_output[1::2, 0::2] = self.inputs[1::2, 0::2] - self.black_level_b
        blc_output[1::2, 1::2] = self.inputs[1::2, 1::2] - self.black_level_gb
        del self.inputs
        return np.clip(blc_output, 0, self.white_level).astype(np.uint16)

    
    @time_cost_decorator
    def __gbrg_bayer_black_level_correction(self) -> np.ndarray:
        """
        GBRG Bayer Black Level Correction
        """
        self.inputs = self.inputs.astype(np.float32)
        blc_output = self.inputs.copy()
        blc_output[0::2, 0::2] = self.inputs[0::2, 0::2] - self.black_level_gb
        blc_output[0::2, 1::2] = self.inputs[0::2, 1::2] - self.black_level_b
        blc_output[1::2, 0::2] = self.inputs[1::2, 0::2] - self.black_level_r
        blc_output[1::2, 1::2] = self.inputs[1::2, 1::2] - self.black_level_gr
        del self.inputs
        return np.clip(blc_output, 0, self.white_level).astype(np.uint16)
    

    
if __name__ == '__main__':
    import cv2
    import os
    from path import Path
        
    root_path = Path(os.path.abspath(__file__)).parent.parent
    img = np.fromfile(root_path / 'test_images' / 'HisiRAW_2592x1536_10bits_RGGB_Linear_20230116191336.raw', dtype=np.uint16)
    img = img.reshape(1536, 2592)
    blc = BLC(inputs=img, bayer_pattern='RGGB', black_level_r=64.0, black_level_gr=64.0, black_level_gb=64.0, black_level_b=64.0)
    blc_output = blc.run()
    blc_output = showimg_with_uint8(blc_output)
    cv2.imwrite(root_path / 'demo_outputs' / 'blc.png', blc_output)
