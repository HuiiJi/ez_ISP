#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/8/28
# @Description: Global Tone Mapping


import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import time_cost_decorator, showimg_with_uint8  



class GTM:
    """
    Global Tone Mapping
    
    description:
        this is a class for Global Tone Mapping, including Sigmoid and Smoothstep, etc
    
    step:
        1. get the GTM method
        2. get the GTM output
        
    usage:
        gtm = GTM(inputs, GTM_method='sigmoid')
    """
    def __init__(self, inputs: np.ndarray = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.inputs = inputs
        self.kwargs = kwargs
        self.white_level = self.kwargs.get('white_level', 1023)
        self.black_level = self.kwargs.get('black_level_r', 64.0)
        self.global_tone_mapping_dict = self.kwargs.get('GTM_method', 'smoothstep')
        self.__check_inputs()
        _dict_ = {
            'smoothstep': self.__get_smoothstep_lut,
            'linear': self.__get_linear_lut
        }
        self.lut = _dict_.pop(self.global_tone_mapping_dict)()
        
        
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.inputs is not None, 'inputs is None, please check it'
        assert len(self.inputs.shape) == 3 and self.inputs.shape[2] == 3, f'inputs shape should be 3, please check it, now is {len(self.inputs.shape)}'
        assert self.global_tone_mapping_dict in ['sigmoid', 'smoothstep', 'linear'], f'Invalid GTM method, please check it, now is {self.global_tone_mapping_dict}'
        assert 0 < self.white_level < 65535, f'white_level should be greater than 0 and less than 65535, please check it, now is {self.white_level}'
        
        
    def run(self) -> np.ndarray:
        return self.__global_tone_mapping()

    
    @time_cost_decorator
    def __global_tone_mapping(self) -> np.ndarray:   
        """
        Global Tone Mapping with Linear
        """
        gtm_output = self.lut[self.inputs]
        del self.inputs, self.lut
        return gtm_output


    def __get_smoothstep_lut(self) -> np.ndarray:
        """
        Get Smoothstep LUT 
        """
        curve = lambda x: 3 * x ** 2 - 2 * x ** 3
        # curve = lambda x: 1.0 - (1.0 - x) ** 3
        # import  matplotlib.pyplot as plt
        # x = np.linspace(0, 1, self.white_level + 1)
        # y = curve(x)
        # plt.plot(x, y)
        # plt.plot(x, x)
        # plt.savefig('/mnt/cvisp/isp/ez_ISP/demo_outputs/smoothstep.png')
        lut = np.zeros(self.white_level + 1, dtype=np.uint8)
        for i in range(0, self.white_level + 1):
            lut[i] = np.clip(curve(float(i) / self.white_level) * 255, 0, 255)
        return lut
    
    
    def __get_linear_lut(self) -> np.ndarray:
        """
        Get Linear LUT
        """
        lut = np.zeros(self.white_level + 1, dtype=np.uint8)
        for i in range(0, self.white_level + 1):
            lut[i] = np.clip(float(i) / self.white_level * 255, 0, 255)
        return lut
    
    
if __name__ == '__main__':
    import cv2
    import os
    from path import Path
    
    root_path = Path(os.path.abspath(__file__)).parent.parent
    img = np.fromfile(root_path / 'test_images' / 'HisiRAW_2592x1536_10bits_RGGB_Linear_20230116191336.raw', dtype=np.uint16)
    img = img.reshape(1536, 2592)
    dmc_cv2 = cv2.cvtColor(img, cv2.COLOR_BAYER_RG2RGB)
    gtm_output = GTM(inputs=dmc_cv2).run()
    cv2.imwrite(root_path / 'demo_outputs' / 'gtm.png', gtm_output[..., ::-1])