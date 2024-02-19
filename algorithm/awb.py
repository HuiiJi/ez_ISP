#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/8/24
# @Description: Auto White Balance

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import time_cost_decorator, showimg_with_uint8


class AWB:
    """
    Auto White Balance
    
    description:
        this is a auto white balance algorithm, it can be used to get the white balance matrix
    
    step:
        1. get the mean value of the R, G, B channel
        2. get the gain of the R, G, B channel
        3. get the white balance matrix
        4. get the white balance output
        
    usage:
        awb = AWB(inputs)
    """ 
    def __init__(self, inputs: np.ndarray = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.inputs = inputs
        self.kwargs = kwargs
        self.bayer_pattern = self.kwargs.get('bayer_pattern', 'RGGB')
        self.r_gain = self.kwargs.get('r_gain', None)
        self.b_gain = self.kwargs.get('b_gain', None)
        self.white_level = self.kwargs.get('white_level', None)
        self.__check_inputs()
        
        
    def __check_inputs(self)-> None:
        """
        check the inputs
        """
        assert self.inputs is not None, 'inputs is None, please check it'
        assert len(self.inputs.shape) == 2, 'inputs shape should be 2, please check it, now is {}'.format(len(self.inputs.shape))
        assert self.bayer_pattern in ['RGGB', 'BGGR', 'GRBG', 'GBRG'], 'bayer_pattern should be RGGB, BGGR, GRBG, GBRG, please check it, now is {}'.format(self.bayer_pattern)


    def run(self)-> np.ndarray:
        _dect_ = {
                'RGGB': self.__rggb_bayer_auto_white_balance,
                'BGGR': self.__bggr_bayer_auto_white_balance,
                'GRBG': self.__grbg_bayer_auto_white_balance,
                'GBRG': self.__gbrg_bayer_auto_white_balance,
        }
        return _dect_.pop(self.bayer_pattern)()
    
    
    @time_cost_decorator
    def __rggb_bayer_auto_white_balance(self) -> np.ndarray:
        """
        RGGB Bayer Auto White Balance
        """
        self.inputs = self.inputs.astype(np.float32)
        awb_output = self.inputs.copy()
        awb_output[0::2, 0::2] = self.inputs[0::2, 0::2] * self.r_gain
        awb_output[1::2, 1::2] = self.inputs[1::2, 1::2] * self.b_gain
        return np.clip(awb_output, 0, self.white_level).astype(np.uint16)
    
    
    @time_cost_decorator
    def __bggr_bayer_auto_white_balance(self) -> np.ndarray:
        """
        BGGR Bayer Auto White Balance
        """
        self.inputs = self.inputs.astype(np.float32)
        awb_output = self.inputs.copy()
        awb_output[0::2, 0::2] = self.inputs[0::2, 0::2] * self.b_gain
        awb_output[1::2, 1::2] = self.inputs[1::2, 1::2] * self.r_gain
        del self.inputs
        return np.clip(awb_output, 0, self.white_level).astype(np.uint16)
      
    
    @time_cost_decorator
    def __grbg_bayer_auto_white_balance(self) -> np.ndarray:
        """
        GRBG Bayer Auto White Balance
        """
        self.inputs = self.inputs.astype(np.float32)
        awb_output = self.inputs.copy()
        awb_output[0::2, 1::2] = self.inputs[0::2, 1::2] * self.r_gain
        awb_output[1::2, 0::2] = self.inputs[1::2, 0::2] * self.b_gain
        del self.inputs
        return np.clip(awb_output, 0, self.white_level).astype(np.uint16)
    
    
    @time_cost_decorator
    def __gbrg_bayer_auto_white_balance(self) -> np.ndarray:
        """
        GBRG Bayer Auto White Balance
        """
        self.inputs = self.inputs.astype(np.float32)
        awb_output = self.inputs.copy()
        awb_output[0::2, 1::2] = self.inputs[0::2, 1::2] * self.b_gain
        awb_output[1::2, 0::2] = self.inputs[1::2, 0::2] * self.r_gain
        del self.inputs
        return np.clip(awb_output, 0, self.white_level).astype(np.uint16)



if __name__ == '__main__':
    import cv2
    import os
    from path import Path
    
    root_path = Path(os.path.abspath(__file__)).parent.parent
    img = np.fromfile(root_path / 'test_images' / 'HisiRAW_2592x1536_10bits_RGGB_Linear_20230116191336.raw', dtype=np.uint16)
    img = img.reshape(1536, 2592)
    awb = AWB(inputs=img)
    awb_output = awb.run()
    awb_output  = cv2.demosaicing(img, cv2.COLOR_BayerRG2RGB)
    awb_output = showimg_with_uint8(awb_output)
    cv2.imwrite(root_path / 'demo_outputs' / 'awb.png', awb_output[..., ::-1])