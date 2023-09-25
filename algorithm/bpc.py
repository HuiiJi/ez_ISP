#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/8/28
# @Description: Bad Pixel Correction


import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from .utils import time_cost_decorator, showimg_with_uint8


class BPC:
    """
    Bad Pixel Correction
    
    description:
        this is a class for Bad Pixel Correction, the algorithm is used for the raw image, it can be used to get the Bad Pixel Correction result
    
    step:
        1. padding the image
        2. get the center pixel and its neighbor pixel
        3. if the center pixel is bad pixel, then replace it with the average of its neighbor pixel
        4. else, keep the center pixel
        
    usage:
        bpc = BadPixelCorrection(inputs, white_level=1023)
    """
    def __init__(self, inputs: np.ndarray = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__() 
        self.inputs = inputs
        self.kwargs = kwargs
        self.bayer_pattern = self.kwargs.get('bayer_pattern', 'RGGB')
        self.white_level = self.kwargs.get('white_level', 1023)
        self.bad_pixel_threshold = self.kwargs.pop('bad_pixel_threshold', 30)
        self.__check_inputs()
        
        
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.inputs is not None, 'inputs is None, please check it'
        assert len(self.inputs.shape) == 2, 'inputs shape should be 2, please check it, now is {}'.format(len(self.inputs.shape))
        assert self.bayer_pattern in ['RGGB', 'BGGR', 'GRBG', 'GBRG'], 'bayer_pattern should be RGGB, BGGR, GRBG, GBRG, please check it, now is {}'.format(self.bayer_pattern)
        assert 0 < self.white_level < 65535, 'white_level should be greater than 0 and less than 65535, please check it, now is {}'.format(self.white_level)
        assert 0 < self.bad_pixel_threshold < 255, 'bad_pixel_threshold should be greater than 0 and less than 255, please check it, now is {}'.format(self.bad_pixel_threshold)


    def __padding_inputs(self, inputs:np.ndarray, padding:int)->np.ndarray:
        """
        padding the inputs

        Args:
            inputs: the inputs of the image
            padding: the padding of the image

        Returns:
            inputs: the padded image
        """
        inputs = np.pad(inputs, ((padding, padding), (padding, padding)), 'reflect')
        return inputs
    
    
    def run(self) ->np.ndarray:
        return self.__bad_pixel_correction()
    
    
    @time_cost_decorator
    def __bad_pixel_correction(self) -> np.ndarray:
        """
        Bad Pixel Correction
        """
        H, W = self.inputs.shape
        self.inputs = self.inputs.astype(np.float32)
        bpc_output = self.inputs.copy()
        padding = 2
        self.inputs = self.__padding_inputs(self.inputs, padding)
        mean_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        mean_kernel /=  np.sum(mean_kernel)
        
        for i in range(padding, H + padding, 2):
            for j in range(padding, W + padding, 2):
                if ((abs(self.inputs[i - 2 : i + 3 : 2, j - 2 : j + 3 : 2] -  self.inputs[i, j])) > self.bad_pixel_threshold).all():
                    bpc_output[i - padding, j - padding] = np.sum(self.inputs[i - 1 : i + 2, j - 1 : j + 2] * mean_kernel)
        del self.inputs
        return np.clip(bpc_output, 0, self.white_level).astype(np.uint16)
                    
                    
if __name__ == "__main__":
    import cv2
    import os
    from path import Path
    
    root_path = Path(os.path.abspath(__file__)).parent.parent
    img = np.fromfile(root_path / 'test_images' / 'HisiRAW_2592x1536_10bits_RGGB_Linear_20230116191336.raw', dtype=np.uint16)
    img = img.reshape(1536, 2592)
    bpc = BPC(inputs=img, bad_pixel_threshold=30.0).run()
    bpc= cv2.demosaicing(bpc, cv2.COLOR_BayerRG2BGR)
    bpc_output = showimg_with_uint8(bpc)
    cv2.imwrite(root_path / 'demo_outputs' / 'bpc.png', bpc_output)
                    
                    
    

                    
                
                    
           
        
             

            

