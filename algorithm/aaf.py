#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/8/27
# @Description: Anti Aliasing Filter

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import time_cost_decorator, showimg_with_uint8


class AAF:
    """
    Anti Aliasing Filter

    description:
        this is a anti aliasing filter algorithm, it can be used to get the anti aliasing filter result
    
    step:
        1. padding the inputs
        2. get the center pixel and its neighbor pixel
        3. if the center pixel is bad pixel, then replace it with the average of its neighbor pixel
        4. else, keep the center pixel
        
    usage:
        aaf = AAF(inputs, white_level=1023)
    """

    def __init__(self, inputs: np.ndarray = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.inputs = inputs    
        self.kwargs = kwargs
        self.white_level = self.kwargs.get('white_level', 1023)
        self.__check_inputs()
    
    
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.inputs is not None, "inputs is None, please check it"
        assert self.inputs.ndim == 2 and len(self.inputs.shape) == 2, f'inputs shape is {self.inputs.shape}, should be 2 dims cause color filter array'

        
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
    

    def run(self) -> np.ndarray:
        return self.__anti_aliasing_filter_img2col()

      
    @time_cost_decorator
    def __anti_aliasing_filter_img2col(self) -> np.ndarray:
        """
        Bayer Anti Aliasing Filter
        """
        H, W = self.inputs.shape
        self.inputs = self.inputs.astype(np.float32)
        padding = 2
        self.inputs = self.__padding_inputs(self.inputs, padding)
        kernel = np.array([[1,0,1,0,1],
                           [0,0,0,0,0],
                           [1,0,8,0,1],
                           [0,0,0,0,0],
                           [1,0,1,0,1]], dtype=np.float32) 
        kernel /= np.sum(kernel)
        """
        img2col
        """
        kernel = kernel.reshape(1, -1)
        img_col = np.zeros(((2 * padding + 1) ** 2, H * W), dtype=np.float32)
        for i in range(padding, H + padding):
            for j in range(padding, W + padding):
                img_col[:, (i - padding) * W + (j - padding):(i - padding) * W + (j - padding) + 1] = self.inputs[i - 2:i + 3, j - 2:j + 3].reshape((2 * padding + 1 ) ** 2, 1)
        aaf_output = np.matmul(kernel, img_col).reshape(H, W)
        return np.clip(aaf_output, 0, self.white_level).astype(np.uint16)
                
                
                
if __name__ == '__main__':
    import cv2
    import os
    from path import Path
    
    root_path = Path(os.path.abspath(__file__)).parent.parent
    img = np.fromfile(root_path / 'test_images' / 'HisiRAW_2592x1536_10bits_RGGB_Linear_20230116191336.raw', dtype=np.uint16)
    img = img.reshape(1536, 2592)
    aaf = AAF(inputs=img).run()
    aaf  = cv2.demosaicing(aaf, cv2.COLOR_BayerRG2RGB)
    aaf_output = showimg_with_uint8(aaf)
    cv2.imwrite(root_path / 'assets' / 'awb.png', aaf_output[..., ::-1])
    
   
                        
                           
         
                            
                       
 
        
                


