#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/8/27
# @Description: Color Filter Array Interpolation


import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import time_cost_decorator, showimg_with_uint8


class CFA:
    """
    Color Filter Array Interpolation
    
    description:
        this is a class for Color Filter Array Interpolation, including bilinear and malvar
    
    step:
        1. padding the inputs
        2. get the pixel value
        3. get the demosaic output
        
    usage:
        dmc = CFA(inputs, bayer_pattern='RGGB', white_level=1023, dmc_method='bilinear')
    """
    def __init__(self, inputs: np.ndarray = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.inputs = inputs
        self.kwargs = kwargs
        self.bayer_pattern = self.kwargs.get('bayer_pattern', 'RGGB')
        self.white_level = self.kwargs.get('white_level', 1023)
        self.cfa_method = self.kwargs.get('CFA_method', 'bilinear')
        self.__check_inputs()
        
        
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.bayer_pattern in ['RGGB', 'BGGR'], "bayer_pattern should be in ['RGGB', 'BGGR'], but got {}".format(self.bayer_pattern)
        assert self.cfa_method in ['malvar', 'bilinear'], "type should be in ['malvar', 'bilinear'], but got {}".format(self.cfa_method)
        assert self.inputs is not None, "inputs is None, please check it"
        assert self.inputs.ndim == 2 and len(self.inputs.shape) == 2, "inputs shape is {}, should be 2 dims cause color filter array".format(self.inputs.shape)
        
    
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
        __dict__ = {
            'malvar': self.__malvar_demosaic,
            'bilinear': self.__bilinear_demosaic,
        }
        return __dict__.pop(self.cfa_method)()
    
    
    @time_cost_decorator
    def __malvar_demosaic(self) -> np.ndarray:
        """
        use malvar demosaic algorithm
        """
        H, W = self.inputs.shape
        self.inputs = self.inputs.astype(np.float32)
        demosaic_output = np.zeros((H, W, 3), dtype = np.float32)
        padding = 2
        self.inputs = self.__padding_inputs(self.inputs, padding)
        
        for i in range(padding, H + padding, 2):
            for j in range(padding, W + padding, 2):
                if self.bayer_pattern == 'RGGB':
                    demosaic_output[i - padding, j - padding] = self.__get_malvar_pixel(color_type = 'r', center_y = i, center_x = j)
                    demosaic_output[i - padding, j - padding + 1] = self.__get_malvar_pixel(color_type = 'gr', center_y = i, center_x = j + 1)
                    demosaic_output[i - padding + 1, j - padding] = self.__get_malvar_pixel(color_type = 'gb', center_y = i + 1, center_x = j)
                    demosaic_output[i - padding + 1, j - padding + 1] = self.__get_malvar_pixel(color_type = 'b', center_y = i + 1, center_x = j + 1)
                elif self.bayer_pattern == 'BGGR':  
                    demosaic_output[i - padding, j - padding] = self.__get_malvar_pixel(color_type = 'b', center_y = i, center_x = j)
                    demosaic_output[i - padding, j - padding + 1] = self.__get_malvar_pixel(color_type = 'gb', center_y = i, center_x = j + 1)
                    demosaic_output[i - padding + 1, j - padding] = self.__get_malvar_pixel(color_type = 'gr', center_y = i + 1, center_x = j)
                    demosaic_output[i - padding + 1, j - padding + 1] = self.__get_malvar_pixel(color_type = 'r', center_y = i + 1, center_x = j + 1)
                elif self.bayer_pattern == 'GRBG':
                    demosaic_output[i - padding, j - padding] = self.__get_malvar_pixel(color_type = 'gr', center_y = i, center_x = j)
                    demosaic_output[i - padding, j - padding + 1] = self.__get_malvar_pixel(color_type = 'r', center_y = i, center_x = j + 1)
                    demosaic_output[i - padding + 1, j - padding] = self.__get_malvar_pixel(color_type = 'b', center_y = i + 1, center_x = j)
                    demosaic_output[i - padding + 1, j - padding + 1] = self.__get_malvar_pixel(color_type = 'gb', center_y = i + 1, center_x = j + 1)
                elif self.bayer_pattern == 'GBRG':
                    demosaic_output[i - padding, j - padding] = self.__get_malvar_pixel(color_type = 'gb', center_y = i, center_x = j)
                    demosaic_output[i - padding, j - padding + 1] = self.__get_malvar_pixel(color_type = 'b', center_y = i, center_x = j + 1)
                    demosaic_output[i - padding + 1, j - padding] = self.__get_malvar_pixel(color_type = 'r', center_y = i + 1, center_x = j)
                    demosaic_output[i - padding + 1, j - padding + 1] = self.__get_malvar_pixel(color_type = 'gr', center_y = i + 1, center_x = j + 1)
                else:
                    raise ValueError("bayer pattern is not support, please check it, should be one of ['RGGB', 'BGGR', 'GRBG', 'GBRG']")
        return np.clip(demosaic_output, 0, self.white_level).astype(np.uint16)
    
    
    @time_cost_decorator
    def __bilinear_demosaic(self) -> np.ndarray:
        """
        use bilinear demosaic algorithm
        """
        H, W = self.inputs.shape
        self.inputs = self.inputs.astype(np.float32)
        demosaic_output = np.zeros((H, W, 3), dtype=np.float32)
        padding = 2
        self.inputs = self.__padding_inputs(self.inputs, padding)
        
        for i in range(padding, H + padding, 2):
            for j in range(padding, W + padding, 2):
                if self.bayer_pattern == 'RGGB':
                    demosaic_output[i - padding, j - padding] = self.__get_bilinear_pixel(color_type = 'r', center_y = i, center_x = j)
                    demosaic_output[i - padding, j - padding + 1] = self.__get_bilinear_pixel(color_type = 'gr', center_y = i, center_x = j + 1)
                    demosaic_output[i - padding + 1, j - padding] = self.__get_bilinear_pixel(color_type = 'gb', center_y = i + 1, center_x = j)
                    demosaic_output[i - padding + 1, j - padding + 1] = self.__get_bilinear_pixel(color_type = 'b', center_y = i + 1, center_x = j + 1)
                elif self.bayer_pattern == 'BGGR':
                    demosaic_output[i - padding, j - padding] = self.__get_bilinear_pixel(color_type = 'b', center_y = i, center_x = j)
                    demosaic_output[i - padding, j - padding + 1] = self.__get_bilinear_pixel(color_type = 'gb', center_y = i, center_x = j + 1)
                    demosaic_output[i - padding + 1, j - padding] = self.__get_bilinear_pixel(color_type = 'gr', center_y = i + 1, center_x = j)
                    demosaic_output[i - padding + 1, j - padding + 1] = self.__get_bilinear_pixel(color_type = 'r', center_y = i + 1, center_x = j + 1)
                elif self.bayer_pattern == 'GRBG':
                    demosaic_output[i - padding, j - padding] = self.__get_bilinear_pixel(color_type = 'gr', center_y = i, center_x = j)
                    demosaic_output[i - padding, j - padding + 1] = self.__get_bilinear_pixel(color_type = 'r', center_y = i, center_x = j + 1)
                    demosaic_output[i - padding + 1, j - padding] = self.__get_bilinear_pixel(color_type = 'b', center_y = i + 1, center_x = j)
                    demosaic_output[i - padding + 1, j - padding + 1] = self.__get_bilinear_pixel(color_type = 'gb', center_y = i + 1, center_x = j + 1)
                elif self.bayer_pattern == 'GBRG':
                    demosaic_output[i - padding, j - padding] = self.__get_bilinear_pixel(color_type = 'gb', center_y = i, center_x = j)
                    demosaic_output[i - padding, j - padding + 1] = self.__get_bilinear_pixel(color_type = 'b', center_y = i, center_x = j + 1)
                    demosaic_output[i - padding + 1, j - padding] = self.__get_bilinear_pixel(color_type = 'r', center_y = i + 1, center_x = j)
                    demosaic_output[i - padding + 1, j - padding + 1] = self.__get_bilinear_pixel(color_type = 'gr', center_y = i + 1, center_x = j + 1)
                else:
                    raise ValueError("bayer pattern is not support, please check it, should be one of ['RGGB', 'BGGR', 'GRBG', 'GBRG']")
        return np.clip(demosaic_output, 0, self.white_level).astype(np.uint16)
                    
    
    def __get_bilinear_pixel(self, color_type :str = 'R', center_y : int = 0, center_x : int = 0) -> int:
        """
        use bilinear algorithm to get pixel 
        """
        if color_type == 'r':
            r_pixel = self.inputs[center_y, center_x]
            g_pixel = (self.inputs[center_y - 1, center_x] + self.inputs[center_y + 1, center_x] + self.inputs[center_y, center_x - 1] + self.inputs[center_y, center_x + 1]) // 4
            b_pixel = (self.inputs[center_y - 1, center_x - 1] + self.inputs[center_y - 1, center_x + 1] + self.inputs[center_y + 1, center_x - 1] + self.inputs[center_y + 1, center_x + 1]) // 4
            return np.array([r_pixel, g_pixel, b_pixel])
        elif color_type == 'gr':
            r_pixel = (self.inputs[center_y, center_x - 1] + self.inputs[center_y, center_x + 1]) // 2
            g_pixel = self.inputs[center_y, center_x]
            b_pixel = (self.inputs[center_y - 1, center_x] + self.inputs[center_y + 1, center_x]) // 2
            return np.array([r_pixel, g_pixel, b_pixel])
        elif color_type == 'gb':
            r_pixel = (self.inputs[center_y - 1, center_x] + self.inputs[center_y + 1, center_x]) // 2
            g_pixel = self.inputs[center_y, center_x]
            b_pixel = (self.inputs[center_y, center_x + 1] + self.inputs[center_y, center_x - 1]) // 2
            return np.array([r_pixel, g_pixel, b_pixel])
        elif color_type == 'b':
            r_pixel = (self.inputs[center_y - 1, center_x - 1] + self.inputs[center_y - 1, center_x + 1] + self.inputs[center_y + 1, center_x - 1] + self.inputs[center_y + 1, center_x + 1]) // 4
            g_pixel = (self.inputs[center_y - 1, center_x] + self.inputs[center_y + 1, center_x] + self.inputs[center_y, center_x - 1] + self.inputs[center_y, center_x + 1]) // 4
            b_pixel = self.inputs[center_y, center_x]
            return np.array([r_pixel, g_pixel, b_pixel])
        
        
    def __get_malvar_pixel(self, color_type :str = 'R', center_y : int = 0, center_x : int = 0) -> int:
        """
        use malvar algorithm to get pixel 
        """
        if color_type == 'r':
            r_pixel = self.inputs[center_y, center_x]
            g_pixel = 4 * self.inputs[center_y, center_x] - self.inputs[center_y - 2, center_x] - self.inputs[center_y + 2, center_x] - self.inputs[center_y, center_x - 2] - self.inputs[center_y, center_x + 2] \
                + 2 * (self.inputs[center_y - 1, center_x] + self.inputs[center_y + 1, center_x] + self.inputs[center_y, center_x - 1] + self.inputs[center_y, center_x + 1])
            b_pixel = 6 * self.inputs[center_y, center_x] - 2 * (self.inputs[center_y - 2, center_x] + self.inputs[center_y + 2, center_x] + self.inputs[center_y, center_x - 2] + self.inputs[center_y, center_x + 2]) \
                + 2 * (self.inputs[center_y - 1, center_x - 1] + self.inputs[center_y - 1, center_x + 1] + self.inputs[center_y + 1, center_x - 1] + self.inputs[center_y + 1, center_x + 1]) 
            g_pixel /= 8
            b_pixel /= 8
            return np.array([r_pixel, g_pixel, b_pixel])
        elif color_type == 'gr':
            g_pixel = self.inputs[center_y, center_x]
            r_pixel = 5 * self.inputs[center_y, center_x] - self.inputs[center_y, center_x - 2] - self.inputs[center_y, center_x + 2] - self.inputs[center_y - 1, center_x - 1] - self.inputs[center_y - 1, center_x + 1] \
                - self.inputs[center_y + 1, center_x - 1] - self.inputs[center_y + 1, center_x + 1] + 0.5 * (self.inputs[center_y - 2, center_x] + self.inputs[center_y + 2, center_x] ) \
                + 4 * (self.inputs[center_y, center_x - 1] + self.inputs[center_y, center_x + 1])
            b_pixel = 5 * self.inputs[center_y, center_x] - self.inputs[center_y - 2, center_x] - self.inputs[center_y + 2, center_x] - self.inputs[center_y - 1, center_x - 1] - self.inputs[center_y -1, center_x + 1] \
                - self.inputs[center_y + 1, center_x + 1] - self.inputs[center_y + 1, center_x - 1] + 0.5 * (self.inputs[center_y, center_x - 2] + self.inputs[center_y, center_x + 2]) \
                + 4 * (self.inputs[center_y - 1, center_x] + self.inputs[center_y + 1, center_x])
            r_pixel /= 8
            b_pixel /= 8
            return np.array([r_pixel, g_pixel, b_pixel])
        elif color_type == 'gb':
            g_pixel = self.inputs[center_y, center_x]
            r_pixel = 5 * self.inputs[center_y, center_x] - self.inputs[center_y - 2, center_x] - self.inputs[center_y + 2, center_x] - self.inputs[center_y - 1, center_x - 1] - self.inputs[center_y + 1, center_x - 1] \
                - self.inputs[center_y - 1, center_x + 1] - self.inputs[center_y + 1, center_x + 1] + 0.5 * (self.inputs[center_y, center_x - 2] + self.inputs[center_y, center_x + 2]) \
                + 4 * (self.inputs[center_y - 1, center_x] + self.inputs[center_y + 1, center_x])
            b_pixel = 5 * self.inputs[center_y, center_x] - self.inputs[center_y, center_x - 2] - self.inputs[center_y, center_x + 2] - self.inputs[center_y - 1, center_x - 1] - self.inputs[center_y - 1, center_x + 1] \
                - self.inputs[center_y + 1, center_x - 1] - self.inputs[center_y + 1, center_x + 1] + 0.5 * (self.inputs[center_y - 2, center_x] + self.inputs[center_y + 2, center_x]) \
                + 4 * (self.inputs[center_y, center_x - 1] + self.inputs[center_y, center_x + 1])
            r_pixel /= 8
            b_pixel /= 8
            return np.array([r_pixel, g_pixel, b_pixel])
        elif color_type == 'b':
            b_pixel = self.inputs[center_y, center_x]
            g_pixel = 4 * self.inputs[center_y, center_x] - self.inputs[center_y - 2, center_x] - self.inputs[center_y + 2, center_x] - self.inputs[center_y, center_x - 2] - self.inputs[center_y, center_x + 2] \
                + 2 * (self.inputs[center_y - 1, center_x] + self.inputs[center_y + 1, center_x] + self.inputs[center_y, center_x - 1] + self.inputs[center_y, center_x + 1])
            r_pixel = 6 * self.inputs[center_y, center_x] - 2 * (self.inputs[center_y - 2, center_x] + self.inputs[center_y + 2, center_x] + self.inputs[center_y, center_x - 2] + self.inputs[center_y, center_x + 2]) \
                + 2 * (self.inputs[center_y - 1, center_x - 1] + self.inputs[center_y - 1, center_x + 1] + self.inputs[center_y + 1, center_x - 1] + self.inputs[center_y + 1, center_x + 1]) 
            g_pixel /= 8
            r_pixel /= 8
            return np.array([r_pixel, g_pixel, b_pixel])
            
            
if __name__ == '__main__':
    import cv2
    import os
    from path import Path
    
    root_path = Path(os.path.abspath(__file__)).parent.parent
    img = np.fromfile(root_path / 'test_images' / 'HisiRAW_2592x1536_10bits_RGGB_Linear_20230116191336.raw', dtype=np.uint16)
    img = img.reshape(1536, 2592)
  
    dmc = CFA(inputs = img)
    dmc_output = dmc.run()
    dmc_output = showimg_with_uint8(dmc_output)
    cv2.imwrite(root_path / 'demo_outputs' / 'dmc2.png', dmc_output[:, :, ::-1])

   

  

        
            
        
      
     