#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/9/3
# @Description: Chroma Noise Reduction


import numpy as np  
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import time_cost_decorator, showimg_with_uint8


class CNR:
    """
    Chroma Noise Reduction
    
    description:
        this is a class for Chroma Noise Reduction, the method is mean or median, default is mean
    
    step:
        1. get the chroma channel
        2. padding the chroma channel
        3. get the mean or median kernel
        4. get the mean or median value
        5. get the chroma noise reduction output
        
    usage:
        cnr = CNR(inputs, CNR_method='mean', CNR_threshold=0.5)
    """
    def __init__(self, inputs: np.ndarray = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.inputs = inputs
        self.kwargs = kwargs
        self.white_level = self.kwargs.get('white_level', 1024)
        self.cnr_method = self.kwargs.pop('CNR_method', 'gaussian')
        self.cnr_threshold = self.kwargs.pop('CNR_threshold', 0.3)
        self.__check_inputs()
        self.kernel = self.__get_gaussian_kernel()
        
    
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.inputs is not None, 'inputs is None, please check it'
        assert len(self.inputs.shape) == 3 and self.inputs.shape[2] == 3, f'inputs shape should be 3, please check it, now is {len(self.inputs.shape)}'
        assert self.cnr_method in ['mean', 'median', 'gaussian'], f'Invalid CNR method, please check it, now is {self.cnr_method}'
        assert 0 < self.cnr_threshold < 1, f'CNR_threshold should be greater than 0 and less than 1, please check it, now is {self.cnr_threshold}'
        assert 0 < self.white_level < 65535, f'white_level should be greater than 0 and less than 65535, please check it, now is {self.white_level}'
        
        
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
        _dict_ = {
            'gaussian': self.__chroma_noise_reduction_gaussian
        }
        return _dict_.get(self.cnr_method)()
    
    
    @time_cost_decorator
    def __chroma_noise_reduction_gaussian(self, radius:Union[np.ndarray, int]=1, sigma:Union[np.ndarray, int]=20) -> np.ndarray:
        """
        Chroma Noise Reduction with Gaussian
        """
        assert radius%2 == 1, f'radius must be a odd number, please check your input radius: {radius}'
        assert sigma > 0, f'sigma must be a positive number, please check your input sigma: {sigma}'
        
        H, W = self.inputs.shape[:2]
        self.inputs = self.inputs.astype(np.float32)
        chroma_inputs = self.inputs[..., 0].copy()
        cnr_output = self.inputs.copy()
        padding = (2 * radius + 1) // 2
        chroma_inputs = self.__padding_inputs(chroma_inputs, padding)
        """
        img2col
        """
        gaussian_kernel = self.kernel.reshape(1, -1)
        img_col = np.zeros(((2 * radius + 1) ** 2, H*W), dtype=np.float32)
        for i in range(padding, H+padding):
            for j in range(padding, W+padding):
                img_col[:, (i - padding) * W + (j - padding):(i - padding)* W + (j - padding) + 1] = chroma_inputs[i - padding:i + padding + 1, j - padding:j + padding + 1].reshape((2 * radius + 1) ** 2, 1)
        gaussian_output = np.matmul(gaussian_kernel, img_col).reshape(H, W)
        cnr_output[..., 0] = gaussian_output
        return np.clip(cnr_output, 0, 255).astype(np.uint8)
    
    
    def __get_gaussian_kernel(self, radius:Union[np.ndarray, int]=1, sigma:Union[np.ndarray, int]=20) -> np.ndarray:
        """
        get the gaussian kernel
        """
        assert radius%2 == 1, f'radius must be a odd number, please check your input radius: {radius}'
        assert sigma > 0, f'sigma must be a positive number, please check your input sigma: {sigma}'
        
        gaussian_kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.float32)
        for y in range(-radius, radius+1):
            for x in range(-radius, radius+1):
                gaussian_kernel[y+radius, x+radius] = np.exp(-(y**2+x**2)/(2*sigma**2))
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        return gaussian_kernel
 
 
if __name__ == "__main__":
    import cv2
    import os
    from path import Path

    
    root_path = Path(os.path.abspath(__file__)).parent.parent
    img = cv2.imread(root_path / 'test_images' / 'pose2-noisy.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cnr = CNR(img).run()
    cv2.imwrite(root_path / 'test_images' / 'pose2-cnr.png', cnr)

