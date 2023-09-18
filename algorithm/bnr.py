#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/8/23
# @Description: Bayer Noise Reduction


import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import time_cost_decorator, showimg_with_uint8


class BNR:
    """
    Bayer Noise Reduction
    
    description:
        this is a noise reduction class, it can reduce the noise of the image, it can be used in the raw image
    
    step:
        1. padding the inputs
        2. bayer to rggb (not used)
        3. get the noise reduction output
        
    usage:
        bnr = BNR(inputs=inputs, BNR_method='nlm', white_level=1023)
    """
    def __init__(self, inputs:np.ndarray = None, **kwargs:Dict[str, Any])->None:
        super().__init__()
        self.inputs = inputs
        self.kwargs = kwargs
        self.bnr_method = self.kwargs.pop('BNR_method', 'bilateral')
        self.white_level = self.kwargs.get('white_level', 1023)
        self.__check_inputs()
        self.mse_table = self.__get_mse_table()
        
        
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.inputs is not None, f'inputs must be a np.ndarray, please check your input inputs: {self.inputs}'
        assert len(self.inputs.shape) == 2, f'inputs shape should be 2, please check it, now is {len(self.inputs.shape)}'
        assert self.bnr_method in ['nlm', 'mean', 'median', 'bilateral', 'gaussian'], f'bnr_method should be nlm, mean, median, bilateral or gaussian, please check it, now is {self.bnr_method}'
        
    
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

    
    def __bayer2rggb(self, inputs:np.ndarray)->np.ndarray:
        """
        bayer to rggb

        Args:
            inputs: the inputs of the image

        Returns:
            rggb: the rggb image
        """
        rggb = np.zeros((inputs.shape[0] // 2, inputs.shape[1] // 2, 4), dtype=np.float32)
        rggb[..., 0] = inputs[0::2, 0::2]
        rggb[..., 1] = inputs[0::2, 1::2]
        rggb[..., 2] = inputs[1::2, 0::2]
        rggb[..., 3] = inputs[1::2, 1::2]
        return rggb
    
    
    def run(self)->np.ndarray:
        _dict_ = {
            'nlm': self.__nlm,
            'mean': self.__mean,
            'median': self.__median,
            'bilateral': self.__bilateral,
            'gaussian': self.__gaussian_img2col
        }
        return _dict_.pop(self.bnr_method)()
        
    
    @time_cost_decorator
    def __mean(self, radius:Union[np.ndarray, int]=5)->np.ndarray:
        """
        this is a mean filter, it can reduce the noise of the image.
        
        Args:
            radius: the radius of the filter, it must be a odd number

        Returns:
            mean_output: the output of the mean filter  
        """
        assert radius%2 == 1, f'radius must be a odd number, please check your input radius: {radius}'
        
        H, W = self.inputs.shape
        padding = (2 * radius + 1) // 2
        self.inputs = self.inputs.astype(np.float32)
        mean_output = self.inputs.copy()
        self.inputs = self.__padding_inputs(self.inputs, padding)
        mean_kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
        
        for y in range(-radius, radius+1):
            for x in range(-radius, radius+1):
                mean_kernel[y+radius, x+radius] = 1.0
        mean_kernel /= mean_kernel.sum()
    
        for i in range(padding, H+padding):
            for j in range(padding, W+padding):
                mean_output[i-padding, j-padding] = np.sum(self.inputs[i-radius:i+radius+1, j-radius:j+radius+1] * mean_kernel)
        return np.clip(mean_output, 0, self.white_level).astype(np.uint16) 
    
    
    @time_cost_decorator
    def __median(self, radius:Union[np.ndarray, int]=5)->np.ndarray:
        """
        this is a median filter, it can reduce the noise of the image.

        Args:
            radius: the radius of the filter, it must be a odd number
            
        Returns:
            median_output: the output of the median filter       
        """
        assert radius%2 == 1, f'radius must be a odd number, please check your input radius: {radius}'

        H, W = self.inputs.shape
        padding = (2 * radius + 1) // 2
        self.inputs = self.inputs.astype(np.float32)
        median_output = self.inputs.copy()
        self.inputs = self.__padding_inputs(self.inputs, padding)
        
        for i in range(padding, H+padding):
            for j in range(padding, W+padding):
                median_output[i-padding, j-padding] = np.median(self.inputs[i-radius:i+radius+1, j-radius:j+radius+1])
        return np.clip(median_output, 0, self.white_level).astype(np.uint16)
                        
                    
    @time_cost_decorator
    def __gaussian(self, radius:Union[np.ndarray, int]=5, sigma:Union[np.ndarray, int]=10)->np.ndarray:
        """
        this is a gaussian filter, it can reduce the noise of the image.
        
        Args:
            radius: the radius of the filter, it must be a odd number
            sigma: the sigma of the gaussian kernel

        Returns:
            gaussian_output: the output of the gaussian filter   
        """
        assert radius%2 == 1, f'radius must be a odd number, please check your input radius: {radius}'
        assert sigma > 0, f'sigma must be a positive number, please check your input sigma: {sigma}'
        
        H, W = self.inputs.shape
        self.inputs = self.inputs.astype(np.float32)
        padding = (2 * radius + 1) // 2
        gaussian_output = self.inputs.copy()
        self.inputs = self.__padding_inputs(self.inputs, padding)
        gaussian_kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.float32)
        """
        get gaussian kernel
        """
        for y in range(-radius, radius+1):
            for x in range(-radius, radius+1):
                gaussian_kernel[y+radius, x+radius] = np.exp(-(y**2+x**2)/(2*sigma**2))
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        """
        element-wise multiplication and sum
        """
        for i in range(padding, H+padding):
            for j in range(padding, W+padding):
                gaussian_output[i-padding, j-padding] = np.sum(gaussian_kernel * self.inputs[i-padding:i+padding+1, j-padding:j+padding+1])      
        return np.clip(gaussian_output, 0, self.white_level).astype(np.uint16) 
    
    
    @time_cost_decorator
    def __gaussian_img2col(self, radius:Union[np.ndarray, int]=5, sigma:Union[np.ndarray, int]=10)->np.ndarray:
        """ 
        this is a gaussian filter, it can reduce the noise of the image, but it is slower than the __gaussian function.
   
        Args:
            radius: the radius of the filter, it must be a odd number
            sigma: the sigma of the gaussian kernel

        Returns:
            gaussian_output: the output of the gaussian filter   
        """
        assert radius%2 == 1, f'radius must be a odd number, please check your input radius: {radius}'
        assert sigma > 0, f'sigma must be a positive number, please check your input sigma: {sigma}'
        H, W = self.inputs.shape
        C = self.inputs.shape[2] if len(self.inputs.shape) == 3 else 1
        self.inputs = self.inputs.astype(np.float32)
        padding = (2 * radius + 1) // 2
        self.inputs = self.__padding_inputs(self.inputs, padding)
        gaussian_kernel = np.zeros((2*radius+1, 2*radius+1), dtype=np.float32)
        
        for y in range(-radius, radius+1):
            for x in range(-radius, radius+1):
                gaussian_kernel[y+radius, x+radius] = np.exp(-(y**2+x**2)/(2*sigma**2))
        gaussian_kernel = gaussian_kernel / np.sum(gaussian_kernel)
        
        """
        img2col
        """
        gaussian_kernel = gaussian_kernel.reshape(1, 1, -1)
        img_col = np.zeros((C, (2*radius+1)**2, H*W), dtype=np.float32)
        for i in range(padding, H+padding):
            for j in range(padding, W+padding):
                img_col[:, :, (i-padding)*W+(j-padding):(i-padding)*W+(j-padding)+1] = self.inputs[i-padding:i+padding+1, j-padding:j+padding+1].reshape(C, (2*radius+1)**2, 1)
        gaussian_output = np.matmul(gaussian_kernel, img_col).reshape(H, W, C)
        return np.clip(gaussian_output, 0, self.white_level).astype(np.uint16)
        
         
    @time_cost_decorator
    def __bilateral(self, radius:Union[np.ndarray, int]=5, sigma_s:Union[np.ndarray, int]=10, sigma_p:Union[np.ndarray, int]=15)->np.ndarray:
        """
        this is a bilateral filter, it can reduce the noise of the image.

        Args:
            radius: the radius of the filter, it must be a odd number
            sigma_s: the sigma of the spatial gaussian kernel
            sigma_p: the sigma of the pixel gaussian kernel

        Returns:
            bilateral_output: the output of the bilateral filter
        """
        assert radius%2 == 1, f'radius must be a odd number, please check your input radius: {radius}'
        assert sigma_s > 0, f'sigma_s must be a positive number, please check your input sigma_s: {sigma_s}'
        assert sigma_p > 0, f'sigma_p must be a positive number, please check your input sigma_p: {sigma_p}'
        
        H, W = self.inputs.shape
        padding = (2 * radius + 1) // 2
        self.inputs = self.inputs.astype(np.float32)
        bilateral_output = self.inputs.copy()
        self.inputs = self.__padding_inputs(self.inputs, padding)
       
        for i in range(padding, H+padding):
            for j in range(padding, W+padding):
                weights_sum = 0
                pixel_sum = 0
                for y in range(-radius, radius+1):
                    for x in range(-radius, radius+1):
                        spital_weights = -(y**2 + x**2)/(2*sigma_s**2)
                        pixel_weights = -(self.inputs[i,j] - self.inputs[i+y, j+x])**2/(2*sigma_p**2)
                        weights = np.exp(spital_weights + pixel_weights)
                        weights_sum += weights
                        pixel_sum += weights * self.inputs[i+y, j+x]     
                bilateral_output[i-padding, j-padding] = pixel_sum / weights_sum
        return np.clip(bilateral_output, 0, self.white_level).astype(np.uint16)
    
    
    @time_cost_decorator          
    def __nlm(self, search_window_radius:Union[np.ndarray, int]=5, block_window_radius:Union[np.ndarray, int]=1, h:Union[np.ndarray, int]=25)->np.ndarray:
        """
        this is a non-local means filter, it can reduce the noise of the image.
        
        Args:
            search_window: the search window size, it must be a odd number
            block_window: the block window size, it must be a odd number
            h: the parameter h, it must be a int number
            
        Returns:
            nlm_output: the output of the nlm filter
        """
        assert search_window_radius%2 == 1, f'search_window must be a odd number, please check your input search_window: {search_window_radius}'
        assert block_window_radius%2 == 1, f'block_window must be a odd number, please check your input block_window: {block_window_radius}'
        assert isinstance(h, int), f'h must be a int number, please check your input h: {h}'
        assert block_window_radius < search_window_radius, f'block_window must be smaller than search_window, please check your input block_window: {block_window_radius}, search_window: {search_window_radius}'
        
        H, W = self.inputs.shape[:2]
        padding = (2 * search_window_radius + 1) // 2
        self.inputs = self.inputs.astype(np.float32)
        nlm_output = self.inputs.copy()
        self.inputs = self.__padding_inputs(self.inputs, padding)
        
        for i in range(padding, H+padding):
            for j in range(padding, W+padding):
                w_max = 0
                pixel_sum = 0
                weights_sum = 0
                for y in range(2 * search_window_radius + 1 - 2 * block_window_radius - 1):
                    for x in range(2 * search_window_radius + 1 - 2 * block_window_radius - 1):
                        if  y != i or x != j:
                            neighbor_center_y = i - search_window_radius + block_window_radius + y
                            neighbor_center_x = j - search_window_radius + block_window_radius + x
                            center_block = self.inputs[i - block_window_radius: i + block_window_radius + 1, j - block_window_radius: j +  block_window_radius + 1]
                            neighbor_block = self.inputs[neighbor_center_y - block_window_radius: neighbor_center_y + block_window_radius + 1, neighbor_center_x - block_window_radius: neighbor_center_x + block_window_radius + 1]
                            mse_dist = ((neighbor_block - center_block) ** 2) / ((2 * block_window_radius + 1) ** 2)
                            w = np.exp(-np.sum(mse_dist) / (h ** 2))
                            if w > w_max:
                                w_max = w
                            weights_sum += w
                            pixel_sum += w * self.inputs[neighbor_center_y, neighbor_center_x]
                weights = weights_sum + w_max
                pixel = pixel_sum + w_max * self.inputs[i, j]
                nlm_output[i - padding, j - padding] = pixel / weights
        return np.clip(nlm_output, 0, self.white_level).astype(np.uint16)
    
    
    def __get_mse_table(self)->np.ndarray:
        """
        this function is used to get the mse table
        
        Returns:
            mse_table: the mse table
        """
        mse_table = np.zeros((self.white_level+1, self.white_level+1), dtype=np.float32)
        for i in range(0, self.white_level):
            for j in range(0, self.white_level):
                mse_table[i, j] = (i - j) ** 2
        return mse_table
                            
                        
if __name__ == '__main__':
    import cv2
    import os
    from path import Path
   
    root_path = Path(os.path.abspath(__file__)).parent.parent
    img = cv2.imread(root_path / 'test_images' / 'pose2-noisy.png', 0)
    # img = img.reshape(1536, 2592)
    noise_reduce = BNR(inputs=img, BNR_method='nlm')
    nr_output = noise_reduce.run()
    nr_output = showimg_with_uint8(nr_output)
    cv2.imwrite(root_path / 'demo_outputs' / 'bnr.png', nr_output)
   
   
