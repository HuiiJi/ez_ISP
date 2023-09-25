#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/9/1
# @Description: Edge Enhancement


import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import time_cost_decorator



class EE:
    """
    Edge Enhancement
    
    description:
        this is a class for Edge Enhancement, it is a part of EE
    
    step:
        1. low pass filter
        2. edge enhancement
        
    usage:
        ee = EE(inputs, edge_enhancement_strength=0.36, edge_enhancement_kernel_radius=21)
    """
    def __init__(self, inputs: np.ndarray = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.inputs = inputs
        self.kwargs = kwargs
        self.edge_enhancement_strength = self.kwargs.pop('edge_enhancement_strength', 0.5)
        self.__check_inputs()
        self.edge_enhancement_kernel = self.__get_edge_enhancement_kernel()

    
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.inputs is not None, 'inputs is None, please check it'
        assert len(self.inputs.shape) == 3, 'inputs shape should be 3, please check it, now is {}'.format(len(self.inputs.shape))
        assert 0 < self.edge_enhancement_strength < 1, 'edge_enhancement_strength should be greater than 0 and less than 1, please check it, now is {}'.format(self.edge_enhancement_strength)
    
    
    def padding_inputs(self, inputs:np.ndarray, padding:int)->np.ndarray:
        """
        padding the inputs

        Args:
            inputs: the inputs of the image
            padding: the padding of the image

        Returns:
            inputs: the padded image
        """
        H, W = inputs.shape[:2]
        C = inputs.shape[2] if len(inputs.shape) == 3 else 1
        inputs = inputs.reshape(H, W, C).astype(np.float32)
        inputs = np.pad(inputs, ((padding, padding), (padding, padding), (0, 0)), 'reflect')
        return inputs
    
        
    def run(self) -> np.ndarray:
        return self.__edge_enhancement()
    
    
    @time_cost_decorator
    def __edge_enhancement(self) -> np.ndarray:
        """
        Edge Enhancement
        """
        edge_enhancement_output = np.zeros_like(self.inputs, dtype = np.float32)
        H, W= self.inputs.shape[:2]
        padding = 1
        self.inputs = self.inputs.astype(np.float32)
        edge_enhancement_output = self.inputs.copy()
        self.inputs = self.padding_inputs(self.inputs, padding)
        """
        img2col
        """
        high_pass_output = np.zeros(((2 * padding + 1)**2, H*W), dtype = np.float32)
        for i in range(padding , H + padding):
            for j in range(padding, W + padding):
                high_pass_output[:, (i - padding) * W + (j - padding):(i - padding)* W + (j - padding) + 1] = self.inputs[i-padding:i+padding+1, j-padding:j+padding+1, 0].reshape((2 * padding + 1)**2, 1)
        high_pass_output = np.matmul(self.edge_enhancement_kernel, high_pass_output).reshape(H,W)
        edge_enhancement_output[..., 0] = edge_enhancement_output[..., 0] + self.edge_enhancement_strength * high_pass_output
        return np.clip(edge_enhancement_output, 0, 255).astype(np.uint8)


    def __get_edge_enhancement_kernel(self) -> np.ndarray:
        """
        get edge enhancement kernel
        """
        edge_enhancement_kernel = np.array([0, -1, 0, -1, 4, -1, 0, -1, 0], dtype = np.float32).reshape(1, -1)
        return edge_enhancement_kernel
    
    
if __name__ == '__main__':
    import cv2
    from path import Path
    import os
    root_path = Path(os.path.abspath(__file__)).parent.parent
    test_image = cv2.imread(root_path / 'test_images' / 'pose2.png')
    edge_enhancement = EE(test_image).run()
    cv2.imwrite(root_path / 'demo_outputs' / 'pose2-ee.png', edge_enhancement)

    
