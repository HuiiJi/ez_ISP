#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/8/30
# @Description: Color Space Conversion


import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import time_cost_decorator


class Y2R:
    """
    Color Space Conversion
    
    description:
        this is a class for Color Space Conversion, now support RGB2YUV and YUV2RGB, more color space will be added in the future
    
    step:
        1. YUV2RGB
        
    usage:
        csc = Y2R(inputs)
    """
    def __init__(self, inputs : np.ndarray = None, **kwargs: Dict[str, Any]) -> None:
        self.inputs = inputs
        self.kwargs = kwargs
        self.__check_inputs()
        
        
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.inputs is not None, 'inputs is None, please check it'
        assert len(self.inputs.shape) == 3, 'inputs shape should be 3, please check it, now is {}'.format(len(self.inputs.shape))
        
        
    def run(self) -> np.ndarray:
        return self.__YUV2RGB()

    
    @time_cost_decorator
    def __YUV2RGB(self) -> np.ndarray:
        """
        YUV2RGB
        """
        YUV2RGB_matrix = np.array([[1.0, 0, 1.402, -128], [1.0, -0.344, -0.714, -128.0], [1.0, 1.772, 0.0, -128.0]])
        self.inputs = self.inputs.astype(np.float32)
        RGB_output = self.inputs.copy()
        RGB_output[..., 0] = self.inputs[..., 0] * YUV2RGB_matrix[0, 0] + (self.inputs[..., 1] + YUV2RGB_matrix[0, 3]) * YUV2RGB_matrix[0, 1] + (self.inputs[..., 2] + YUV2RGB_matrix[0, 3]) * YUV2RGB_matrix[0, 2]
        RGB_output[..., 1] = self.inputs[..., 0] * YUV2RGB_matrix[1, 0] + (self.inputs[..., 1] + YUV2RGB_matrix[1, 3]) * YUV2RGB_matrix[1, 1] + (self.inputs[..., 2] + YUV2RGB_matrix[1, 3]) * YUV2RGB_matrix[1, 2]
        RGB_output[..., 2] = self.inputs[..., 0] * YUV2RGB_matrix[2, 0] + (self.inputs[..., 1] + YUV2RGB_matrix[2, 3]) * YUV2RGB_matrix[2, 1] + (self.inputs[..., 2] + YUV2RGB_matrix[2, 3]) * YUV2RGB_matrix[2, 2]
        return np.clip(RGB_output, 0, 255).astype(np.uint8)

         




        
     
        
        
    
