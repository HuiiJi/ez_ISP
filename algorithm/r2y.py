#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/8/30
# @Description: Color Space Conversion


import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import time_cost_decorator


class R2Y:
    """
    Color Space Conversion
    
    description:
        this is a class for Color Space Conversion, now support RGB2YUV and YUV2RGB, more color space will be added in the future
    
    step:
        1. RGB2YUV
        
    usage:
        csc = R2Y(inputs)
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
        return self.__RGB2YUV()
    
    
    @time_cost_decorator
    def __RGB2YUV(self) -> np.ndarray:
        """
        RGB2YUV
        """
        RGB2YUV_matrix = np.array([[0.299, 0.587, 0.114, 0.0], [-0.168, -0.331, 0.5, 128.0], [0.5, -0.418, -0.081, 128.0]])
        self.inputs = self.inputs.astype(np.float32)
        YUV_output = self.inputs.copy()
        YUV_output[..., 0] = self.inputs[..., 0] * RGB2YUV_matrix[0, 0] + self.inputs[..., 1] * RGB2YUV_matrix[0, 1] + self.inputs[..., 2] * RGB2YUV_matrix[0, 2] + RGB2YUV_matrix[0, 3]
        YUV_output[..., 1] = self.inputs[..., 0] * RGB2YUV_matrix[1, 0] + self.inputs[..., 1] * RGB2YUV_matrix[1, 1] + self.inputs[..., 2] * RGB2YUV_matrix[1, 2] + RGB2YUV_matrix[1, 3]
        YUV_output[..., 2] = self.inputs[..., 0] * RGB2YUV_matrix[2, 0] + self.inputs[..., 1] * RGB2YUV_matrix[2, 1] + self.inputs[..., 2] * RGB2YUV_matrix[2, 2] + RGB2YUV_matrix[2, 3]
        del self.inputs
        return np.clip(YUV_output, 0, 255).astype(np.uint8)
    

         




        
     
        
        
    
