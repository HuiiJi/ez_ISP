#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/8/31
# @Description: Local Tone Mapping


import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import time_cost_decorator, showimg_with_uint8


class LocalToneMapping:
    """
    Local Tone Mapping
    
    description:
        this is a class for Local Tone Mapping, including Sigmoid and Smoothstep, etc
    
    step:
        1. get the LTM method
        2. get the LTM output
        
    usage:
        ltm = LocalToneMapping(inputs, LTM_method='sigmoid')
    """
    def __init__(self, inputs: np.ndarray = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.inputs = inputs
        self.kwargs = kwargs
        self.white_level = self.kwargs.get('white_level', 1023)
        self.local_tone_mapping_dict = self.kwargs.get('LTM_method', 'linear')
        self.__check_inputs()
        _dict_ = {
            'linear': self.__get_linear_lut
        }
        self.lut = _dict_.pop(self.local_tone_mapping_dict)()
        
        
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.inputs is not None, 'inputs is None, please check it'
        assert len(self.inputs.shape) == 3 and self.inputs.shape[2] == 3, f'inputs shape should be 3, please check it, now is {len(self.inputs.shape)}'
        assert self.local_tone_mapping_dict in ['sigmoid', 'smoothstep', 'linear'], f'Invalid LTM method, please check it, now is {self.local_tone_mapping_dict}'
        
        
    def run(self) -> np.ndarray:
        return self.__local_tone_mapping()

    
    @time_cost_decorator
    def __local_tone_mapping(self) -> np.ndarray:   
        """
        Local Tone Mapping with Linear
        """
        ltm_output = self.lut[self.inputs]
        del self.inputs, self.lut
        return ltm_output
    
    
    def __get_linear_lut(self) -> np.ndarray:
        """
        Linear LUT
        """
        lut = np.arange(0, 256, dtype=np.uint8)
        return lut
    
    