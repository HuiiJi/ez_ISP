#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/8/31
# @Description: Gamma Correction


import numpy as np  
from typing import Any, Dict, List, Optional, Tuple, Union  
from utils import time_cost_decorator, showimg_with_uint8


class GMC:
    """
    Gamma Correction
    
    description:
        this is a class for Gamma Correction, the gamma value is 0.6, you can change it
    
    step:
        1. get the gamma table
        2. get the gamma output
        
    usage:
        gma = GMC(inputs, gamma=0.6)
    """
    def __init__(self, inputs: np.ndarray = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.inputs = inputs
        self.kwargs = kwargs
        self.gamma = self.kwargs.pop('gamma', 2.)
        self.gamma_table = self.__gamma_table()
        self.__check_inputs()
        
        
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.inputs is not None, 'inputs is None, please check it'
        assert len(self.inputs.shape) == 3, 'inputs shape should be 3, please check it, now is {}'.format(len(self.inputs.shape))
        assert 0 < self.gamma < 3, 'gamma should be greater than 0 and less than 10, please check it, now is {}'.format(self.gamma)
 
        
    def run(self) -> np.ndarray:
        return self.__gamma_correction()
    
    
    @time_cost_decorator
    def __gamma_correction(self) -> np.ndarray:
        """
        Gamma Correction
        """
        gamma_output = self.gamma_table[self.inputs]
        del self.inputs, self.gamma_table
        return gamma_output
    
    
    def __gamma_table(self) -> np.ndarray:
        """
        Gamma Table
        """
        curve = lambda x : x ** (1.0 / self.gamma)
        gamma_table = np.zeros(256, dtype=np.uint8)
        for i in range(0, 256):
            gamma_table[i] = np.clip(curve(float(i) / 255) * 255 , 0, 255)
        return gamma_table
    

