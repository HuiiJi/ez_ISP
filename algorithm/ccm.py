#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/8/23
# @Description: Color Correction Matrix

import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union  
from utils import time_cost_decorator, showimg_with_uint8


class CCM:
    """
    Color Correction Matrix
    
    description:
        this is a class for Color Correction Matrix, which is used to correct the color of the image
    
    step:
        1. get the cc matrix
        2. matmul the cc matrix with the image
        
    usage:
        ccm = CCM(inputs)
    """
    def __init__(self, inputs: np.ndarray = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.inputs = inputs
        self.kwargs = kwargs
        self.ccm_matrix = self.kwargs.pop('ccm_matrix', 
                            [[ 1.631906, -0.381807, -0.250099], 
                            [-0.298296, 1.614734, -0.316438], 
                            [0.023770, -0.538501, 1.514732 ]]
                            )
        # self.ccm_matrix = self.kwargs.pop('ccm_matrix',
        #                     [[ 1.9435506 , -0.7152609 , -0.2282897  ],
        #                     [-0.22348748,  1.4704359 , -0.24694835   ],
        #                     [ 0.03258422, -0.69400704,  1.6614228   ]])
        self.white_level = self.kwargs.pop('white_level', 1023)
        self.__check_inputs()
        
        
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.inputs is not None, 'inputs is None, please check it'
        assert len(self.inputs.shape) == 3, 'inputs shape should be 3, please check it, now is {}'.format(len(self.inputs.shape))
        
    
    def run(self) -> np.ndarray:
        return self.__color_correction()
    
    
    @time_cost_decorator
    def __color_correction(self) -> np.ndarray:
        """
        CCM Matrix
        """
        self.inputs = self.inputs.astype(np.float32)
        ccm_matrix = np.array(self.ccm_matrix).T
        ccm_output = np.matmul(self.inputs, ccm_matrix)
        return np.clip(ccm_output, 0, self.white_level).astype(np.uint16)
    


    
