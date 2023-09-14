#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/9/3
# @Description: Brightness Contrast Control


import numpy as np  
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import time_cost_decorator, showimg_with_uint8


class BCC:
    """
    Brightness Contrast Control 
    
    description:
        this is a class for Brightness Contrast Control
    
    step:
        1. get the BCC method
        2. get the BCC brightness
        3. get the BCC contrast
        4. get the BCC output
        
    usage:
        bcc = BCC(inputs, BCC_method='sigmoid', BCC_brightness=0.3, BCC_contrast=0.3)
    """
    def __init__(self, inputs: np.ndarray = None, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.inputs = inputs
        self.kwargs = kwargs
        self.bcc_contrast = self.kwargs.get('BCC_contrast', 0.01)
        self.bcc_bright = self.kwargs.get('BCC_brightness', 10)
        self.__check_inputs()
        self.lut = self.__get_linear_lut()
        
        
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.inputs is not None, 'inputs is None, please check it'
        assert len(self.inputs.shape) == 3 and self.inputs.shape[2] == 3, f'inputs shape should be 3, please check it, now is {len(self.inputs.shape)}'
      
      
    def run(self) -> np.ndarray:
        return self.__brightness_contrast_control()


    @time_cost_decorator
    def __brightness_contrast_control(self) -> np.ndarray:
        """
        Brightness Contrast Control with Linear
        """
        bcc_output = self.inputs.copy()
        bcc_output[..., 0] = self.lut[self.inputs[..., 0]]
        return np.clip(bcc_output, 0, 255).astype(np.uint8)


    def __get_linear_lut(self) -> np.ndarray:
        """
        get linear lut
        """
        lut = np.zeros(256, dtype=np.uint8)
        for i in range(0, 256):
            lut[i] = np.clip(self.bcc_contrast * (i - 127) + i + self.bcc_bright, 0, 255)
        return lut
    
    
if __name__ == "__main__":
    import cv2
    import os
    from path import Path
    
    root_path = Path(os.path.abspath(__file__)).parent.parent
    img = cv2.imread(root_path / 'test_images' / 'pose2.png')
    bcc = BCC(img).run()
    cv2.imwrite(root_path / 'demo_outputs' / 'bcc.png', bcc)