#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/9/3
# @Description: Feed In RAW Image


import numpy as np
import rawpy
import cv2
from typing import Any, Dict, List, Optional, Tuple, Union
from utils import time_cost_decorator
import os
import sys

class FIR:
    """
    Feed In RAW Image
    
    description:
        this is a class for Execute ISP algorithm
    
    step:
        1. get the raw image
        2. preprocess the raw image
        
    usage:
        raw = FIR(raw_img_path, Height=4032, Width=3024)
    """
    def __init__(self, **kwargs: Dict[str, Any]) -> None:
        super().__init__()
        self.kwargs = kwargs
        self.raw_img_path = self.kwargs.get('RAW_img_path', None)
        self.raw_height = self.kwargs.get('RAW_Height', None)
        self.raw_width = self.kwargs.get('RAW_Width', None)
        self.__check_inputs()
        
        
    def __check_inputs(self) -> None:
        """
        check the inputs
        """
        assert self.raw_img_path is not None, 'raw_img_path is None, please check it'
        if not isinstance(self.raw_img_path, str):
            raise TypeError(f'raw_img_path should be str, please check it, now is {type(self.raw_img_path)}')
        if self.raw_img_path.split('.')[-1] not in ('dng', 'DNG', 'raw', 'RAW', 'nef', 'NEF', 'cr2', 'CR2', 'tif', 'TIFF', 'png', 'PNG'):
            raise TypeError(f'RAW image should be dng, DNG, raw, RAW, nef, NEF, cr2, CR2, tif, TIFF, please check it, now is {self.raw_img_path.split(".")[-1]}')
        if not os.path.exists(self.raw_img_path):
            raise TypeError(f'RAW image path not exists, please check it, now is {self.raw_img_path}')
        
    
    def run(self) -> np.ndarray:
        """
        get the raw image
        """
        raw_img_dtype = 'Metadata' if self.raw_img_path.split('.')[-1] in ('dng', 'DNG','nef', 'NEF', 'cr2', 'CR2') else 'NoMetadata'
        __dict__ = {
            'Metadata': self.__get_raw_with_metadata,
            'NoMetadata': self.__get_raw_without_metadata
        }
        print('''
        ====================================================================================================
        |                                                                                                  |
        |                                ez - ISP is runing                                            |
        |                                                                                                  |
        ====================================================================================================

        ''')
        return __dict__.pop(raw_img_dtype)()
    
    
    @time_cost_decorator
    def __get_raw_with_metadata(self) -> np.ndarray:
        """
        get the raw image with metadata, such as .dng, .DNG, .nef, .NEF, .cr2, .CR2
        """
        raw = rawpy.imread(self.raw_img_path)
        raw_img = raw.raw_image_visible.astype(np.uint16)
        del raw
        return raw_img
    
    
    @time_cost_decorator
    def __get_raw_without_metadata(self) -> np.ndarray: 
        """
        get the raw image without metadata, such as .raw, .RAW
        """
        raw_img = np.fromfile(self.raw_img_path, dtype=np.uint16).reshape((self.raw_height, self.raw_width)) if self.raw_height is not None else cv2.imread(self.raw_img_path, cv2.IMREAD_UNCHANGED)
        return raw_img


if __name__ == '__main__':
    raw_img_path = '/mnt/cvisp/isp/my_isp/test_images/HisiRAW_2592x1536_10bits_RGGB_Linear_20230116191449.raw'
    raw_img = FIR(raw_img_path=raw_img_path, Height=1536, Width=2592).run()