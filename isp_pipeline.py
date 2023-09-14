#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/9/11
# @Description: ISP Pipeline


import numpy as np
from path import Path
import os
import yaml
import importlib
import sys


class ISP_Pipeline:
    """
    this is a class for ISP Pipeline

    step:
        1. get the ISP Pipeline from yaml
        2. run the ISP Pipeline
        3. get the ISP Pipeline output

    usage:
        isp = ISP_Pipeline(config_path)
    """
    def __init__(self, config_path: str = None) -> None:
        super().__init__()
        self.config_path = config_path
        self.root_path = Path(os.path.abspath(__file__)).parent
        self.__check_envs()
        self.cfg = self.__from_yaml(self.config_path)
        self.pipe = self.__get_isp_pipeline()
 

    def __check_envs(self) -> None:
        """
        check the inputs
        """
        assert self.config_path is not None, 'config_path is None, please check it'
        assert os.path.exists(self.config_path), f'config_path {self.config_path} is not exists, please check it'
        sys.path.insert(0, self.root_path + '/algorithm')
        sys.path.insert(0, self.root_path + '/config')
        sys.path.insert(0, self.root_path)
        os.makedirs(self.root_path + '/demo_outputs', exist_ok=True)


    def run(self) -> np.ndarray:
        return self.__run_isp_pipeline()
    
    
    def __from_yaml(self, yaml_path):
        """ Instantiation from a yaml file. """
        if not isinstance(yaml_path, str):
            raise TypeError(
                f'expected a path string but given a {type(yaml_path)}'
            )
        with open(yaml_path, 'r', encoding='utf-8') as fp:
            yml = yaml.safe_load(fp)
        return yml


    def __get_isp_pipeline(self) -> None:
        """
        get ISP Pipeline
        """
        enable_pipeline = self.cfg['enable'].items()
        module = [k for k, v in enable_pipeline if v is True]
        pipe = []
        for m in module:
            py = importlib.import_module(f'algorithm.{m.lower()}')
            cla = getattr(py, m)
            pipe.append(cla)
        return pipe
    
    
    def __run_isp_pipeline(self) -> np.ndarray:
        """
        run ISP Pipeline
        """
        from algorithm.fir import FIR
        inp = FIR(**self.cfg).run()
        for p in self.pipe:
            inp = p(inp, **self.cfg).run()
        return self.__save_isp_pipeline_outputs(inp)
    
    
    def __save_isp_pipeline_outputs(self, output: np.ndarray) -> None:
        """
        save ISP Pipeline outputs
        """
        import cv2
        image_id = self.cfg['RAW_img_path'].split('/')[-1].split('.')[0]
        cv2.imwrite(self.root_path / 'demo_outputs' / f'{image_id}.png', output[..., ::-1])
    



