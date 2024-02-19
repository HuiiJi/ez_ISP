#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author  : Ji Hui
# @Date    : 2023/9/11
# @Description: run demo

import sys
from isp_pipeline import ISP_Pipeline
from path import Path
import os
import time

sys.path.insert(0, os.path.dirname(__file__) + '/algorithm')
sys.path.insert(0, os.path.dirname(__file__) + '/config')
sys.path.insert(0, os.path.dirname(__file__) + '/assets')
sys.path.insert(0, os.path.dirname(__file__) + '/test_images')
sys.path.insert(0, os.path.dirname(__file__))



def run_demo():
    """
    run demo
    
    description:
        this is a demo for ISP Pipeline
        
    step:
        1. get the ISP Pipeline from yaml
        2. run the ISP Pipeline
        3. get the ISP Pipeline output
    
    usage:
        run_demo()
    """
    root_path = Path(os.path.abspath(__file__)).parent
    yaml_path = root_path / 'config' / 'isp_config.yaml'
    ISP_Pipeline(config_path=yaml_path).run()
    

def get_module_output():
    import os
    import cv2
    import numpy as np
    from path import Path
    from algorithm.utils import showimg_with_uint8
    from algorithm.aaf import AAF
    from algorithm.awb import AWB
    from algorithm.bcc import BCC
    from algorithm.blc import BLC
    from algorithm.bnr import BNR
    from algorithm.bpc import BPC
    from algorithm.ccm import CCM
    from algorithm.cfa import CFA
    from algorithm.cnr import CNR
    from algorithm.gmc import GMC
    from algorithm.ee import EE
    from algorithm.gtm import GTM
    from algorithm.r2y import R2Y
    from algorithm.y2r import Y2R
  
    root_path = Path(os.path.abspath(__file__)).parent
    raw = np.fromfile('/mnt/cvisp/isp/ez_ISP/test_images/2DNR_Case_1_1.raw', dtype=np.uint16).reshape(1080, 1920)
    cv2.imwrite(root_path / 'assets' / 'raw.png', showimg_with_uint8(raw))
    bpc = BPC(raw).run()
    cv2.imwrite(root_path / 'assets' / 'bpc.png', showimg_with_uint8(bpc))
    blc = BLC(bpc).run()
    cv2.imwrite(root_path / 'assets' / 'blc.png', showimg_with_uint8(blc))
    # bnr = BNR(blc).run()
    # cv2.imwrite(root_path / 'assets' / 'bnr.png', showimg_with_uint8(bnr))
    aaf = AAF(blc).run()
    cv2.imwrite(root_path / 'assets' / 'aaf.png', showimg_with_uint8(aaf))
    awb = AWB(aaf).run()
    cv2.imwrite(root_path / 'assets' / 'awb.png', showimg_with_uint8(awb))
    dmc = CFA(awb).run()
    cv2.imwrite(root_path / 'assets' / 'dmc.png', showimg_with_uint8(dmc[..., ::-1]))
    ccm = CCM(dmc).run()
    cv2.imwrite(root_path / 'assets' / 'ccm.png', showimg_with_uint8(ccm[..., ::-1]))
    gtm = GTM(ccm).run()
    cv2.imwrite(root_path / 'assets' / 'gtm.png', gtm[..., ::-1])
    gmc = GMC(gtm).run()
    cv2.imwrite(root_path / 'assets' / 'gmc.png', gmc[..., ::-1])
    yuv = R2Y(gmc).run()
    cv2.imwrite(root_path / 'assets' / 'yuv.png', (Y2R(yuv).run())[..., ::-1])
    cnr = CNR(yuv).run()
    cv2.imwrite(root_path / 'assets' / 'cnr.png', (Y2R(cnr).run())[..., ::-1])
    ee = EE(cnr).run()
    cv2.imwrite(root_path / 'assets' / 'ee.png', (Y2R(ee).run())[..., ::-1])
    bcc = BCC(ee).run()
    cv2.imwrite(root_path / 'assets' / 'bcc.png', (Y2R(bcc).run())[..., ::-1])
    rgb = Y2R(bcc).run()
    cv2.imwrite(root_path / 'assets' / 'rgb.png', rgb[..., ::-1])
    
    
if __name__ == "__main__":
    run_demo()
    # get_module_output()
