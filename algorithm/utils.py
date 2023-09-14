import time
import numpy as np

def time_cost_decorator(func):
    """
    Decorator for time cost, print the time cost of the function
    """
    def warp(*args, **kwargs):
        start_time = time.time()
        out = func(*args, **kwargs)
        end_time = time.time()
        print('|',f'{func.__name__}'.ljust(50), f': cost time is {1000 * (end_time - start_time):.2f} ms'.ljust(20),'|') 
        print('-'* 88)
        return out
    return warp


def showimg_with_uint8(img: np.ndarray = None) -> None:
    """
    Show image with uint8
    """
    assert img is not None, 'img is None, please check it'
    assert img.dtype != np.uint8, 'img dtype is {}, should not be uint8'.format(img.dtype)
    img_uint8 = img.astype(np.float32) / (2 ** 10 - 1) * 255
    return img_uint8.astype(np.uint8)
  