"""
In this Package, some networks is implemented. 

Supported network:
    - NetworkInNetwork
    - Resnet34
"""

from .network_in_network import NIN as NetworkInNetwork
from .my_resnet34 import MyResNet34

from .nets_utlis import load_net as load
from .nets_utlis import save_net as save


__version__ = '1.0.0'
__author__ = 'Ali Hedayatnia, M.Sc. Student of Artificial Intelligence @ University of Tehran'
