import torch
from .cnet import Cnet
from .mnet import Mnet

class BCDnet(torch.nn.Module):
    def __init__(self, cnet_name, mnet_name) -> None:
        """__init__ method for BCDNet
        Args:
            cnet_name (str): 'unet_nc_nblocks' 
            mnet_name (str): 'net-name_hidden-dim'
        """
        super().__init__()
        self.cnet_name = cnet_name
        self.mnet_name = mnet_name
        nc = int(cnet_name.split('_')[1])
        nblocks = int(cnet_name.split('_')[2])
        self.cnet = Cnet(nc=nc, num_blocks=nblocks)
        net_name = mnet_name.split('_')[0]
        hidden_dim = int(mnet_name.split('_')[1])
        self.mnet = Mnet(net_name, hidden_dim)

    def forward(self, y):
        """Forward method for BCDNet

        Args:
            y (tensor): (batch_size, 3, H, W)

        Returns:
            out_M_mean (tensor): (batch_size, 3, 2)
            out_M_var (tensor): (batch_size, 1, 2)
            out_C_mean (tensor): (batch_size, 2, H, W)
        """

        out_M_mean, out_M_var = self.mnet(y) # (batch_size, 3, 2), (batch_size, 1, 2)
        out_C_mean = self.cnet(y) # (batch_size, 2, H, W)

        return out_M_mean, out_M_var, out_C_mean
    