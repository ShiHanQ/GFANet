from models.GFANet import GFANet


def build_net(netname):

    if netname == 'GFANet':
        net = GFANet()

    return net
