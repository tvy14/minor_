#Attention-based method from : https://github.com/Floating-LY/HARMONY1/blob/main/model/harmony.py#L43
# Construct Real-NVP 
class RealNVP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_flows,configs):
        super(RealNVP, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(n_flows):
            # Construct CouplingLayer
            mask = torch.arange(input_dim) % 2
            if input_dim % 2 == 1:
                mask[:input_dim // 2 + 1] = 1
                mask[input_dim // 2 + 1:] = 0
            mask = mask.float().to('cuda:0')
            coupling_layer = CouplingLayer(input_dim, hidden_dim, mask)
            self.layers.append(coupling_layer)

            # inversr mask after CouplingLayer
            mask = 1 - mask
            coupling_layer = CouplingLayer(input_dim, hidden_dim, mask)
            self.layers.append(coupling_layer)
