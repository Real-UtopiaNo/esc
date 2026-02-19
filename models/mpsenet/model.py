import torch
import torch.nn as nn

from .utils import get_padding, get_padding_2d

class LearnableSigmoid2d(nn.Module):
    def __init__(self, in_features, beta=1):
        super().__init__()
        self.beta = beta
        self.slope = nn.Parameter(torch.ones(in_features, 1))
        self.slope.requiresGrad = True

    def forward(self, x):
        return self.beta * torch.sigmoid(self.slope * x)

class DenseBlock(nn.Module):
    def __init__(self, dense_channel, kernel_size=(2, 3), depth=4):
        super(DenseBlock, self).__init__()
        self.depth = depth
        self.dense_block = nn.ModuleList([])
        for i in range(depth):
            dilation = 2 ** i
            pad_h, pad_w = get_padding_2d(kernel_size, (dilation, 1))
            dense_conv = nn.Sequential(
                nn.ConstantPad2d((pad_w, pad_w, pad_h, pad_h), value=0.0),
                nn.Conv2d(dense_channel*(i+1), dense_channel, kernel_size, dilation=(dilation, 1)),
                nn.InstanceNorm2d(dense_channel, affine=True),
                nn.PReLU(dense_channel)
            )
            self.dense_block.append(dense_conv)

    def forward(self, x):
        skip = x
        for i in range(self.depth):
            x = self.dense_block[i](skip)
            skip = torch.cat([x, skip], dim=1)
        return x

class DenseEncoder(nn.Module):
    def __init__(self, dense_channel, in_channel):
        super(DenseEncoder, self).__init__()
        self.dense_conv_1 = nn.Sequential(
            nn.Conv2d(in_channel, dense_channel, (1, 1)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

        self.dense_block = DenseBlock(dense_channel, kernel_size=(3, 3), depth=4) #kernel size 2,3 for GRU based model

        self.dense_conv_2 = nn.Sequential(
            nn.Conv2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel))

    def forward(self, x):
        x = self.dense_conv_1(x)  # [b, 64, T, F]
        x = self.dense_block(x)   # [b, 64, T, F]
        x = self.dense_conv_2(x)  # [b, 64, T, F//2]
        return x

class MaskDecoder(nn.Module):
    def __init__(self, dense_channel, n_fft, beta, out_channel=1):
        super(MaskDecoder, self).__init__()
        self.dense_block = DenseBlock(dense_channel, kernel_size=(3, 3), depth=4)
        self.mask_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.Conv2d(dense_channel, out_channel, (1, 1)),
            nn.InstanceNorm2d(out_channel, affine=True),
            nn.PReLU(out_channel),
            nn.Conv2d(out_channel, out_channel, (1, 1))
        )
        self.lsigmoid = LearnableSigmoid2d(n_fft//2+1, beta=beta)

    def forward(self, x):
        x = self.dense_block(x)
        x = self.mask_conv(x)
        x = x.permute(0, 3, 2, 1).squeeze(-1)
        x = self.lsigmoid(x).permute(0, 2, 1).unsqueeze(1)
        return x

class PhaseDecoder(nn.Module):
    def __init__(self, dense_channel, out_channel=1):
        super(PhaseDecoder, self).__init__()
        self.dense_block = DenseBlock(dense_channel, kernel_size=(3, 3), depth=4)
        self.phase_conv = nn.Sequential(
            nn.ConvTranspose2d(dense_channel, dense_channel, (1, 3), (1, 2)),
            nn.InstanceNorm2d(dense_channel, affine=True),
            nn.PReLU(dense_channel)
        )
        self.phase_conv_r = nn.Conv2d(dense_channel, out_channel, (1, 1))
        self.phase_conv_i = nn.Conv2d(dense_channel, out_channel, (1, 1))

    def forward(self, x):
        x = self.dense_block(x)
        x = self.phase_conv(x)
        x_r = self.phase_conv_r(x)
        x_i = self.phase_conv_i(x)
        x = torch.atan2(x_i, x_r)
        return x

class TSConformerBlock(nn.Module):
    def __init__(self, dense_channel):
        super(TSConformerBlock, self).__init__()
        self.time_conformer = ConformerBlock(dim=dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)
        self.freq_conformer = ConformerBlock(dim=dense_channel,  n_head=4, ccm_kernel_size=31, 
                                             ffm_dropout=0.2, attn_dropout=0.2)
    def forward(self, x):
        b, c, t, f = x.size()
        x = x.permute(0, 3, 2, 1).contiguous().view(b*f, t, c)
        x = self.time_conformer(x) + x
        x = x.view(b, f, t, c).permute(0, 2, 1, 3).contiguous().view(b*t, f, c)
        x = self.freq_conformer(x) + x
        x = x.view(b, t, f, c).permute(0, 3, 1, 2)
        return x 

class ConformerBlock(nn.Module):
    def __init__(self, dim, n_head=8, ffm_mult=4, ccm_expansion_factor=2, ccm_kernel_size=31, 
                 ffm_dropout=0., attn_dropout=0., ccm_dropout=0.):
        super(ConformerBlock, self).__init__()
        self.ffm1 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.attn = AttentionModule(dim, n_head, dropout=attn_dropout)
        self.ccm = ConformerConvModule(dim, ccm_expansion_factor, ccm_kernel_size, dropout=ccm_dropout)
        self.ffm2 = FeedForwardModule(dim, ffm_mult, dropout=ffm_dropout)
        self.post_norm = nn.LayerNorm(dim)
        self.scale = torch.tensor(0.5)

    def forward(self, x):
        x = x + self.scale * self.ffm1(x)
        x = x + self.attn(x)
        x = x + self.ccm(x)
        x = x + self.scale * self.ffm2(x)
        x = self.post_norm(x)
        return x

class FeedForwardModule(nn.Module):
    def __init__(self, dim, mult=4, dropout=0):
        super(FeedForwardModule, self).__init__()
        self.ffm = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * mult),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.ffm(x)

class AttentionModule(nn.Module):
    def __init__(self, dim, n_head=8, dropout=0.):
        super(AttentionModule, self).__init__()
        self.attn = nn.MultiheadAttention(dim, n_head, dropout=dropout)
        self.layernorm = nn.LayerNorm(dim)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        x = self.layernorm(x)
        x, _ = self.attn(x, x, x, 
                         attn_mask=attn_mask,
                         key_padding_mask=key_padding_mask)
        return x
    
class ConformerConvModule(nn.Module):
    def __init__(self, dim, expansion_factor=2, kernel_size=31, dropout=0.):
        super(ConformerConvModule, self).__init__()
        inner_dim = dim * expansion_factor
        self.norm = nn.LayerNorm(dim)
        self.conv_in = nn.Conv1d(dim, inner_dim * 2, 1)
        self.glu = nn.GLU(dim=1)        
        self.pad = nn.ConstantPad1d(15, 0.0)
        self.depthwise_conv = nn.Conv1d(
            inner_dim, inner_dim, kernel_size=kernel_size,
            groups=inner_dim
        )
        self.bn = nn.BatchNorm1d(inner_dim)
        self.act = nn.SiLU()        
        self.conv_out = nn.Conv1d(inner_dim, dim, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.norm(x)
        x = x.transpose(1, 2)
        x = self.conv_in(x)
        x = self.glu(x) 
        x = self.pad(x)
        x = self.depthwise_conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.conv_out(x)
        x = x.transpose(1, 2)
        x = self.drop(x)
        return x

class MPNet(nn.Module):
    def __init__(self, dense_channel=64, n_fft=400, beta=2.0, num_tsblocks=4):
        super(MPNet, self).__init__()
        self.num_tscblocks = num_tsblocks
        self.dense_encoder = DenseEncoder(dense_channel, in_channel=2)

        self.TSConformer = nn.ModuleList([])
        for i in range(num_tsblocks):
            self.TSConformer.append(TSConformerBlock(dense_channel=64))
        
        self.mask_decoder = MaskDecoder(dense_channel, n_fft, beta, out_channel=1)
        self.phase_decoder = PhaseDecoder(dense_channel, out_channel=1)

    def forward(self, x): # [B, F, T]

        x = self.dense_encoder(x)

        for i in range(self.num_tscblocks):
            x = self.TSConformer[i](x)
        
        denoised_amp = self.mask_decoder(x)
        denoised_pha = self.phase_decoder(x)

        return denoised_amp, denoised_pha