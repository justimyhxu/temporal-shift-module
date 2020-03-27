import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init
from mmcv import Config
import numpy as np
from .resnet_tpn import *

class Identity(nn.Module):

  def __init__(self):
    super(Identity, self).__init__()

  def forward(self, x):
    return x

class CONV(nn.Module):
   def __init__(
       self,
       inplanes, 
       planes,
       kernel_size,
       stride,
       padding,
       bias=False,
       groups=1,
       ):
       super(CONV, self).__init__()
       self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)
       self.bn = nn.BatchNorm3d(planes)
       self.relu = nn.ReLU(inplace=True)

   def forward(self, x):
       out = self.relu(self.bn(self.conv(x)))
       return out


class HEAD(nn.Module):
   def __init__(
       self,
       inplanes,
       planes,
       kernel_size,
       stride,
       padding,
       bias,
       conv_num=1,
       loss_weight=0.5,
       multilabel=False
       ):
       super(HEAD, self).__init__()

       self.convs = nn.ModuleList()
       for i in range(conv_num):
          dim_in = 2 ** (i)
          dim_out = 2 ** (i+1)
          self.convs.append(CONV(inplanes*dim_in, inplanes*dim_out, kernel_size, stride, padding, bias))

       self.loss_weight = loss_weight
       self.dropout = nn.Dropout(p=0.5)
       dim = 2 ** (conv_num)
       self.fc = nn.Linear(inplanes*dim, planes)
       self.new_cls = None
       self.indim = inplanes*dim
       self.outdim = planes
       self.multilabel = multilabel
   def init_weights(self):
       for m in self.modules():
            if isinstance(m, nn.Linear):
               nn.init.normal_(m.weight, 0, 0.01)
               nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)

   def forward(self, x, target=None, fcn_testing=False):
       loss = dict()
       if len(self.convs) > 0:
           for i, conv in enumerate(self.convs):
               x = conv(x)

       if fcn_testing:
           # fcn testing
           if self.new_cls is None:
               # create a conv head          
               self.new_cls = nn.Conv3d(self.indim,self.outdim,1,1,0).cuda()
               self.new_cls.weight.copy_(self.fc.weight.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))
               self.new_cls.bias.copy_(self.fc.bias)
               self.fc = None
           # predict the calss map
           x = self.new_cls(x)
           return x
           
       x = F.adaptive_avg_pool3d(x,1).squeeze(-1).squeeze(-1).squeeze(-1)
       x = self.dropout(x)
       x = self.fc(x)

       if target is None:
           return x
       if self.multilabel:
           loss['loss_aux'] = self.loss_weight * binary_weighted_multilabel_binary_cross_entropy(x, target, target.new_ones(target.shape))
       else:
           loss['loss_aux'] = self.loss_weight * F.cross_entropy(x, target)
       return loss


class LCONV(nn.Module):
    def __init__(self,
                 inplanes, 
                 planes, 
                 kernel_size, 
                 stride, 
                 padding, 
                 bias=False, 
                 groups=1, 
                 norm=False, 
                 activation=False,
                 downsample_type='avg',
                 downsample_position='before',
                 downsample_scale=8,
                 outconv=False,
                 ):
        super(LCONV, self).__init__()

        self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)
        self.norm = nn.BatchNorm3d(planes) if norm else None
        self.relu = nn.ReLU(inplace=True) if activation else None
   
        assert(downsample_type in ['avg','max'])
        assert(downsample_position in ['before','after'])

        self.downsample_position = downsample_position

        self.pool = nn.AvgPool3d((downsample_scale, 1, 1), (downsample_scale, 1, 1), (0, 0, 0), ceil_mode=True, count_include_pad=False) if downsample_type == 'avg' else nn.MaxPool3d((downsample_scale, 1, 1), (downsample_scale, 1, 1), (0, 0, 0), ceil_mode=True)

        self.outconv = nn.Conv3d(planes, planes, (1,1,1), (1,1,1), (0,0,0), bias=False, groups=groups) if outconv else None

    def forward(self, x):
        if self.downsample_position == 'before':
           x = self.pool(x)

           x = self.conv(x)
           if self.norm is not None:
               x = self.norm(x)
           if self.relu is not None:
               x = self.relu(x)
           # 1x1 conv
           if self.outconv is not None:
               x = self.outconv(x)

        elif self.downsample_position == 'after':
           x = self.conv(x)
           if self.norm is not None:
               x = self.norm(x)
           if self.relu is not None:
               x = self.relu(x)
           x = self.pool(x)
           # 1x1 conv
           if self.outconv is not None:
               x = self.outconv(x)

        return x


class UPOP(nn.Module):
    def __init__(self,
                 inplanes,
                 planes,
                 groups=1,
                 norm=False,
                 activation=False,
                 upscale=(2,1,1),
                 upconv=False,
                 pre_act=False,
                ):
        super(UPOP, self).__init__()
        self.conv = nn.Conv3d(inplanes, planes, 1, 1, 0, bias=False, groups=groups) if upconv else None
        if not upconv:
           norm = False
           activation = False
        self.pre_act = pre_act
        if self.pre_act:
            self.norm = nn.BatchNorm3d(inplanes) if norm else None
        else:
            self.norm = nn.BatchNorm3d(planes) if norm else None
        self.relu = nn.ReLU(inplace=True) if activation else None

        #self.upop = nn.Upsample(scale_factor=upscale, mode='nearest')
        self.scale = upscale
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale, mode='nearest')
        if self.pre_act:
            if self.norm is not None:
                x = self.norm(x)
            if self.relu is not None:
                x = self.relu(x)
            if self.conv is not None:
                x = self.conv(x)
        else:
            if self.conv is not None:
                x = self.conv(x)
            if self.norm is not None:
                x = self.norm(x)
            if self.relu is not None:
                x = self.relu(x)
        return x


class FCONV(nn.Module):
    def __init__(self,
                inplanes,
                planes, 
                kernel_size,
                stride,
                padding,
                bias=False,
                groups=1,
                norm=False,
                activation=False,
                pre_act=False,
                ):
        super(FCONV, self).__init__()

        self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)
        self.pre_act = pre_act
        if self.pre_act:
            self.norm = nn.BatchNorm3d(inplanes) if norm else None
        else:
            self.norm = nn.BatchNorm3d(planes) if norm else None
        self.relu = nn.ReLU(inplace=True) if activation else None

    def forward(self, x):
        if self.pre_act:
            if self.norm is not None:
               x = self.norm(x)
            if self.relu is not None:
               x = self.relu(x)
            x = self.conv(x)
           
        else:
            x = self.conv(x)
            if self.norm is not None:
               x = self.norm(x)
            if self.relu is not None:
               x = self.relu(x)
        return x


class DOWNOP(nn.Module):
    def __init__(self,
                 inplanes, 
                 planes, 
                 kernel_size, 
                 stride, 
                 padding, 
                 bias=False, 
                 groups=1, 
                 norm=False, 
                 activation=False,
                 downsample_type='avg',
                 downsample_position='before',
                 downsample_scale=(1,2,2),
                 outconv=False,
                 pre_act=False,
                 ):
        super(DOWNOP, self).__init__()

        self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)
        if pre_act:
            self.norm = nn.BatchNorm3d(inplanes) if norm else None
        else:
            self.norm = nn.BatchNorm3d(planes) if norm else None
  
        self.relu = nn.ReLU(inplace=True) if activation else None
        self.pre_act = pre_act
        assert(downsample_type in ['avg','max','none','sampler'])
        assert(downsample_position in ['before','after'])

        self.downsample_position = downsample_position

        self.pool = nn.AvgPool3d(downsample_scale, downsample_scale, (0, 0, 0), ceil_mode=True, count_include_pad=False) if downsample_type == 'avg' else nn.MaxPool3d(downsample_scale, downsample_scale, (0, 0, 0), ceil_mode=True)


        self.outconv = nn.Conv3d(planes, planes, (1,1,1), (1,1,1), (0,0,0), bias=False, groups=groups) if outconv else None

        if downsample_type == 'sampler':
           self.pool = None
           self.outconv = None
           self.conv = None
           self.norm = None
           self.relu = None
           self.tscale = int(downsample_scale[0])
        else:
           self.tscale = None

    def forward(self, x):
        if self.tscale is not None:
           return x[:,:,0::self.tscale,:,:]
        if self.downsample_position == 'before':
           x = self.pool(x)

        if self.pre_act:
           if self.norm is not None:
               x = self.norm(x)
           if self.relu is not None:
               x = self.relu(x)
           x = self.conv(x)

        else:
           x = self.conv(x)
           if self.norm is not None:
               x = self.norm(x)
           if self.relu is not None:
               x = self.relu(x)

        if self.downsample_position == 'after':
           x = self.pool(x)

        if self.outconv is not None:
           x = self.outconv(x)

        return x

class ADAPOOL(nn.Module):
    def __init__(self, 
                 in_channels=[256,512,1024,2048],
                 mid_channels=[512,512,512,512],
                 out_channels=2048,
                 sampler='ds',
                 ds_scales=[(8,8,8),(4,4,4),(2,2,2),(1,1,1)],
                 kernel_size=(1,3,3),
                 stride=(1,1,1),
                 padding=(0,1,1),
                 bias=False,
                 groups=1,
                 norm=True,
                 activation=True,
                 ds_type='avg',
                 ds_position='before',
                 fusion_type='sum',
                 pre_act=False,
                 fz_groups=1,
                 no_fz_conv=False,
                 ):
        super(ADAPOOL, self).__init__()
        self.fusion_type = fusion_type 
        assert(fusion_type in ['sum','concat','max'])
        assert(sampler in ['ds', 'us'])
        self.fusion_type = fusion_type

        self.ops = nn.ModuleList()
        num_ins = len(in_channels)
        for i in range(num_ins):
            if sampler == 'ds':
                op = DOWNOP(in_channels[i], mid_channels[i], kernel_size, stride, padding, bias, groups, norm, activation, ds_type, ds_position, ds_scales[i], False, pre_act)
            else:
                op = UPOP(in_channels[i], mid_channels[i], groups, norm, activation, ds_scales[i], True, pre_act)
            self.ops.append(op)

        if fusion_type == 'sum' or fusion_type == 'max': 
            in_dims = mid_channels[0]
        else:
            in_dims = np.sum(mid_channels)

        if pre_act:
            self.fusion_conv = nn.Sequential(
                nn.BatchNorm3d(in_dims),
                nn.ReLU(inplace=True),
                nn.Conv3d(in_dims, out_channels, 1, 1, 0, bias=False, groups=fz_groups)
                )
        else:
            self.fusion_conv = nn.Sequential(
                nn.Conv3d(in_dims, out_channels, 1, 1, 0, bias=False, groups=fz_groups),
                nn.BatchNorm3d(out_channels),
                nn.ReLU(inplace=True)
                )
        if no_fz_conv:
            self.fusion_conv = None
   
    def forward(self, inputs):
        out = [self.ops[i](feature) for i, feature in enumerate(inputs)]
        if self.fusion_type == 'sum':
           out = torch.sum(torch.stack(out),0)
        elif self.fusion_type == 'max':
           out = torch.max(torch.stack(out),0)[0]
        elif self.fusion_type == 'concat':
           out = torch.cat(out, 1)
        if self.fusion_conv is not None:
           out = self.fusion_conv(out)
        return out     
 

class SDS(nn.Module):
   def __init__(
       self,
       inplanes=[256,512,1024,2048],
       planes=2048,
       kernel_size=(1,3,3),
       stride=(1,2,2),
       padding=(0,1,1),
       bias=False,
       groups=1, 
       slim=False,
       save_mode=False,
       ):
       super(SDS, self).__init__()

       self.sds = nn.ModuleList()
       self.save_mode = save_mode

       slim=slim
       if False:
           for i, dim in enumerate(inplanes):
               ds_factor = planes // dim
               if dim == planes:
                   current_op = Identity()
               else:
                   current_op = nn.Sequential(
                       nn.MaxPool3d((1, ds_factor, ds_factor), (1, ds_factor, ds_factor), (0, 0, 0), ceil_mode=True), 
                       nn.Conv3d(dim, planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups)
                   )
               self.sds.append(current_op)

       elif slim:
           if save_mode:
               self.sds = nn.ModuleList()
               op=nn.ModuleList()
               curr_op = nn.Sequential(
                   nn.MaxPool3d((1, 2, 2), (1, 2, 2), (0, 0, 0), ceil_mode=True),
                   nn.Conv3d(1024, planes, 1, 1, 0, bias=bias, groups=groups)
               )
               op.append(curr_op)
               self.sds.append(op)

               op=nn.ModuleList()
               op.append(nn.Conv3d(2048, planes, 1, 1, 0, bias=bias, groups=groups))
               self.sds.append(op)
           else:
               # slim version
               self.sds = nn.ModuleList()
               op=nn.ModuleList()
               op.append(CONV(inplanes[0], planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups))
               self.sds.append(op)
        
               op=nn.ModuleList()
               op.append(CONV(inplanes[1], planes, kernel_size=1, stride=1, padding=0, bias=bias, groups=32))
               self.sds.append(op)
        

       else:
           for i, dim in enumerate(inplanes):
               op = nn.ModuleList()           
               ds_factor = planes // dim
               ds_num  = int(np.log2(ds_factor))
               if ds_num < 1:
                  op = Identity()
               else:
                  for dsi in range(ds_num):
                      in_factor = 2 ** dsi
                      out_factor = 2 ** (dsi+1)
                      op.append(CONV(dim*in_factor, dim*out_factor, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, groups=groups))
               self.sds.append(op)

   def forward(self, inputs):
       out = []
       for i, feature in enumerate(inputs):
            if isinstance(self.sds[i], nn.ModuleList):
                out_ = inputs[i]
                for III, op in enumerate(self.sds[i]):
                    out_ = op(out_)
                out.append(out_) 
            else:
                out.append(self.sds[i](inputs[i]))
       return out
                  

class FZOP(nn.Module):
   def __init__(
       self,
       inplanes=2048,
       planes=1024,
       kernel_size=1,
       stride=1,
       padding=0,
       bias=False,
       groups=1,
       fz_type='add',
       ):
       super(FZOP, self).__init__()
       assert(fz_type in ['add','concat'])
       self.fz_type = fz_type
       if fz_type == 'concat':
            self.conv = nn.Conv3d(inplanes, planes, kernel_size, stride, padding, bias=bias, groups=groups)

   def forward(self, x, y):
       if self.fz_type == 'add':
            return x+y
       if self.fz_type == 'concat':
            return self.conv(torch.cat([x,y],1))



class TPN(nn.Module):

    def __init__(self,
                 num_seg=num_seg,
                 in_channels=[256,512,1024,2048],
                 out_channels=256,
                 return_original=True,
                 fz_conv=True,
                 swap=False,
                 TEST_MODE=False,
                 concat_fusion=False,
                 two_pyramid=False,
                 two_pyramid_fusion=None,
                 fz_pyramid_groups=1,
                 dynamic_sampler_rate=None,
                 fz_config=None,
                 res_config=None,
                 sds_config=None,
                 lconv_config=None,
                 upop_config=None,
                 fconv_config=None,
                 downop_config=None,
                 adapool_config=None,
                 aux_head_config=None,
                 aux_head_return=True,
                 ):
        super(TPN, self).__init__()
        self.num_seg = num_seg
        assert isinstance(in_channels, list)
        assert isinstance(out_channels, int)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.return_original = return_original
        self.aux_head_return = aux_head_return

        self.res_config = Config(res_config) if isinstance(res_config, dict) else res_config 
        sds_config = Config(sds_config) if isinstance(sds_config, dict) else sds_config 
        lconv_config = Config(lconv_config) if isinstance(lconv_config, dict) else lconv_config 
        fconv_config = Config(fconv_config) if isinstance(fconv_config, dict) else fconv_config
        upop_config = Config(upop_config)  if isinstance(upop_config, dict) else upop_config
        downop_config = Config(downop_config) if isinstance(downop_config, dict) else downop_config 
        aux_head_config = Config(aux_head_config) if isinstance(aux_head_config ,dict) else aux_head_config 
        adapool_config = Config(adapool_config) if isinstance(adapool_config, dict) else adapool_config
        fz_config = Config(fz_config) if isinstance(fz_config, dict) else fz_config

        self.TEST_MODE = TEST_MODE

        self.concat_fusion = concat_fusion
        self.lconvs = nn.ModuleList()
        self.fconvs = nn.ModuleList()
        self.upops= nn.ModuleList()
        self.upfzops = nn.ModuleList()
        self.downops = nn.ModuleList()
        self.downfzops = nn.ModuleList()
        self.adapool_ops = ADAPOOL(**adapool_config) if adapool_config is not None else None 
        self.sds = SDS(**sds_config) if sds_config is not None else None
        self.swap = swap

        self.tdfzops= nn.ModuleList()
        self.bufzops= nn.ModuleList()
        self.dynamic_sampler_rate = dynamic_sampler_rate
        if dynamic_sampler_rate is not None:
            assert(self.num_ins == len(dynamic_sampler_rate))


        for i in range(0, self.num_ins, 1):
            if dynamic_sampler_rate is None:
                inplanes = in_channels[i] if sds_config is None else sds_config.planes
            else:
                inplanes = in_channels[-1]

            planes = out_channels
            
            if lconv_config is not None:
                # overwrite the lconv_config
                lconv_config.param.downsample_scale = lconv_config.scales[i]
                lconv_config.param.inplanes = inplanes
                lconv_config.param.planes = planes
                lconv = LCONV(**lconv_config.param)
                self.lconvs.append(lconv)
          
            if fconv_config is not None:
                # overwrite the fconv_config
                fconv_config.param.inplanes = planes
                fconv_config.param.planes = planes
                fconv = FCONV(**fconv_config.param)
                self.fconvs.append(fconv)
                 
            if i < self.num_ins - 1:
                if upop_config is not None:
                    # overwrite the upop_config
                    upop_config.param.inplanes = planes
                    upop_config.param.planes = planes
                    upop_config.param.upscale = upop_config.scales
                    upop = UPOP(**upop_config.param)
                    self.upops.append(upop)


                if downop_config is not None: 
                    # overwrite the downop_config
                    downop_config.param.inplanes = planes
                    downop_config.param.planes = planes
                    downop_config.param.downsample_scale = downop_config.scales
                    downop = DOWNOP(**downop_config.param)
                    self.downops.append(downop)

                if fz_config is None:
                    tdfz_op = FZOP(-1,-1,-1,-1,-1,False,1,'add')
                    bufz_op = FZOP(-1,-1,-1,-1,-1,False,1,'add')
                else:
                    tdfz_op = FZOP(**fz_config)
                    bufz_op = FZOP(**fz_config)

                self.tdfzops.append(tdfz_op)
                self.bufzops.append(bufz_op)
                    


        if self.res_config is not None and self.res_config.activation:
            assert(self.res_config.position in ['before_fconv', 'after_fconv', 'before_ada'])
            self.relu = nn.ReLU(inplace=True)

        if len(self.lconvs) == 0:
            self.lconvs = None
        if len(self.fconvs) == 0:
            self.fconvs = None
        if len(self.upops) == 0:
            self.upops = None
        if len(self.downops) == 0:
            self.downops = None
        if len(self.tdfzops) == 0:
            self.tdfzops = None
        if len(self.bufzops) == 0:
            self.bufzops = None
        if not concat_fusion:
            self.upfzops = None
            self.downfzops = None

        if self.lconvs is not None:
            if adapool_config is not None:
                out_dims = adapool_config.out_channels
            else:
                out_dims = out_channels
            if fz_conv:
                self.fusion_conv = nn.Sequential(
                    nn.Conv3d(out_dims, in_channels[-1], 1, 1, 0, bias=False),
                    nn.BatchNorm3d(in_channels[-1]),
                    nn.ReLU(inplace=True)
                    )
            else:
                self.fusion_conv = None

        # Two pyramid
        self.two_pyramid_fusion = two_pyramid_fusion
        if self.adapool_ops is not None and two_pyramid:
            self.adapool_ops2 = ADAPOOL(**adapool_config) if adapool_config is not None else None

            out_dims = adapool_config.out_channels
            if two_pyramid_fusion is None:
                indim = out_dims*2
            elif two_pyramid_fusion == 'concat':
                indim = out_dims*2
            elif two_pyramid_fusion == 'sum':
                indim = out_dims
            elif two_pyramid_fusion == 'max':
                indim = out_dims
            
            self.fz_pyramid = nn.Sequential(
                    nn.Conv3d(indim, 2048, 1, 1, 0, bias=False, groups=fz_pyramid_groups),
                    nn.BatchNorm3d(2048),
                    nn.ReLU(inplace=True)
                    )
        else: 
            self.adapool_ops2 = None
            self.fz_pyramid = None

 

        if aux_head_config is not None:
           # overwrite aux_head_config
           self.aux_position = aux_head_config.position 
           #assert(self.aux_position in ['ori','lconv','bottomup','fconv','topdown'])
           assert(self.aux_position in ['ori','sds'])
               
           aux_head_config.param.inplanes = self.in_channels[-2] if aux_head_config.position == 'ori' else sds_config.planes
           aux_head_config.param.planes = 400 if aux_head_config.param.planes < 0 else aux_head_config.param.planes

           self.aux_head = HEAD(**aux_head_config.param)
        else:
           self.aux_head = None
           self.aux_position = 'no_aux_head'


    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.fill_(0)
       
        if self.fusion_conv is not None and self.return_original:
            self.fusion_conv[1].weight.data.fill_(0)
            self.fusion_conv[1].bias.data.fill_(0)
        if self.aux_head is not None:
            self.aux_head.init_weights()

    def forward(self, inputs, target=None):

        def trans(tensor):
            bs, c, h, w = tensor.shape
            tensor = tensor.reshape(int(bs//self.num_seg), self.num_seg, c, h, w)
            tensor = tensor.permute(0,2,1,3,4)
            return tensor

        inputs = [trans(x) for x in inputs]

        if self.dynamic_sampler_rate is None:
             assert len(inputs) == len(self.in_channels)

        loss = None

        # aux loss
        if self.aux_position == 'ori':
            loss = self.aux_head(inputs[-2], target) if self.aux_head is not None else None
            if self.aux_head_return:
                return inputs[-1], loss

        # sds
        if self.sds is not None:
            outs = self.sds(inputs)
        else:
            outs = inputs

        # aux loss
        if self.aux_position == 'sds':
            loss = self.aux_head(outs[-2], target) if self.aux_head is not None else None

        # dynamic sampling
        if self.dynamic_sampler_rate is not None:
            #outs = [inputs[-1][:,:,0::int(rate),:,:] for i, rate in enumerate(self.dynamic_sampler_rate)] # dynamic sampling
            outs = [F.max_pool3d(inputs[-1], (rate,1,1)) for i, rate in enumerate(self.dynamic_sampler_rate)] # maxpool to downsample

        # lconv
        if self.lconvs is not None:
            outs = [lconv(outs[i]) for i, lconv in enumerate(self.lconvs)]

        # get the lconv outs for pyramid 2
        if self.adapool_ops2 is not None:
            lconv_outs = outs

        # build top-down path
        if self.swap:
            # down operation i.e. top-down path
            if self.downops is not None:
                for i in range(self.num_ins-1, 0, -1):
                     if self.tdfzops is None:
                         outs[i - 1] = outs[i - 1] + self.downops[i-1](outs[i])  
                     else:
                         outs[i - 1] = self.tdfzops[i-1](outs[i - 1], self.downops[i-1](outs[i]))
                     
            
        else:
            # up operation i.e. top-down path
            if self.upops is not None:
                for i in range(self.num_ins-1, 0, -1):
                     if self.tdfzops is None:
                         outs[i - 1] = outs[i - 1] + self.upops[i-1](outs[i])
                     else:
                         outs[i - 1] = self.tdfzops[i-1](outs[i - 1], self.upops[i-1](outs[i]))
                

        # fconv
        if self.fconvs is not None:
            outs = [self.fconvs[i](outs[i]) for i in range(self.num_ins)]

        # get pyramid 1 outs
        if self.adapool_ops2 is not None:
            assert(self.fconvs is None)
            topdownouts = self.adapool_ops2(outs)
            outs = lconv_outs


        # build bottom-up path
        if self.swap:
            # up operation i.e. top-down path
            if self.upops is not None:
                for i in range(0, self.num_ins-1, 1):
                     if self.bufzops is None:
                         outs[i + 1] = outs[i + 1] + self.upops[i](outs[i])
                     else:
                         outs[i + 1] = self.bufzops[i](outs[i + 1], self.upops[i](outs[i]))
                        
        else:
            # down operation i.e. bottom-up path
            if self.downops is not None:
                for i in range(0, self.num_ins-1, 1):
                     if self.bufzops is None:
                         outs[i+1] = outs[i+1] + self.downops[i](outs[i])  
                     else:
                         outs[i+1] = self.bufzops[i](outs[i+1], self.downops[i](outs[i]))
    
        # adapool operation
        if self.adapool_ops is not None:
            outs = self.adapool_ops(outs)
            if self.fusion_conv is not None:
                 outs = self.fusion_conv(outs)
        else:
            if self.fusion_conv is not None:
                 outs = self.fusion_conv(outs[-1])
            else:
                 outs = outs[-1]

        # fuse two pyramid outs
        if self.fz_pyramid is not None:
            if self.two_pyramid_fusion is None:
                outs = self.fz_pyramid(torch.cat([topdownouts, outs],1))
            elif self.two_pyramid_fusion == 'max':
                outs = self.fz_pyramid(torch.max(torch.stack([topdownouts, outs]) ,0)[0])
            elif self.two_pyramid_fusion == 'sum':
                outs = self.fz_pyramid(torch.sum(torch.stack([topdownouts, outs]) ,0))

        return outs, loss


def main():
   res2 = torch.FloatTensor(8,256,8,56,56).cuda()
   res3 = torch.FloatTensor(8,512,8,28,28).cuda()
   res4 = torch.FloatTensor(8,1024,8,14,14).cuda()
   res5 = torch.FloatTensor(8,2048,8,7,7).cuda()
   #feature = tuple([res2, res3, res4, res5])
   feature = tuple([res4, res5])
   model = TPN(
        in_channels=[1024,2048],
        out_channels=1024,
        return_original=False,
        fz_conv=False,
        dynamic_sampler_rate=None,
        swap=False,
        two_pyramid=True,
        sds_config=dict(inplanes=[1024,2048],
           planes=2048,
           kernel_size=(1,1,1),
           stride=(1,1,1),
           padding=(0,0,0),
           bias=False,
           save_mode=True,
           groups=32),
        lconv_config=dict(scales=(8,8),
           param=dict(inplanes=-1,
           planes=-1,
           kernel_size=(3,1,1),
           stride=(1,1,1),
           padding=(1,0,0),
           bias=False,
           groups=32,
           norm=False,
           activation=False,
           downsample_type='max',
           downsample_position='after',
           downsample_scale=-1,
           outconv=False)),
        upop_config=dict(scales=(1,1,1),
           param=dict(inplanes=-1,
           planes=-1,
           groups=1,
           norm=False,
           activation=False,
           upscale=-1,
           upconv=False)),
        downop_config=dict(scales=(1,1,1),
           param=dict(inplanes=-1,
           planes=-1,
           kernel_size=(3,1,1),
           stride=(1,1,1),
           padding=(1,0,0),
           bias=False,
           groups=1,
           norm=False,
           activation=False,
           downsample_type='max',
           downsample_position='after',
           downsample_scale=-1,
           outconv=False)),
        adapool_config=dict(
           in_channels=[1024,1024],
           mid_channels=[1024,1024],
           out_channels=2048,
           sampler='ds',
           #ds_scales=[(8,8,8),(4,4,4),(2,2,2),(1,1,1)],
           ds_scales=[(1,1,1),(1,1,1)],
           ds_type='max',
           ds_position='before',
           kernel_size=(1,1,1),
           stride=(1,1,1),
           padding=(0,0,0),
           bias=False,
           groups=32,
           norm=False,
           activation=False,
           fusion_type='concat'),
        aux_head_config=dict(
           position='ori',
           param=dict(
           inplanes=-1,
           planes=-1,
           kernel_size=(1,3,3),
           stride=(1,2,2),
           padding=(0,1,1),
           bias=False,
           conv_num=1,
           loss_weight=0.5
           )),
       aux_head_return=False,
                 ).cuda()
   print(model)
   out, prob = model(feature)
   print(out.shape)
   #print(prob.shape)

if __name__=='__main__':
   main()
