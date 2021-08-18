from collections import OrderedDict

import numpy as np
import torch
from torch import nn
from torchmeta.modules import MetaModule

from modules import init_weights_normal, BatchLinear


class MetaSDF(nn.Module):
    """MAML"""
    def __init__(self, hypo_net, hypo_loss, init_lr=1e-1, num_meta_steps=3, first_order=False, lr_type='static'):
        super().__init__()

        self.hypo_net = hypo_net
        self.hypo_loss = hypo_loss
        self.num_meta_steps = num_meta_steps
        self.first_order = first_order
        self.lr_type = lr_type

        if self.lr_type == 'static':
            self.register_buffer('lr', torch.tensor([init_lr]))
        elif self.lr_type == 'global':
            self.lr = nn.Parameter(torch.tensor([init_lr]))
        elif self.lr_type == 'per_step':
            self.lr = nn.ParameterList([nn.Parameter(torch.tensor([init_lr])) for _ in range(num_meta_steps)])
        elif self.lr_type == 'per_parameter':
            self.lr = nn.ParameterList([nn.Parameter(torch.tensor([init_lr])) for _ in hypo_net.parameters()])
        elif self.lr_type == 'per_parameter_metasgd':
            self.lr = nn.ParameterList([
                nn.Parameter(torch.ones(param.shape) * init_lr)
                for param in hypo_net.parameters()
            ])
        elif self.lr_type == 'per_parameter_per_step':
            self.lr = nn.ModuleList([
                nn.ParameterList([nn.Parameter(torch.ones(param.shape) * init_lr) for _ in range(num_meta_steps)])
                for param in hypo_net.parameters()
            ])

        self.hyper_sigma = nn.Parameter(torch.ones(2))
        self.hypo_sigma = nn.Parameter(torch.ones(2))

    def generate_params(self, context, num_meta_steps=None, return_preds=False):
        meta_batch_size = context['coords'].shape[0]
        num_meta_steps = num_meta_steps if num_meta_steps is not None else self.num_meta_steps

        with torch.enable_grad():
            context_params = OrderedDict()
            for name, param in self.hypo_net.meta_named_parameters():  # ~hyper_net parameters
                # xMeta_batch_size first dim
                context_params[name] = param.unsqueeze(0).repeat((meta_batch_size,) + (1,) * len(param.shape))

            inner_preds = []
            for step_num in range(num_meta_steps):
                context['coords'].requires_grad_()
                context_pred_sdf = self.hypo_net.forward(context['coords'], context_params)
                inner_preds.append(context_pred_sdf)

                loss = self.hypo_loss(context_pred_sdf, context['real_sdf'], sigma=self.hypo_sigma)

                grads = torch.autograd.grad(
                    loss,
                    context_params.values(),
                    allow_unused=False,
                    create_graph=(not self.first_order) or (step_num == num_meta_steps - 1),
                )

                for param_num, ((name, param), grad) in enumerate(zip(context_params.items(), grads)):
                    if self.lr_type in ['static', 'global']:
                        lr = self.lr
                    elif self.lr_type in ['per_step']:
                        lr = self.lr[step_num]
                    elif self.lr_type in ['per_parameter', 'per_parameter_metasgd']:
                        lr = self.lr[param_num]
                    elif self.lr_type in ['per_parameter_per_step']:
                        lr = self.lr[param_num][step_num] if num_meta_steps <= self.num_meta_steps else 1e-2
                    else:
                        raise NotImplementedError
                    context_params[name] = param - lr * grad
                    # TODO: Add proximal regularization from iMAML
                    # Add meta-regularization

        if return_preds:
            return inner_preds

        return context_params

    def forward(self, coords, context_params):
        output = self.hypo_net(coords, params=context_params)
        return output


class HyperNetwork(nn.Module):
    def __init__(self, in_features, hidden_layers, hidden_features, hypo_net):
        super().__init__()

        self.names = []
        self.nets = nn.ModuleList()
        self.param_shapes = []
        hypo_parameters = hypo_net.meta_named_parameters()
        for name, param in hypo_parameters:
            self.names.append(name)
            self.param_shapes.append(param.shape)

            hn = FCBlock(
                hidden_features, hidden_layers, in_features,
                out_features=int(torch.prod(torch.tensor(param.shape))),
                outermost_linear=True,
            )
            with torch.no_grad():
                hn.net[-1].weight *= 1e-1
            self.nets.append(hn)

    def forward(self, z):
        params = OrderedDict()
        for name, net, param_shape in zip(self.names, self.nets, self.param_shapes):
            batch_param_shape = (-1,) + param_shape
            params[name] = net(z).reshape(batch_param_shape)

        return params


class FCLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, input_):
        return self.net(input_)


class FCBlock(nn.Module):
    def __init__(self, hidden_ch, num_hidden_layers, in_features, out_features, outermost_linear=False):
        super().__init__()

        self.net = []
        self.net.append(FCLayer(in_features=in_features, out_features=hidden_ch))

        for i in range(num_hidden_layers):
            self.net.append(FCLayer(in_features=hidden_ch, out_features=hidden_ch))

        if outermost_linear:
            self.net.append(nn.Linear(in_features=hidden_ch, out_features=out_features))
        else:
            self.net.append(FCLayer(in_features=hidden_ch, out_features=out_features))

        self.net = nn.Sequential(*self.net)
        self.net.apply(init_weights_normal)

    def forward(self, input_):
        return self.net(input_)


class SDFHyperNetwork(nn.Module):
    """
    Framework for swapping in different types of encoders and modules to use with
    hypernetworks.
    See Hypernetworks_MNIST for examples.
    """

    def __init__(self, encoder, hyper_net, hypo_net):
        super().__init__()
        self.encoder = encoder
        self.hypo_net = hypo_net
        self.hyper_net = hyper_net

    def forward(self, coords, context_params):
        z = self.encoder(context_params['index'])
        batch_size = z.shape[0]
        z = z.reshape(batch_size, -1)
        params = self.hyper_net.forward(z)
        out = self.hypo_net.forward(coords, params)
        return out

    def freeze_hyper_net(self):
        # Freeze hypernetwork for latent code optimization
        for param in self.hyper_net.parameters():
            param.requires_grad = False

    def unfreeze_hyper_net(self):
        # Unfreeze hypernetwork for training
        for param in self.hyper_net.parameters():
            param.requires_grad = True


class AutoDecoder(nn.Module):
    """
    Autodecoder module; takes an idx as input and returns a latent code, z
    """

    def __init__(self, num_instances, latent_dim):
        super().__init__()
        self.latent_codes = nn.Embedding(num_instances, latent_dim)
        torch.nn.init.normal_(self.latent_codes.weight.data, 0.0, 1e-3)

    def forward(self, idx):
        z = self.latent_codes(idx)
        return z


class SineLayer(MetaModule):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = float(omega_0)

        self.is_first = is_first

        self.in_features = in_features
        self.linear = BatchLinear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0
                )

    def forward(self, input_, params):
        intermediate = self.linear(input_, params=self.get_subdict(params, 'linear'))
        return torch.sin(self.omega_0 * intermediate)


class Siren(MetaModule):
    def __init__(
            self, in_features, hidden_features, hidden_layers, out_features,
            outermost_linear=False, first_omega_0=30, hidden_omega_0=30., special_first=True
    ):
        super().__init__()
        self.hidden_omega_0 = hidden_omega_0

        layer = SineLayer

        self.net = [layer(in_features, hidden_features, is_first=special_first, omega_0=first_omega_0)]

        for i in range(hidden_layers):
            self.net.append(layer(hidden_features, hidden_features, is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = BatchLinear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / 30.,
                    np.sqrt(6 / hidden_features) / 30.
                )
            self.net.append(final_linear)
        else:
            self.net.append(layer(hidden_features, out_features, is_first=False, omega_0=hidden_omega_0))

        self.net = nn.ModuleList(self.net)

    def forward(self, coords, params):
        res = coords

        for i, layer in enumerate(self.net):
            res = layer(res, params=self.get_subdict(params, f'net.{i}'))

        return res
