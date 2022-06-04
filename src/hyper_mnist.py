import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import MNISTSDFDataset, next_step, l2_hyper_loss
from meta_modules import AutoDecoder, HyperNetwork, SDFHyperNetwork
from modules import ReLUFC

dtype = torch.float32
torch.set_default_dtype(dtype)
train_dataset = MNISTSDFDataset(split='train', dtype=dtype, side_len=64)
val_dataset = MNISTSDFDataset(split='val', dtype=dtype, side_len=64)

train_dataloader = DataLoader(train_dataset, batch_size=128)
val_dataloader = DataLoader(val_dataset, batch_size=128)

encoder = AutoDecoder(num_instances=len(train_dataset), latent_dim=256)
hypo_net = ReLUFC(in_features=2, out_features=1, num_hidden_layers=2, hidden_features=256)
hyper_net = HyperNetwork(in_features=256, hidden_layers=1, hidden_features=256, hypo_net=hypo_net)
model = SDFHyperNetwork(encoder, hyper_net, hypo_net).cuda()
# model = torch.load('../output/hyper.pth')  # TODO

optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

writer = SummaryWriter()
for epoch in tqdm(range(3000), desc='Epoch'):
    model.train()
    for step, batch_cpu in enumerate(tqdm(train_dataloader, desc='Train')):
        train_loss = next_step(
            model, l2_hyper_loss, train_dataset, epoch, step, batch_cpu,
            get_context_params=lambda batch_gpu: {'index': batch_gpu['index']},
        )

        writer.add_scalar('Loss/train', train_loss, global_step=step + epoch * len(train_dataloader))

        optim.zero_grad()
        train_loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        for step, batch_cpu in enumerate(tqdm(val_dataloader, desc='Valid')):
            valid_loss = next_step(
                model, l2_hyper_loss, val_dataset, epoch, step, batch_cpu,
                get_context_params=lambda batch_gpu: batch_gpu['index'],
            )

            writer.add_scalar('Loss/valid', valid_loss, global_step=step + epoch * len(val_dataloader))
