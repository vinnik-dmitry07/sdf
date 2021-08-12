import torch
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import MNISTSDFDataset, next_step
from meta_modules import AutoDecoder, HyperNetwork, SDFHyperNetwork
from modules import ReLUFC

train_dataset = MNISTSDFDataset(split='train', side_len=64)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)

encoder = AutoDecoder(num_instances=len(train_dataset), latent_dim=256)
hyponet = ReLUFC(in_features=2, out_features=1, num_hidden_layers=2, hidden_features=256)
hypernet = HyperNetwork(in_features=256, hidden_layers=1, hidden_features=256, hyponet=hyponet)
model = SDFHyperNetwork(encoder, hypernet, hyponet).cuda()

optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

model.train()
writer = SummaryWriter()
train_losses = []
for epoch in tqdm(range(500), desc='Epoch'):
    for step, batch_cpu in enumerate(tqdm(train_dataloader, desc='Train')):
        train_loss = next_step(
            model, train_dataset, epoch, step, batch_cpu, train_losses,
            get_forward_params=lambda batch_gpu: batch_gpu['index'],
            get_inner_preds=None,
        )

        writer.add_scalar('Loss/train', train_loss, step)

        optim.zero_grad()
        train_loss.backward()
        optim.step()
