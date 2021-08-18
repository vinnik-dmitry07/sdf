import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import MNISTSDFDataset, l2_batch_loss, next_step, l2_loss
from meta_modules import MetaSDF
from modules import ReLUFC, sal_init, sal_init_last_layer

dtype = torch.float32
torch.set_default_dtype(dtype)
train_dataset = MNISTSDFDataset(split='train', dtype=dtype, side_len=64)
val_dataset = MNISTSDFDataset(split='val', dtype=dtype, side_len=64)

train_dataloader = DataLoader(train_dataset, batch_size=16)
val_dataloader = DataLoader(val_dataset, batch_size=16)

hypo_net = ReLUFC(in_features=2, out_features=1, num_hidden_layers=2, hidden_features=256)
hypo_net.net.apply(sal_init)
hypo_net.net[-1].apply(sal_init_last_layer)

model = MetaSDF(
    hypo_net,
    l2_batch_loss,
    init_lr=1e-1,  # 1e-5
    num_meta_steps=3,
    first_order=False,
    lr_type='global',
).cuda()

optim = torch.optim.Adam(lr=1e-4, params=model.parameters())


writer = SummaryWriter()
for epoch in tqdm(range(3000), desc='Epoch'):
    model.train()
    for step, batch_cpu in enumerate(tqdm(train_dataloader, desc='Train')):
        train_loss = next_step(
            model, l2_loss, train_dataset, epoch, step, batch_cpu,
            get_context_params=lambda batch_gpu: model.generate_params(batch_gpu['context']),
            get_context_params_test=lambda batch_gpu: model.generate_params(batch_gpu['surface']),
        )

        writer.add_scalar('Loss/train', train_loss, global_step=step + epoch * len(train_dataloader))

        optim.zero_grad()
        train_loss.backward()
        optim.step()

    model.eval()
    with torch.no_grad():
        for step, batch_cpu in enumerate(tqdm(val_dataloader, desc='Valid')):
            valid_loss = next_step(
                model, l2_loss, val_dataset, epoch, step, batch_cpu,
                get_context_params=lambda batch_gpu: model.generate_params(batch_gpu['context']),
                get_context_params_test=lambda batch_gpu: model.generate_params(batch_gpu['surface']),
            )

            writer.add_scalar('Loss/valid', valid_loss, global_step=step + epoch * len(val_dataloader))
