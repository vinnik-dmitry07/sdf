import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import l2_loss, next_step, CIFAR10
from meta_modules import MetaSDF, Siren

dtype = torch.float32
torch.set_default_dtype(dtype)
train_dataset = CIFAR10(split='train', dtype=dtype)
val_dataset = CIFAR10(split='val', dtype=dtype)

train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=32)

hyponet = Siren(in_features=2, hidden_features=128, hidden_layers=3, out_features=3, outermost_linear=True)
model = MetaSDF(
    hyponet,
    l2_loss,
    init_lr=1e-5,
    num_meta_steps=3,
    first_order=False,
    lr_type='per_parameter_per_step',
).cuda()

optim = torch.optim.Adam(lr=5e-5, params=model.parameters())

writer = SummaryWriter()
train_losses = []
val_losses = []
for epoch in tqdm(range(3), desc='Epoch'):
    model.train()
    for step, batch_cpu in enumerate(tqdm(train_dataloader, desc='Train')):
        train_loss = next_step(
            model, train_dataset, epoch, step, batch_cpu, train_losses,
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
                model, val_dataset, epoch, step, batch_cpu, val_losses,
                get_context_params=lambda batch_gpu: model.generate_params(batch_gpu['context']),
                get_context_params_test=lambda batch_gpu: model.generate_params(batch_gpu['surface']),
            )

            writer.add_scalar('Loss/valid', valid_loss, global_step=step + epoch * len(val_dataloader))
