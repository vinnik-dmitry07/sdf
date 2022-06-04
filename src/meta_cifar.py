import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import l2_hyper_loss, next_step, CIFAR10
from meta_modules import MetaSDF, Siren

dtype = torch.float32
torch.set_default_dtype(dtype)
train_dataset = CIFAR10(split='train', dtype=dtype)
val_dataset = CIFAR10(split='val', dtype=dtype)

train_dataloader = DataLoader(train_dataset, batch_size=32)
val_dataloader = DataLoader(val_dataset, batch_size=32)

hypo_net = Siren(in_features=2, hidden_features=128, hidden_layers=3, out_features=3, outermost_linear=True)
model = MetaSDF(
    hypo_net,
    hypo_loss=l2_hyper_loss,  # MAML ignores batch
    init_lr=1e-5,
    num_meta_steps=3,
    first_order=False,
    lr_type='per_parameter_per_step',
).cuda()

optim = torch.optim.Adam(lr=5e-5, params=model.parameters())

writer = SummaryWriter()
for epoch in tqdm(range(3000), desc='Epoch'):
    model.train()
    for step, batch_cpu in enumerate(tqdm(train_dataloader, desc='Train')):
        train_loss = next_step(
            model, l2_hyper_loss, train_dataset, epoch, step, batch_cpu,
            draw_meta_steps=True,
            get_context_params=lambda batch_gpu: model.generate_params(batch_gpu['context']),
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
                draw_meta_steps=True,
                get_context_params=lambda batch_gpu: model.generate_params(batch_gpu['context']),
            )

            writer.add_scalar('Loss/valid', valid_loss, global_step=step + epoch * len(val_dataloader))
