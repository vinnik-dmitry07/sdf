import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import sdf_loss, next_step, CIFAR10
from meta_modules import MetaSDF, Siren

hyponet = Siren(in_features=2, hidden_features=128, hidden_layers=3, out_features=3, outermost_linear=True)
model = MetaSDF(
    hyponet,
    sdf_loss,
    init_lr=1e-5,
    num_meta_steps=3,
    first_order=False,
    lr_type='per_parameter_per_step',
).cuda()

train_dataset = CIFAR10()
train_dataloader = DataLoader(train_dataset, batch_size=32, num_workers=0)

optim = torch.optim.Adam(lr=5e-5, params=model.parameters())

model.train()
writer = SummaryWriter()
train_losses = []
val_losses = []
for epoch in tqdm(range(3), desc='Epoch'):
    for step, batch_cpu in enumerate(tqdm(train_dataloader, desc='Train')):
        train_loss = next_step(
            model, train_dataset, epoch, step, batch_cpu, train_losses,
            get_forward_params=lambda batch_gpu: model.generate_params(batch_gpu['context'])[0],
            get_inner_preds=lambda batch_gpu: model.generate_params(batch_gpu['context'])[1],
        )

        writer.add_scalar('Loss/train', train_loss, step)

        optim.zero_grad()
        train_loss.backward()
        optim.step()
