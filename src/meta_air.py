import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from common import l2_batch_loss, next_step, ShapeNetDataset, multitask_loss
from meta_modules import MetaSDF
from modules import ReLUFC, sal_init, sal_init_last_layer

dtype = torch.float32
torch.set_default_dtype(dtype)
train_dataset = ShapeNetDataset(split='train', dtype=dtype)
val_dataset = ShapeNetDataset(split='val', dtype=dtype)

train_dataloader = DataLoader(train_dataset, batch_size=1)
val_dataloader = DataLoader(val_dataset, batch_size=1)

hyponet = ReLUFC(
    in_features=3,
    out_features=2,  # 1 - l1, l2; 2 - multitask
    num_hidden_layers=8,
    hidden_features=256,  # 512
)
hyponet.net.apply(sal_init)
hyponet.net[-1].apply(sal_init_last_layer)

model = MetaSDF(
    hyponet,
    multitask_loss,  # 3d - multitask_loss, l2_batch_loss - mnist
    init_lr=5e-3,
    num_meta_steps=5,  # 5
    first_order=False,
    lr_type='per_parameter_per_step',
).cuda()

optim = torch.optim.Adam(lr=1e-4, params=model.parameters())

writer = SummaryWriter()
train_losses = []
val_losses = []
for epoch in tqdm(range(1500), desc='Epoch'):
    model.train()
    for step, batch_cpu in enumerate(tqdm(train_dataloader, desc='Train')):
        train_loss = next_step(
            model, train_dataset, epoch, step, batch_cpu, train_losses,
            log_every=100,
            get_context_params=lambda batch_gpu: model.generate_params(batch_gpu['context']),
            get_context_params_test=lambda batch_gpu: model.generate_params(batch_gpu['surface']),
        )

        writer.add_scalar('Loss/train', train_loss, global_step=step + epoch * len(train_dataloader))

        optim.zero_grad()
        train_loss.backward()
        # torch.nn.utils.clip_grad_value_(model.parameters(), 0.5)
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
