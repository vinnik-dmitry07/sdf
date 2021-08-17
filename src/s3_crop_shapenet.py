import pandas as pd
from tqdm import tqdm

from render import coords_to_image

store = pd.HDFStore('../datasets/shapenet30000.h5', 'r')

num = 30000
keys = store.keys()
for k in tqdm(keys):
    coords = store[k].iloc[:, 0:3].values
    real_sdf = store[k].iloc[:, 3].values
    render_coords = coords[real_sdf <= 0]
    coords_to_image(render_coords).convert('RGB').save(f'../datasets/points/{k}.jpg', 'JPEG')
    # if len(store[k]) < num or k.lstrip('/') in skip:
    #     tqdm.write('skip ' + k)
    #     continue
    # cropped_store[k] = store[k].sample(num)

store.close()
