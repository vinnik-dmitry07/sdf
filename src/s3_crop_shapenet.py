import pandas as pd
from tqdm import tqdm

store = pd.HDFStore('../datasets/shapenet6000030000.h5', 'r')
cropped_store = pd.HDFStore('../datasets/shapenet_cropped.h5', 'w')

skip = {
    '_12877bdca58ccbea402991f646f01d6c',
    '_17ac3afd54143b797172a40a4ca640fe',
    '_2066f1830765183544bb6b89d63deb6f',
    '_240136d2ae7d2dec9fe69c7ccc27d2bf',
    '_24bdf389877fb7f21b1f694e36340ebb',
    '_2af04ef09d49221b85e5214b0d6a7',
    '_310f0957ae1436d88025a4ffa6c0c22b',
    '_3b82e575165383903c83f6e156ad107a',
    '_3f9cab3630160be9f19e1980c1653b79',
    '_45bd6c0723e555d4ba2821676102936',
    '_461891f9231fc09a3d21c2160f47f16',
    '_541ad6a69f87b134f4c1adce71073351',
    '_57f1dfb095bbe82cafc7bdb2f8d1ea84',
    '_5e34c340059f5b4b1c97b7d78f1a34d4',
    '_66e60b297f733435fff6684ee288fa59',
    '_692797a818b4630f1aa3e317da5a1267',
    '_81e6b629264dad5daf2c6c19cc41708a',
    '_84b84d39581a6766925c10fa44a32fd7',
    '_8e2e03ed888e0eace4f2488af2c37f8d',
    '_99ee9ae50909ac0cd3cd0742a4ec7e9b',
    '_ad66ac8155a316422068b7c500584ade',
    '_b29c650e4d7582d11ae96ac7591d0dc5',
    '_e0df54a0990000035dde10d3d9d6a06',
    '_e53547a01129eef87eda1e12bd28fb7',
    '_e7e73007e0373933c4c280b3db0d6264',
    '_f31be358cb57dffffe198fc7b510f52f',
    '_fa27e66018f82cf6e549ab640f51dca9',
    '_fa9d933674352c3711a50c43722a30cb',
}

num = 30000
for k in tqdm(store.keys()):
    if len(store[k]) < num or k.lstrip('/') in skip:
        tqdm.write('skip ' + k)
        continue
    cropped_store[k] = store[k].sample(num)

store.close()
cropped_store.close()
