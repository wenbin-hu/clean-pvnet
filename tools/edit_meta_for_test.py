import numpy as np

meta_path = "/home/wenbin/clean-pvnet/demo_images/mug/"

old_meta = np.load(meta_path + "meta.npy", allow_pickle=True).item()
new_meta = old_meta.copy()

new_meta['kpt_3d'] = np.array([[-0.030731409788131714, 0.03298131003975868, -0.055316731333732605],
                               [0.03488881140947342, -0.030913710594177246, -0.05337449908256531],
                               [-0.033415451645851135, -0.033679720014333725, -0.05031098052859306],
                               [-0.03622955083847046, 0.02747773937880993, 0.04688483104109764],
                               [0.035258948802948, 0.03317528963088989, -0.04303786903619766],
                               [-0.0036999830044806004, -0.03482731059193611, 0.05347912013530731],
                               [0.019706500694155693, 0.01780232973396778, 0.05171598121523857],
                               [0.0012268710415810347, 0.0004025551024824381, -0.04642834886908531],
                               [0., 0., 0.]])
new_meta['corner_3d'] = np.array([[-0.04076768, -0.04901402, -0.05740402],
                                  [-0.04076768, -0.04901402, 0.05740402],
                                  [-0.04076768, 0.04901402, -0.05740402],
                                  [-0.04076768, 0.04901402, 0.05740402],
                                  [0.04076768, -0.04901402, -0.05740402],
                                  [0.04076768, -0.04901402, 0.05740402],
                                  [0.04076768, 0.04901402, -0.05740402],
                                  [0.04076768, 0.04901402, 0.05740402]])

np.save(meta_path+'meta.npy', new_meta, allow_pickle=True)