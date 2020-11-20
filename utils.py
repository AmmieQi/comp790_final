import numpy as np
import matplotlib.pyplot as plt
import torch

def plot_tsne(real_data, synthetic_data, encoder, filename):

    with torch.no_grad():

        encoded_tensors = []
        labels = [] #0 is real, 1 is synthetic
        color_dict = {0: 'r', 1: 'b'}

        for x in real_data:

            real_vid, _, _ = x
            batch_size = real_vid.shape[0]

            real_encoded, _= encoder(real_vid) #, raw_vids = True)
            # real_encoded = real_encoded.mean(dim=1).detach().cpu().numpy()
            # real_encoded = real_encoded.max(dim=1)[0].detach().cpu().numpy()
            real_encoded = real_encoded[:,0].detach().cpu().numpy()
            encoded_tensors.append(real_encoded)
            labels.append([0] * batch_size)

        syn_ct = 0
        for idx, x in enumerate(synthetic_data):

            if idx < 5: continue

            syn_vid, _, _ = x
            batch_size = syn_vid.shape[0]
            syn_encoded, _ = encoder(syn_vid) #, raw_vids = False)
            # syn_encoded = syn_encoded.mean(dim=1).detach().cpu().numpy()
            # syn_encoded = syn_encoded.max(dim=1)[0].detach().cpu().numpy()
            syn_encoded = syn_encoded[:,0].detach().cpu().numpy()
            encoded_tensors.append(syn_encoded)
            labels.append([1] * batch_size)

            syn_ct += batch_size


        encoded_tensors = np.vstack(encoded_tensors)
        labels = np.asarray([item for sublist in labels for item in sublist])

        X_embedded = TSNE().fit_transform(encoded_tensors)

        for label in np.unique(labels):
            label_idxs = (label == labels)
            color = color_dict[label]

            these_pts = X_embedded[label_idxs]

            xs = these_pts[:,0]
            ys = these_pts[:,1]
            colors = [color] * len(ys)
            if label == 0:
                label_plt = 'real-life'
            else:
                label_plt = 'synthetic'
            plt.scatter(xs, ys, c = colors, label=label_plt)
            plt.legend(loc = 'lower left')
            plt.axis('off')


            # if label == 0:
            #     fname = '{}_real_pts_latex.txt'.format(filename)
            # else:
            #     fname = '{}_syn_pts_latex.txt'.format(filename)
            #
            # with open(fname, 'w+') as f:
            #
            #     f.write('x y\n')
            #     for x, y in zip(xs, ys):
            #         f.write(str(x) + ' ' +str(y) + '\n')


        plt.savefig(filename)
        plt.clf()
