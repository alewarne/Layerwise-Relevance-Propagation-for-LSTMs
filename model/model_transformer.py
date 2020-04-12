import pickle as pkl
import numpy as np
import argparse


def permute(weights, order):
    no_units = int(weights.shape[-1]/4)
    weights_reshaped = weights.reshape((-1, 4, no_units))
    new_weights_stacked = [weights_reshaped[:,i] for i in order]
    new_weights = np.stack(new_weights_stacked, axis=1)
    new_weights = new_weights.reshape((-1,4*no_units))
    return np.squeeze(new_weights)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('weight_path', type=str, help='Path to .pkl file storing list of weights.')
    parser.add_argument('order', type=int, nargs=4, help='Permutation order, e.g. [0,2,1,3] swaps second'
                                                        'and third quarter of the weights.')
    args = parser.parse_args()
    weights_old = pkl.load(open(args.weight_path, 'rb'))
    # last two entries are final weight matrix and bias which do not have to be permuted
    weights_new = [permute(w, args.order) for w in weights_old[:-2]]
    weights_new += weights_old[-2:]
    pkl.dump(weights_new, open('weights_permuted.pkl', 'wb'))
