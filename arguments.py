import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='Node_UPS Training')
    parser.add_argument('--dataset', type=str, default='cora', help='Data set')
    # 'cora', 'citeseer', 'pubmed'
    # cs, computers, photo

    parser.add_argument('--out', default='outputs', help='directory to output the result')
    parser.add_argument('--num-workers', type=int, default=8,
                        help='number of workers')
    parser.add_argument('--iterations', default=20, type=int,
                        help='number of total pseudo-labeling iterations to run')
    parser.add_argument('--epochs', type=int, default=500,
                        help='Number of epochs to train.')
    parser.add_argument('--seed', type=int, default=-1,
                        help="random seed (-1: don't use random seed)")
    parser.add_argument('--dropout', default=0.2, type=float,
                        help='dropout probs')
    parser.add_argument('--class-blnc', default=1, type=int,
                        help='total number of class balanced iterations')
    parser.add_argument('--tau-p', default=0.9, type=float,
                        help='confidece threshold for positive pseudo-labels, default 0.70')
    parser.add_argument('--kappa-p', default=0.2, type=float,
                        help='uncertainty threshold for positive pseudo-labels, default 0.05')
    parser.add_argument('--temp-nl', default=2.0, type=float,
                        help='temperature for generating negative pseduo-labels, default 2.0')
    parser.add_argument('--no_uncertainty', action='store_true',
                        help='do not use uncertainty in the pesudo-label selection')

    parser.add_argument('--lr', type=float, default=0.01,
                        help='Initial learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--hidden', type=int, default=256,
                        help='Number of hidden units.')
    parser.add_argument('--droprate', type=float, default=0.2,
                        help='Dropout rate of the layer (1 - keep probability).')
    parser.add_argument('--patience', type=int, default=200, help='Patience')

    parser.add_argument('--K', type=int, default=5)
    parser.add_argument('--normalize_features', type=bool, default=True)
    parser.add_argument('--public_splits', default=False, action='store_true')

    parser.add_argument('--percentiles_holder', default=20, type=int, 
                        help='mu parameter - sets the steping percentile for thresholding after each iteration')
    parser.add_argument('--uncertainty_percentiles', default=0.1, type=float,
                        help='sets the uncertainty steping percentile for thresholding after each iteration')

    parser.add_argument("--edge_aug_ratio", default=0.1, type=float, help='pedges')
    parser.add_argument("--mask_aug_ratio", default=0.1, type=float, help='mask_nodes')
    parser.add_argument("--tag_ratio", default=0.65, type=float, help='adjust args.select_unlabel dynamically')
    parser.add_argument('--aug_sample_times', type=int, default=10)
    parser.add_argument('--log_file', type=str, default='result_0107_new.txt', help='name of file for logging')
    parser.add_argument('--train_class_size', type=int, default=30)
    parser.add_argument("--edge_aug_ratio_train", default=0.1, type=float, help='pedges at the training stage')
    parser.add_argument("--mask_aug_ratio_train", default=0.1, type=float, help='mask_nodes at the training stage')
    parser.add_argument('--expt_name', type=str, default='main', help='name of experiment')
    parser.add_argument('--no_curriculum', action='store_true',
                        help='do not use curriculum strategy in the pesudo-label selection')
    parser.add_argument('--unlabel_part', default=False, action='store_true')
    parser.add_argument('--log_dir', default='log_dir', help='directory to save log')
    return parser.parse_args()
