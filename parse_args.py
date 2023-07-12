import argparse

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(
        description='Face similairy using triplet loss')
    parser.add_argument(
        '--gpu', dest='gpu_id', help='GPU device id to use [0]',
        default=0, type=int)
    parser.add_argument(
        '--num_epochs', dest='num_epochs',
        help='Maximum number of training epochs.',
        default=64, type=int)
    parser.add_argument(
        '--batch_size', dest='batch_size', help='Batch size.',
        default=3, type=int)
    parser.add_argument(
        '--workers', dest='workers', help='Num workers.',
        default=5, type=int)
    parser.add_argument(
        '--lr', dest='lr', help='Base learning rate.',
        default=0.0001, type=float)
    parser.add_argument(
        '--data_dir', dest='data_dir', help='Directory path for data.',
        default='/home/lumalfa/datasets/raw-img/', type=str)
    parser.add_argument(
        '--output_string', dest='output_string',
        help='String appended to output snapshots.', default='', type=str)
    parser.add_argument(
        '--snapshot', dest='snapshot', help='Path of model snapshot.',
        default='./snapshots', type=str)

    args = parser.parse_args()
    return args