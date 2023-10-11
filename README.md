# Face Similarity using Triplet Loss

This repository contains a PyTorch-based project for face similarity using triplet loss. The project is designed to help you perform facial similarity analysis using deep learning techniques.

## Installation

To set up the environment for this project, you can follow these steps:

1. Clone the repository to your local machine:

```bash
git clone https://github.com/your-username/Face-Similarity.git
cd face-similarity-pytorch
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use 'venv\Scripts\activate'
```

3. Install the required dependencies using pip:

```bash
pip install -r requirements.txt
```

## Usage

To run the face similarity project, you can use the following command-line arguments:

```bash
python main.py \
      --gpu <gpu_id> \
      --num_epochs <num_epochs> \
      --batch_size <batch_size> \
      --workers <workers> \
      --lr <learning_rate> \
      --data_dir <data_directory> \
      --output_string <output_string> \
      --snapshot <snapshot_path>
```

Here's a brief explanation of the available arguments:

- `--gpu`: GPU device id to use (default: 0).
- `--num_epochs`: Maximum number of training epochs (default: 64).
- `--batch_size`: Batch size (default: 3).
- `--workers`: Number of data loading workers (default: 5).
- `--lr`: Base learning rate (default: 0.0001).
- `--data_dir`: Directory path for the data (default: '/home/lumalfa/datasets/raw-img/').
- `--output_string`: String appended to output snapshots (default: '').
- `--snapshot`: Path of the model snapshot (default: './snapshots').

Adjust these arguments as needed to suit your specific project requirements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

Feel free to contribute to this project or provide feedback by creating issues or pull requests. If you have any questions or encounter any problems, please don't hesitate to contact us.

**Happy coding!**
