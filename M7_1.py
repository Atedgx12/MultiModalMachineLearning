# -----------------------------
# 1. Import Libraries
# -----------------------------
import os
import sys
import math
import json
import queue
import logging
import random
import threading
from io import BytesIO

# Numerical & Data Handling
import numpy as np
import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.model_selection import train_test_split

# Visualization & GUI
import matplotlib
matplotlib.use('TkAgg')  # Use Tkinter-compatible backend for Matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d import Axes3D  # For 3D plots

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk

# Model Architectures
from torchvision.models import resnet18, ResNet18_Weights

# Tree Structure & Visualization
from anytree import NodeMixin, RenderTree
from anytree.exporter import DotExporter

# Learning Rate Schedulers
from torch.optim import lr_scheduler
from dataclasses import dataclass
from graphviz import Source
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm  # For progress bars
import time
from PIL import Image, ImageTk, ImageDraw

# -----------------------------
# 2. Configure Logging and Seed
# -----------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logger = logging.getLogger(__name__)

def add_noise_and_dead_squares(grid, noise_prob=0.1, dead_prob=0.05):
    noise_mask = np.random.rand(*grid.shape) < noise_prob
    dead_mask = np.random.rand(*grid.shape) < dead_prob

    noise = np.random.randint(0, NUM_CLASSES - 1, size=grid.shape)
    grid = np.where(noise_mask, noise, grid)
    grid = np.where(dead_mask, -1, grid)  # Assign dead squares

    return grid

# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed()

# -----------------------------
# 3. Define Constants
# -----------------------------

# Define the number of classes
NUM_CLASSES = 11  # 0-10, where 10 represents dead squares

# Define fixed image dimensions
FIXED_HEIGHT = 32
FIXED_WIDTH = 32

# -----------------------------
# 4. Define Data Structures and Loading Functions
# -----------------------------

# Data Class for Grid Pairs
@dataclass
class GridPair:
    task_id: str
    input_grid: np.ndarray
    output_grid: np.ndarray

def load_arc_data():
    file_paths = {
        "arc-agi_training-challenges": "arc-agi_training_challenges.json",
        "arc-agi_evaluation-challenges": "arc-agi_evaluation_challenges.json",
        "arc-agi_training-solutions": "arc-agi_training_solutions.json",
        "arc-agi_evaluation-solutions": "arc-agi_evaluation_solutions.json",
    }
    arc_data = {key: load_json_file(path) for key, path in file_paths.items()}
    return arc_data

def load_json_file(path):
    try:
        with open(path, 'r') as f:
            data = json.load(f)
            logger.info(f"Loaded data from {path}.")
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading {path}: {e}")
        return {}

def get_device():
    """Detect the best available device: CUDA or CPU."""
    if torch.cuda.is_available():
        device = torch.device('cuda')  # CUDA GPU
        logger.info("Using NVIDIA GPU via CUDA.")
    else:
        device = torch.device('cpu')  # Fallback to CPU
        logger.info("Using CPU as fallback.")
    return device

def extract_and_reshape_grid(grid):
    try:
        # Convert to NumPy array if not already
        grid = np.array(grid)
        # Handle empty grids or grids with zero dimensions
        if grid.size == 0 or 0 in grid.shape:
            logger.error(f"Empty grid or grid with zero dimension encountered: {grid.shape}")
            return None
        # Ensure grid is 2D
        if grid.ndim == 1:
            # If the grid is 1D, reshape to (1, N)
            grid = grid.reshape(1, -1)
            logger.warning(f"Grid reshaped to 2D: {grid.shape}")
        elif grid.ndim > 2:
            grid = grid.squeeze()
            if grid.ndim > 2:
                logger.error(f"Grid has more than 2 dimensions after squeeze: {grid.shape}")
                return None
        return grid  # Return as is, without resizing
    except Exception as e:
        logger.error(f"Error processing grid: {e}")
        return None

# Flatten and Reshape Grid Data
def flatten_and_reshape(task_data):
    flattened_pairs = []
    for task_id, task_content in task_data.items():
        logger.info(f"Parsing task {task_id}...")
        train_pairs = task_content.get('train', [])
        for pair in train_pairs:
            input_grid = extract_and_reshape_grid(pair.get("input"))
            output_grid = extract_and_reshape_grid(pair.get("output"))
            if input_grid is not None and output_grid is not None:
                # Check for zero dimensions in input or output grid
                if 0 in input_grid.shape or 0 in output_grid.shape:
                    logger.warning(f"Task ID: {task_id} has grid with zero dimension. Skipping.")
                    continue
                # Store the grids even if shapes differ
                flattened_pairs.append(GridPair(task_id, input_grid, output_grid))
            else:
                logger.warning(f"Task ID: {task_id} has invalid input/output grids.")
    logger.info(f"Total valid grid pairs extracted: {len(flattened_pairs)}")
    return flattened_pairs

def grid_to_image(grid, color_map):
    img_array = np.zeros((grid.shape[0], grid.shape[1], 3), dtype=np.uint8)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            img_array[i, j] = color_map.get(grid[i, j], [0, 0, 0])  # Default to black
    return Image.fromarray(img_array)

color_map = {
    0: [0, 0, 0],       # Black
    1: [255, 0, 0],     # Red
    2: [0, 255, 0],     # Green
    3: [0, 0, 255],     # Blue
    # Add more colors as needed
}

class TreeNode(NodeMixin):
    def __init__(self, name, input_grid=None, parent=None, children=None):
        self.name = name
        self.input_grid = input_grid
        self.embedding = None
        self.parent = parent
        if children:
            self.children = children

        logger.info(f"Node '{self.name}' initialized.")

    def set_embedding(self, embedding):
        """
        Set the embedding for the node.
        """
        self.embedding = embedding
        logger.info(f"Embedding set for node '{self.name}'.")

    def __repr__(self):
        """String representation for easier debugging."""
        return f"TreeNode(name={self.name}, children={len(self.children) if self.children else 0})"

def build_data_tree(grid_pairs):
    """
    Build a hierarchical tree from the ARC data using the Node class,
    and create a task dictionary for quick access.

    Args:
        grid_pairs (list): List of GridPair objects.

    Returns:
        tuple: (Node, dict) - Root node and task dictionary.
    """
    # Create the root node
    root = TreeNode(name="ARC Dataset")

    # Initialize the task dictionary
    task_dict = {}

    # Loop through the grid pairs to build task nodes
    for idx, pair in enumerate(grid_pairs):
        try:
            if not isinstance(pair, GridPair):
                raise TypeError(f"Expected GridPair, got {type(pair)}: {pair}")

            # Ensure grids are NumPy arrays
            input_grid = np.array(pair.input_grid) if not isinstance(pair.input_grid, np.ndarray) else pair.input_grid
            output_grid = np.array(pair.output_grid) if not isinstance(pair.output_grid, np.ndarray) else pair.output_grid

            # Create task and output nodes
            task_node = TreeNode(name=f"Task {pair.task_id}", parent=root)
            output_node = TreeNode(name=f"Output {pair.task_id}", parent=task_node)

            # Set embeddings for the nodes
            task_node.set_embedding(input_grid)
            output_node.set_embedding(output_grid)

            # Store the nodes in the task dictionary
            task_dict[pair.task_id] = {
                'task_node': task_node,
                'output_node': output_node,
                'grids': (input_grid, output_grid)
            }

            # Log success
            logger.info(f"Created task node for {pair.task_id} with embedding shape: {task_node.embedding.shape}")

        except Exception as e:
            logger.exception(f"Failed to create nodes for grid pair {idx}: {e}")
            continue  # Skip this pair if there's an issue

    # Return the root node and the task dictionary
    return root, task_dict

# -----------------------------
# 5. Data Augmentation Functions
# -----------------------------

def augment_grid(grid, noise_prob=0.2, dead_square_prob=0.1):
    augmented_grid = np.array(grid)

    # Ensure the grid is 2D
    if augmented_grid.ndim != 2:
        logger.error(f"Augmenting grid failed due to invalid shape: {augmented_grid.shape}")
        return augmented_grid  # Return the original grid without augmentation

    # Random noise and dead square masks
    noise_mask = np.random.rand(*augmented_grid.shape) < noise_prob
    dead_mask = np.random.rand(*augmented_grid.shape) < dead_square_prob

    # Apply noise
    noise_values = np.random.randint(0, NUM_CLASSES - 1, size=augmented_grid.shape)
    augmented_grid = np.where(noise_mask, noise_values, augmented_grid)
    augmented_grid = np.where(dead_mask, -1, augmented_grid)  # Mark as dead squares

    return augmented_grid

def rotate_grid(grid):
    """Randomly rotates the grid."""
    rotations = random.choice([0, 1, 2, 3])
    return np.rot90(grid, rotations)

def flip_grid(grid):
    """Randomly flips the grid."""
    flip_choice = random.choice(['none', 'vertical', 'horizontal'])
    if flip_choice == 'vertical':
        return np.flipud(grid)  # Vertical flip
    elif flip_choice == 'horizontal':
        return np.fliplr(grid)  # Horizontal flip
    else:
        return grid  # No flip

def generate_multiple_augmented_datasets(grid_pairs, num_augmented_sets=3):
    """
    Generates multiple augmented datasets from the input grid pairs.

    Args:
        grid_pairs (list): List of GridPair objects.
        num_augmented_sets (int): Number of augmented sets to generate.

    Returns:
        list: Augmented grid pairs.
    """
    augmented_pairs = []
    for _ in range(num_augmented_sets):
        for pair in grid_pairs:
            # Apply augmentations to input grid
            augmented_input = augment_grid(pair.input_grid)

            # Optionally rotate and flip
            augmented_input = rotate_grid(augmented_input)
            augmented_input = flip_grid(augmented_input)

            # Append the augmented input with the original target grid
            augmented_pairs.append(GridPair(pair.task_id, augmented_input, pair.output_grid))

    return augmented_pairs

# -----------------------------
# 6. PyTorch Dataset Class
# -----------------------------

from torchvision.transforms import Resize
from PIL import Image

class AugmentedARCDataset(torch.utils.data.Dataset):
    def __init__(self, grid_pairs, augment=False):
        # Filter out pairs where input or output grid has zero dimensions
        self.grid_pairs = [
            pair for pair in grid_pairs
            if pair.input_grid.size != 0 and pair.output_grid.size != 0
            and 0 not in pair.input_grid.shape and 0 not in pair.output_grid.shape
        ]
        self.augment = augment
        self.resize = Resize((FIXED_HEIGHT, FIXED_WIDTH))
        logger.info(f"Dataset initialized with {len(self.grid_pairs)} valid grid pairs.")

    def __len__(self):
        return len(self.grid_pairs)

    def __getitem__(self, idx):
        # Get the GridPair object
        pair = self.grid_pairs[idx]

        # Access the input and target grids
        input_grid = pair.input_grid
        target_grid = pair.output_grid

        # Apply augmentation if enabled
        if self.augment:
            input_grid = augment_grid(input_grid)

        # Convert grids to PIL Images for resizing
        input_image = Image.fromarray(input_grid.astype(np.uint8))
        input_image = self.resize(input_image)
        input_grid_resized = np.array(input_image)

        # Convert to tensors
        input_tensor = torch.tensor(input_grid_resized, dtype=torch.float32).unsqueeze(0)  # Shape: [1, H, W]
        target_image = Image.fromarray(target_grid.astype(np.uint8))
        target_image = self.resize(target_image)
        target_grid_resized = np.array(target_image)
        target_tensor = torch.tensor(target_grid_resized, dtype=torch.long)

        # Ensure target_tensor is 2D
        if target_tensor.dim() > 2:
            target_tensor = target_tensor.squeeze()

        # Debugging statements
        logger.debug(f"Index {idx}:")
        logger.debug(f"  Input tensor shape: {input_tensor.shape}")
        logger.debug(f"  Target tensor shape: {target_tensor.shape}")

        return input_tensor, target_tensor

# -----------------------------
# 7. Define the Deep Neural Network Models
# -----------------------------

class CNNGridMapper(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super(CNNGridMapper, self).__init__()
        self.num_classes = num_classes

        # Use a CNN backbone (e.g., ResNet18)
        self.cnn = resnet18(weights=ResNet18_Weights.DEFAULT)

        # Modify the first convolutional layer for single-channel input
        self.cnn.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        nn.init.kaiming_normal_(self.cnn.conv1.weight, mode='fan_out', nonlinearity='relu')

        # Remove the fully connected layer
        self.cnn_layers = nn.Sequential(*list(self.cnn.children())[:-2])

        # Upsampling layers to recover spatial dimensions
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, num_classes, kernel_size=2, stride=2)
        )

    def forward(self, x):
        x = self.cnn_layers(x)
        x = self.upsample(x)
        return x  # Output shape: (batch_size, num_classes, H', W')

class Generator(nn.Module):
    def __init__(self, latent_dim=100, output_channels=NUM_CLASSES, grid_size=(32, 32)):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.grid_size = grid_size
        self.main = nn.Sequential(
            # Input: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            
            # State size: 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            
            # State size: 256 x 8 x 8
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # State size: 128 x 16 x 16
            nn.ConvTranspose2d(128, output_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        output = self.main(z)
        print(f"Generator output shape: {output.shape}")  # Debugging
        return output  # Expected Output: [batch_size, 11, 32, 32]

class Discriminator(nn.Module):
    def __init__(self, input_channels=NUM_CLASSES, grid_size=(32, 32)):
        super(Discriminator, self).__init__()
        self.input_channels = input_channels
        self.grid_size = grid_size
        self.main = nn.Sequential(
            # input is (input_channels) x 32 x 32
            nn.Conv2d(input_channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: 64 x 16 x 16
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: 128 x 8 x 8
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            # State size: 256 x 4 x 4
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.main(x).view(-1, 1).squeeze(1)

# -----------------------------
# 8. Custom Collate Function
# -----------------------------

def collate_fn(batch):
    inputs = [item[0] for item in batch]  # Shape: [C, H, W]
    targets = [item[1] for item in batch]  # Shape: [H, W]

    # Find max dimensions in the batch for inputs and targets separately
    max_input_height = max(t.size(-2) for t in inputs)
    max_input_width = max(t.size(-1) for t in inputs)
    max_target_height = max(t.size(-2) for t in targets)
    max_target_width = max(t.size(-1) for t in targets)

    batch_size = len(inputs)
    num_channels = inputs[0].size(0)

    # Initialize tensors with zeros
    batch_inputs = torch.zeros((batch_size, num_channels, max_input_height, max_input_width), dtype=inputs[0].dtype)
    batch_targets = torch.zeros((batch_size, max_target_height, max_target_width), dtype=targets[0].dtype)

    for i in range(batch_size):
        input_tensor = inputs[i]
        target_tensor = targets[i]

        # Get shapes
        c, h_inp, w_inp = input_tensor.size()
        h_tar, w_tar = target_tensor.size()

        # Copy input_tensor into batch_inputs
        batch_inputs[i, :, :h_inp, :w_inp] = input_tensor

        # Copy target_tensor into batch_targets
        batch_targets[i, :h_tar, :w_tar] = target_tensor

        # Debugging statements
        logger.debug(f"Batch index {i}:")
        logger.debug(f"  Input tensor shape: {input_tensor.shape}")
        logger.debug(f"  Target tensor shape: {target_tensor.shape}")
        logger.debug(f"  Batch input shape: {batch_inputs[i].shape}")
        logger.debug(f"  Batch target shape: {batch_targets[i].shape}")

    return batch_inputs, batch_targets

# -----------------------------
# 9. Training GUI Class
# -----------------------------

class TrainingGUI:
    """
    A Tkinter-based GUI for real-time training progress visualization with 3D metrics plotting and data tree integration.
    """

    def __init__(self, root, total_epochs, total_batches, model, train_loader, 
                 val_loader, eval_loader, device, data_tree, task_dict, generator, discriminator):
        """Initialize the Training GUI."""
        self.root = root
        self.total_epochs = total_epochs
        self.total_batches = total_batches
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.eval_loader = eval_loader
        self.device = device
        self.data_tree = data_tree  # Data tree integration
        self.task_dict = task_dict  # Store task dictionary for training logic

        # GAN models
        self.generator = generator
        self.discriminator = discriminator

        # Initialize other required attributes
        self.queue = queue.Queue()
        self.stop_event = threading.Event()

        # Initialize data storage for plots
        self.loss_data = []
        self.val_loss_data = []
        self.acc_data = []
        self.prediction_distances = []

        # Set up the GUI
        self.setup_gui()
        self.root.after(100, self.process_queue)

    def setup_gui(self):
        """Set up the GUI components."""
        self.frame = tk.Frame(self.root)
        self.frame.pack(fill=tk.BOTH, expand=True)

        # Top Section for Labels
        self.label_frame = tk.Frame(self.frame)
        self.label_frame.pack(pady=10)

        self.epoch_label = tk.Label(self.label_frame, text=f"Epoch: 0/{self.total_epochs}", font=("Helvetica", 14))
        self.epoch_label.grid(row=0, column=0, padx=10)

        self.batch_label = tk.Label(self.label_frame, text=f"Batch: 0/{self.total_batches}", font=("Helvetica", 12))
        self.batch_label.grid(row=0, column=1, padx=10)

        self.loss_label = tk.Label(self.label_frame, text="Loss: 0.0000", font=("Helvetica", 12))
        self.loss_label.grid(row=0, column=2, padx=10)

        self.accuracy_label = tk.Label(self.label_frame, text="Accuracy: 0.0000", font=("Helvetica", 12))
        self.accuracy_label.grid(row=0, column=3, padx=10)

        # Data Tree Visualization Section
        self.tree_frame = tk.Frame(self.frame, width=300, height=400)
        self.tree_frame.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.Y)

        self.tree_label = tk.Label(self.tree_frame, text="Data Tree", font=("Helvetica", 14))
        self.tree_label.pack()

        self.tree_canvas = tk.Canvas(self.tree_frame, width=300, height=400, bg='white')
        self.tree_canvas.pack()

        # Display the data tree
        self.display_data_tree()

        # Plot Section (2D + 3D)
        self.fig = plt.figure(figsize=(12, 6))

        # 3D Plot on the Left
        self.ax_3d = self.fig.add_subplot(121, projection='3d')
        self.ax_3d.set_xlabel('Epoch')
        self.ax_3d.set_ylabel('Accuracy')
        self.ax_3d.set_zlabel('Distance from Actual')

        # 2D Plot on the Right
        self.ax_2d = self.fig.add_subplot(122)
        self.line_loss, = self.ax_2d.plot([], [], label='Training Loss')
        self.line_val_loss, = self.ax_2d.plot([], [], label='Validation Loss')
        self.ax_2d.legend()

        self.canvas_plot = FigureCanvasTkAgg(self.fig, master=self.frame)
        self.canvas_plot.draw()
        self.canvas_plot.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Bottom Section for Control Buttons
        self.button_frame = tk.Frame(self.frame)
        self.button_frame.pack(pady=10)

        self.start_button = tk.Button(self.button_frame, text="Start Training", command=self.start_training)
        self.start_button.grid(row=0, column=0, padx=10)

        self.stop_button = tk.Button(self.button_frame, text="Stop Training", command=self.stop_training)
        self.stop_button.grid(row=0, column=1, padx=10)

        self.evaluate_button = tk.Button(self.button_frame, text="Evaluate Model", command=self.evaluate_model_button)
        self.evaluate_button.grid(row=0, column=2, padx=10)

    def display_data_tree(self):
        """Generate and display the data tree as an image."""
        try:
            # Render the tree to a PNG using anytree and Graphviz
            dot_file = "tree.dot"
            png_file = "tree.png"

            # Export to .dot file
            DotExporter(self.data_tree).to_dotfile(dot_file)
            logger.info(f"Tree exported to {dot_file}")

            # Convert .dot to .png using Graphviz
            result = os.system(f'dot -Tpng {dot_file} -o {png_file}')
            if result != 0:
                raise RuntimeError("Failed to generate PNG. Ensure Graphviz is installed and in PATH.")

            # Load and display the PNG image
            img = Image.open(png_file)
            img = img.resize((300, 400), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)

            # Display the image in the canvas
            self.tree_canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
            self.tree_canvas.image = img_tk  # Keep reference to avoid garbage collection
            logger.info("Tree visualization displayed successfully.")

        except Exception as e:
            logger.exception("Failed to display the data tree.")
            tk.messagebox.showerror("Tree Display Error", f"Error: {e}")

    def process_queue(self):
        """Process the queue for thread-safe GUI updates."""
        while not self.queue.empty():
            message = self.queue.get()
            if isinstance(message, dict):
                self.update_gui(message)
        self.root.after(100, self.process_queue)

    def update_gui(self, data):
        """Update the GUI with real-time training and validation metrics."""
        try:
            if 'batch' in data:
                # Update batch-level metrics in the GUI
                self.batch_label.config(text=f"Batch: {data['batch']}/{self.total_batches}")
                self.loss_label.config(text=f"Loss: {data['loss']:.4f}")
                self.accuracy_label.config(text=f"Accuracy: {data.get('accuracy', 0.0):.4f}")

                # Append new batch data to the 2D plot lists
                self.loss_data.append(data['loss'])
                self.acc_data.append(data.get('accuracy', 0.0))

                # Update the 2D plot with each batch completion
                batches = list(range(1, len(self.loss_data) + 1))
                self.line_loss.set_data(batches, self.loss_data)
                self.line_val_loss.set_data(batches, self.acc_data)

                # Adjust the axes to fit the new data
                self.ax_2d.relim()
                self.ax_2d.autoscale_view()

                # Redraw the 2D plot with new data
                self.canvas_plot.draw()

            elif 'epoch' in data:
                # Update epoch-level metrics
                self.epoch_label.config(text=f"Epoch: {data['epoch']}/{self.total_epochs}")

                # Calculate prediction error distance
                predicted = np.array(data.get('guesses', []))
                actual = np.array(data.get('actuals', []))

                if predicted.size == 0 or actual.size == 0:
                    distance = float('nan')  # Handle empty arrays gracefully
                elif predicted.shape != actual.shape:
                    distance = float('nan')  # Handle shape mismatch
                else:
                    distance = np.abs(predicted - actual).mean()

                # Replace NaN with 0.0 for plotting purposes
                distance = 0.0 if np.isnan(distance) else distance

                # Store valid distances for 3D plot
                self.prediction_distances.append(distance)

                # Update the 3D plot
                epochs = list(range(1, len(self.prediction_distances) + 1))
                self.ax_3d.clear()
                self.ax_3d.set_xlabel('Epoch')
                self.ax_3d.set_ylabel('Accuracy')
                self.ax_3d.set_zlabel('Distance from Actual')
                self.ax_3d.set_title('3D Prediction Error vs Accuracy')

                # Scatter plot with prediction distances
                self.ax_3d.scatter(epochs, self.acc_data, self.prediction_distances, label='Error vs Accuracy', color='green')
                self.ax_3d.legend()

                # Redraw the 3D plot
                self.canvas_plot.draw()

        except Exception as e:
            logger.exception("An error occurred while updating the GUI.")
            messagebox.showerror("Error", f"An error occurred: {e}")

    def start_training(self):
        """Start training in a new thread."""
        self.stop_event.clear()
        threading.Thread(target=self.train_thread, daemon=True).start()

    def stop_training(self):
        """Stop the training process."""
        self.stop_event.set()

    def train_thread(self):
        """Training logic executed in a separate thread to avoid blocking the GUI."""
        logger.info("Training thread started.")

        # Optimizer, scheduler, and criterion setup for CNN
        optimizer_cnn = torch.optim.AdamW(self.model.parameters(), lr=0.01, weight_decay=1e-4)
        scheduler_cnn = torch.optim.lr_scheduler.StepLR(optimizer_cnn, step_size=10, gamma=0.1)
        criterion_cnn = nn.CrossEntropyLoss()

        # Define GAN optimizers and criterion
        optimizer_gen = torch.optim.Adam(self.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        optimizer_disc = torch.optim.Adam(self.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
        criterion_gan = nn.BCELoss()

        # Mixed precision scaler (if using CUDA)
        scaler_cnn = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None
        scaler_gan = torch.cuda.amp.GradScaler() if self.device.type == 'cuda' else None

        # Loop over epochs
        for epoch in range(self.total_epochs):
            if self.stop_event.is_set():
                logger.info("Training stopped by user.")
                break
            logger.info(f"Starting epoch {epoch + 1}/{self.total_epochs}.")
            self.model.train()  # Set the CNN model in training mode
            self.generator.train()
            self.discriminator.train()
            running_loss = 0.0
            correct = 0
            total = 0

            # Loop over batches
            for batch_idx, (inputs, targets) in enumerate(self.train_loader, 1):
                if self.stop_event.is_set():
                    logger.info("Training stopped by user.")
                    break
                try:
                    # -----------------
                    # Train CNN
                    # -----------------
                    # Move data to the appropriate device
                    inputs_cnn, targets_cnn = inputs.to(self.device), targets.to(self.device)

                    # Reset gradients
                    optimizer_cnn.zero_grad()

                    if self.device.type == 'cuda' and scaler_cnn is not None:
                        # Mixed precision training with autocast
                        with torch.cuda.amp.autocast(enabled=True):
                            outputs_cnn = self.model(inputs_cnn)
                            # Resize outputs to match targets
                            outputs_cnn = F.interpolate(outputs_cnn, size=targets_cnn.shape[1:], mode='bilinear', align_corners=False)
                            loss_cnn = criterion_cnn(outputs_cnn, targets_cnn)

                        # Backward pass and optimizer step with scaler
                        scaler_cnn.scale(loss_cnn).backward()
                        scaler_cnn.step(optimizer_cnn)
                        scaler_cnn.update()
                    else:
                        # Standard training without autocast
                        outputs_cnn = self.model(inputs_cnn)
                        # Resize outputs to match targets
                        outputs_cnn = F.interpolate(outputs_cnn, size=targets_cnn.shape[1:], mode='bilinear', align_corners=False)
                        loss_cnn = criterion_cnn(outputs_cnn, targets_cnn)

                        # Backward pass and optimizer step
                        loss_cnn.backward()
                        optimizer_cnn.step()

                    # Compute metrics
                    batch_loss = loss_cnn.item()
                    _, predicted = outputs_cnn.max(1)
                    correct_predictions = predicted.eq(targets_cnn).sum().item()
                    batch_accuracy = 100.0 * correct_predictions / targets_cnn.numel()

                    running_loss += batch_loss * inputs_cnn.size(0)
                    correct += correct_predictions
                    total += targets_cnn.numel()

                    # Update the GUI every 10 batches or at the end of epoch
                    if batch_idx % 10 == 0 or batch_idx == len(self.train_loader):
                        gui_batch_loss = running_loss / total
                        gui_batch_accuracy = 100.0 * correct / total
                        self.queue.put({
                            'batch': batch_idx,
                            'loss': gui_batch_loss,
                            'accuracy': gui_batch_accuracy
                        })

                except Exception as e:
                    logger.exception(f"Error in CNN batch {batch_idx}: {e}")
                    continue  # Continue with the next batch if an error occurs

            # -----------------
            # Train GAN
            # -----------------
            for batch_idx, (inputs_gan, _) in enumerate(self.train_loader, 1):
                if self.stop_event.is_set():
                    logger.info("Training stopped by user.")
                    break
                try:
                    real_images = inputs_gan.to(self.device)
                    batch_size = real_images.size(0)

                    # Labels
                    label_real = torch.ones(batch_size, device=self.device)
                    label_fake = torch.zeros(batch_size, device=self.device)

                    # Step 1: One-hot encode real images to have 11 channels
                    inputs_one_hot = torch.nn.functional.one_hot(real_images.squeeze(1).long(), num_classes=NUM_CLASSES)
                    inputs_one_hot = inputs_one_hot.permute(0, 3, 1, 2).float()  # [batch_size, 11, H, W]

                    # Step 2: Normalize to [-1, 1] to match Generator's output
                    inputs_one_hot = (inputs_one_hot - 0.5) * 2  # Normalize to [-1, 1]

                    # -----------------
                    #  Train Discriminator
                    # -----------------
                    optimizer_disc.zero_grad()

                    if self.device.type == 'cuda' and scaler_gan is not None:
                        with torch.cuda.amp.autocast(enabled=True):
                            # Real images
                            output_real = self.discriminator(inputs_one_hot)
                            d_loss_real = criterion_gan(output_real, label_real)

                            # Fake images
                            noise = torch.randn(batch_size, 100, device=self.device)
                            fake_images = self.generator(noise)  # [batch_size, 11, 32, 32]
                            output_fake = self.discriminator(fake_images.detach())
                            d_loss_fake = criterion_gan(output_fake, label_fake)

                            # Total Discriminator loss
                            d_loss = d_loss_real + d_loss_fake

                        # Backward pass and optimizer step with scaler
                        scaler_gan.scale(d_loss).backward()
                        scaler_gan.step(optimizer_disc)
                        scaler_gan.update()
                    else:
                        # Standard training without autocast
                        # Real images
                        output_real = self.discriminator(inputs_one_hot)
                        d_loss_real = criterion_gan(output_real, label_real)

                        # Fake images
                        noise = torch.randn(batch_size, 100, device=self.device)
                        fake_images = self.generator(noise)  # [batch_size, 11, 32, 32]
                        output_fake = self.discriminator(fake_images.detach())
                        d_loss_fake = criterion_gan(output_fake, label_fake)

                        # Total Discriminator loss
                        d_loss = d_loss_real + d_loss_fake

                        # Backward pass and optimizer step
                        d_loss.backward()
                        optimizer_disc.step()

                    # -----------------
                    #  Train Generator
                    # -----------------
                    optimizer_gen.zero_grad()

                    if self.device.type == 'cuda' and scaler_gan is not None:
                        with torch.cuda.amp.autocast(enabled=True):
                            output_fake = self.discriminator(fake_images)
                            g_loss = criterion_gan(output_fake, label_real)

                        # Backward pass and optimizer step with scaler
                        scaler_gan.scale(g_loss).backward()
                        scaler_gan.step(optimizer_gen)
                        scaler_gan.update()
                    else:
                        # Standard training without autocast
                        output_fake = self.discriminator(fake_images)
                        g_loss = criterion_gan(output_fake, label_real)

                        # Backward pass and optimizer step
                        g_loss.backward()
                        optimizer_gen.step()

                    # Log GAN losses periodically
                    if batch_idx % 50 == 0:
                        logger.info(f"[Epoch {epoch+1}/{self.total_epochs}] [GAN Batch {batch_idx}/{len(self.train_loader)}] "
                                    f"[D loss: {d_loss.item():.4f}] [G loss: {g_loss.item():.4f}]")

                except Exception as e:
                    logger.exception(f"Error in GAN batch {batch_idx}: {e}")
                    continue

            # -----------------
            # Sample Predictions for Visualization
            # -----------------
            try:
                sample_inputs, sample_targets = next(iter(self.val_loader))  # Get a batch from validation set
                sample_inputs = sample_inputs.to(self.device)
                with torch.no_grad():
                    sample_outputs = self.model(sample_inputs)
                    sample_outputs = F.interpolate(sample_outputs, size=sample_targets.shape[1:], mode='bilinear', align_corners=False)
                    _, sample_predictions = sample_outputs.max(1)
            except StopIteration:
                logger.warning("Validation loader is empty.")
                sample_predictions = torch.tensor([])
                sample_targets = torch.tensor([])

            # Send epoch updates to the GUI
            self.queue.put({
                'epoch': epoch + 1,
                'loss': running_loss / total,
                'accuracy': 100.0 * correct / total,
                'guesses': sample_predictions.cpu(),
                'actuals': sample_targets.cpu()
            })

            # Scheduler step
            scheduler_cnn.step()

        logger.info("Training completed.")
        self.queue.put({'status': 'Training Completed'})

    def train_batch(self, batch_idx, inputs, targets, optimizer, scaler, criterion):
        """Train a single batch."""
        # Move data to the appropriate device
        inputs, targets = inputs.to(self.device), targets.to(self.device)

        # Reset gradients
        optimizer.zero_grad()

        if self.device.type == 'cuda' and scaler is not None:
            # Mixed precision training with autocast
            with torch.cuda.amp.autocast(enabled=True):
                outputs = self.model(inputs)
                # Resize outputs to match targets
                outputs = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)
                loss = criterion(outputs, targets)

            # Backward pass and optimizer step with scaler
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without autocast
            outputs = self.model(inputs)
            # Resize outputs to match targets
            outputs = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)
            loss = criterion(outputs, targets)

            # Backward pass and optimizer step
            loss.backward()
            optimizer.step()

        # Compute metrics
        batch_loss = loss.item()
        _, predicted = outputs.max(1)
        correct_predictions = predicted.eq(targets).sum().item()
        batch_accuracy = 100.0 * correct_predictions / targets.numel()

        return batch_loss, batch_accuracy, targets.numel()

    def evaluate_model_button(self):
        """Evaluate the model in a new thread."""
        threading.Thread(target=self.evaluate_model, daemon=True).start()

    def evaluate_model(self):
        """Evaluate the model."""
        avg_loss, accuracy = evaluate_model(self.model, self.val_loader, self.device)
        messagebox.showinfo("Evaluation", f"Validation Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

# -----------------------------
# 10. Training Function with GUI Integration
# -----------------------------

def train_model_with_gui(model, train_loader, val_loader, device, gui):
    """Train the model and update the GUI in real-time."""
    try:
        # Start the GUI training display
        gui.start_training()

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        gui.queue.put({'error': str(e)})  # Inform the GUI about the error

# -----------------------------
# 11. Evaluation Function
# -----------------------------

def evaluate_model(model, test_loader, device='cpu'):
    """
    Evaluates the model on the test dataset.

    Args:
        model (nn.Module): Trained model.
        test_loader (DataLoader): DataLoader for the test dataset.
        device (str): Device to run evaluation on.

    Returns:
        tuple: (average_loss, accuracy)
    """
    criterion = nn.CrossEntropyLoss()  # Loss function
    model.to(device)  # Move model to the appropriate device
    model.eval()  # Set the model to evaluation mode

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Forward pass
            outputs = model(inputs)
            # Resize outputs to match targets
            outputs = F.interpolate(outputs, size=targets.shape[1:], mode='bilinear', align_corners=False)

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item() * inputs.size(0)  # Accumulate weighted loss

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == targets).sum().item()
            total += targets.numel()

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(test_loader.dataset)
    accuracy = correct / total

    logger.info(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.4f}")
    return avg_loss, accuracy

# -----------------------------
# 12. Traverse and Debug Function
# -----------------------------

def traverse_and_debug(node):
    """Traverse the tree and log node details with reduced verbosity."""
    # Limit logging to fewer nodes
    if hasattr(node, "name"):
        grid_shape = getattr(node, 'embedding', None)
        grid_shape = grid_shape.shape if grid_shape is not None else 'Missing'
        logger.debug(f"Node: {node.name}, Grid Shape: {grid_shape}")
    if len(node.children) > 10:  # Avoid excessive logging if many children exist
        logger.warning(f"Node '{node.name}' has too many children, skipping further logs...")
        return
    for child in node.children:
        traverse_and_debug(child)

# -----------------------------
# 13. Main Workflow with Modifications
# -----------------------------

def main():
    # Detect device
    device = get_device()

    # Initialize the progress bar
    progress_bar = tqdm(total=100, desc="Loading Data", unit="%", leave=True)

    try:
        # Load ARC data
        arc_data = load_arc_data()
        progress_bar.update(20)

        # Extract and reshape grid pairs
        train_grid_pairs = flatten_and_reshape(
            arc_data.get("arc-agi_training-challenges", {})
        )
        eval_grid_pairs = flatten_and_reshape(
            arc_data.get("arc-agi_evaluation-challenges", {})
        )
        progress_bar.update(30)

        # Build the data tree and retrieve the task dictionary
        root_node, task_dict = build_data_tree(train_grid_pairs)
        traverse_and_debug(root_node)
        progress_bar.update(20)

        # Log the task dictionary
        logger.info(f"Task dictionary initialized with {len(task_dict)} tasks:")
        for task_id, task_data in task_dict.items():
            logger.info(
                f"Task ID: {task_id}, Node: {task_data['task_node'].name}, "
                f"Grid Shape: {task_data['grids'][0].shape}"
            )

        # Initialize DataLoaders and models
        train_dataset = AugmentedARCDataset(train_grid_pairs, augment=False)
        val_dataset = AugmentedARCDataset(eval_grid_pairs, augment=False)

        train_loader = DataLoader(
            train_dataset,
            batch_size=4,  # Reduce batch size if you encounter memory issues
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn  # Use the custom collate function
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=4,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
            collate_fn=collate_fn
        )

        # Initialize the CNN model
        model = CNNGridMapper(num_classes=NUM_CLASSES).to(device)
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        logger.info("CNN Model initialized successfully.")

        # Initialize GAN models
        generator = Generator(latent_dim=100, output_channels=NUM_CLASSES, grid_size=(FIXED_HEIGHT, FIXED_WIDTH)).to(device)
        discriminator = Discriminator(input_channels=NUM_CLASSES, grid_size=(FIXED_HEIGHT, FIXED_WIDTH)).to(device)
        if torch.cuda.device_count() > 1:
            generator = nn.DataParallel(generator)
            discriminator = nn.DataParallel(discriminator)
        logger.info("GAN Models initialized successfully.")

    except Exception as e:
        logger.exception(f"Data loading or model initialization failed: {e}")
        progress_bar.close()
        return

    progress_bar.close()

    # Initialize and start the GUI
    root_window = tk.Tk()
    root_window.title("Training Progress Visualization")
    gui = TrainingGUI(
        root_window, total_epochs=10, total_batches=len(train_loader),
        model=model, train_loader=train_loader, val_loader=val_loader,
        eval_loader=None, device=device, data_tree=root_node, task_dict=task_dict,
        generator=generator, discriminator=discriminator  # Pass GAN models
    )

    # Start the training thread
    training_thread = threading.Thread(
        target=train_model_with_gui, args=(model, train_loader, val_loader, device, gui)
    )
    training_thread.daemon = True
    training_thread.start()

    # Start the GUI main loop
    root_window.mainloop()


if __name__ == "__main__":
    main()
