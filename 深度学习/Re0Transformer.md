### **构建总纲：使用PyTorch从零构建文本生成Transformer**

我们的目标是构建一个能够根据输入文本（prompt）生成新文本序列的Transformer模型。为了实现这个目标，我们将分步完成以下几个核心模块的构建和学习，每个模块都会作为后续笔记中的一个独立部分进行详细阐述：

**第一部分：项目概述与环境准备**
*   **目标定义：** 明确我们要构建的是一个什么样的模型（例如，一个基于字符或单词的生成模型），以及它的基本工作原理。
*   **环境搭建：** 介绍所需要安装的Python库（`torch`, `torchtext`, `matplotlib`, `tqdm`, `numpy`等），并展示如何导入它们。
*   **全局设置：** 设置设备（GPU或CPU），并配置随机种子以确保实验的可复现性。

**第二部分：数据处理与管道搭建**
*   **数据源：** 选择一个合适的文本数据集作为我们模型的训练材料。
*   **文本分词 (Tokenization)：** 将原始的文本字符串分解成模型可以理解的最小单元（Token），例如单词或字符。
*   **构建词汇表 (Vocabulary)：** 创建一个从Token到数字索引（index）的双向映射。这是将文本数据转换为数字格式的关键一步。
*   **数据集类 (Dataset Class)：** 使用PyTorch的`Dataset`类来封装我们的文本数据，定义获取单个数据样本的逻辑。
*   **数据加载器 (DataLoader)：** 利用PyTorch的`DataLoader`来高效地实现数据的批量化（Batching）、填充（Padding）和打乱（Shuffling），为模型训练做好最终准备。

**第三部分：Transformer模型架构详解**
*   **核心理念：** 简要回顾Transformer相比于RNN/LSTM在处理序列数据上的优势，特别是其并行计算能力。
*   **位置编码 (Positional Encoding)：** 详细解释为什么Transformer需要位置编码来理解序列的顺序信息，并展示其数学实现。
*   **多头自注意力机制 (Multi-Head Self-Attention)：** 这是Transformer的心脏。我们将从单个头的注意力（Scaled Dot-Product Attention）讲起，逐步构建到多头注意力，并解释其如何让模型关注序列中的不同部分。
*   **前馈神经网络 (Feed-Forward Network)：** 解释在每个Transformer块中注意力层之后的前馈网络的作用。
*   **构建解码器模块 (Decoder Block)：** 将位置编码、多头自注意力（带有掩码）、前馈网络和层归一化（Layer Normalization）等组件组合成一个完整的Transformer解码器块。
*   **完整的生成模型：** 将多个解码器块堆叠起来，并添加词嵌入层（Embedding Layer）和最终的线性输出层，构成我们最终的文本生成模型。

**第四部分：训练模块的构建**
*   **定义超参数：** 设置学习率、批次大小、训练轮数（Epochs）、模型维度等所有可调参数。
*   **实例化组件：** 创建模型、损失函数（如交叉熵损失 `CrossEntropyLoss`）和优化器（如 `Adam`）的实例。
*   **训练循环 (Training Loop)：** 编写核心的训练逻辑。这包括：
    *   从DataLoader获取数据批次。
    *   将数据送入模型进行前向传播。
    *   计算损失。
    *   执行反向传播和梯度下降。
    *   清空梯度。
*   **评估循环 (Evaluation Loop)：** 编写在验证集上评估模型性能的逻辑，此过程不计算梯度以节省资源。

**第五部分：实时训练过程可视化**
*   **目标：** 监控模型在训练过程中的性能变化，例如损失函数值的下降趋势。
*   **工具：** 主要使用 `matplotlib` 库。
*   **实现：** 在每个训练轮次（Epoch）结束后，记录下训练集和验证集的平均损失，并使用 `matplotlib` 动态绘制损失曲线图，帮助我们直观地判断模型是否在有效学习，以及是否存在过拟合现象。

**第六部分：文本生成（推理）模块**
*   **核心任务：** 如何使用训练好的模型来生成新的文本。
*   **生成逻辑：** 编写一个函数，接收一个起始文本（prompt）作为输入。
*   **解码策略：**束搜索


**第七部分：模型保存、加载与整合**
*   **持久化：** 学习如何使用 `torch.save` 保存训练好的模型权重，以及如何使用 `torch.load` 来加载它们，以便未来直接使用或继续训练。
*   **项目整合：** 将以上所有模块（数据处理、模型定义、训练、生成）整合到一个主脚本中，形成一个完整、可执行的文本生成项目。

---

### **第一部分：项目概述与环境准备**

在这一部分，我们将为整个项目打下基础。我们会明确项目的目标，安装并导入所有必需的Python库，并进行一些全局设置，以确保代码的稳定性和可复现性。

#### **1. 目标定义 (Project Goal)**

我们的核心目标是利用PyTorch构建一个基于Transformer架构的文本生成模型。具体来说，我们将构建一个 **仅包含解码器（Decoder-only）** 的Transformer模型，这与GPT（Generative Pre-trained Transformer）系列的早期架构类似。

该模型将以 **字符级别（Character-level）** 的方式学习文本数据。这意味着模型学习的最小单位是单个字符，而不是单词。这样做的好处是词汇表规模小且固定，不会遇到“未知词”（Out-of-Vocabulary）问题，非常适合用于演示和学习核心概念。

模型的任务是：给定一段起始文本（称为"prompt"），模型需要预测下一个最可能的字符，然后将这个新生成的字符加入到输入中，再次预测下一个，如此循环，最终生成一段完整的、风格与训练数据相似的新文本。

#### **2. 环境搭建 (Environment Setup)**

首先，我们需要导入将要使用的所有Python库。这些库各司其职，共同支撑起整个项目的框架。

```python
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm
```

上述代码导入了我们项目所需的全部基本库：
*   `torch` 和 `torch.nn`: 这是PyTorch的核心，提供了张量（Tensors）计算和神经网络层（如线性层、嵌入层等）的构建模块。
*   `torch.utils.data`: 提供了 `Dataset` 和 `DataLoader` 这两个强大的工具，它们将帮助我们构建高效的数据处理管道。
*   `numpy`: 虽然PyTorch有自己的张量，但NumPy在数据预处理和某些数学运算上仍然非常方便，并且它们之间可以轻松转换。
*   `math`: 用于一些基础的数学计算，例如在位置编码中会用到。
*   `random`: 用于设置随机种子。
*   `matplotlib.pyplot`: 这是Python中最流行的绘图库，我们将用它来实时绘制损失曲线，监控训练过程。
*   `tqdm`: 一个非常方便的库，可以为我们的训练循环创建一个可视化的进度条，让我们能直观地看到训练的进度。

#### **3. 全局设置 (Global Settings)**

这部分代码用于配置我们的运行环境，主要做两件事：选择计算设备（CPU或GPU）和设定随机种子以保证实验的可复现性。

```python
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
```

代码解释：
*   `set_seed(seed)` 函数：我们定义了一个函数来统一设置随机种子。 在深度学习实验中，很多操作都具有随机性，比如模型权重的初始化、数据的打乱（shuffle）等。 如果不固定随机种子，每次运行同样的代码都可能会得到略微不同的结果，这不利于调试和比较不同实验的效果。 这个函数为`random`、`numpy`和`torch`（包括CPU和GPU）都设置了相同的种子。
*   `torch.backends.cudnn.deterministic = True` 和 `torch.backends.cudnn.benchmark = False`: 这两行是针对使用NVIDIA GPU（通过CUDA和cuDNN库）时的额外设置。第一行确保cuDNN使用确定性的卷积算法，第二行则关闭了cuDNN的自动调优功能（该功能会根据输入尺寸选择最快的算法，但可能导致不确定性）。
*   `set_seed(42)`: 我们选择一个具体的数字（42是一个常用的惯例）作为我们的种子并调用函数。
*   `device = torch.device(...)`: 这行代码会自动检测当前环境是否有可用的NVIDIA GPU。如果有（`torch.cuda.is_available()` 返回 `True`），它就会将 `device` 设置为 "cuda"，这样后续的张量和模型都可以被发送到GPU上进行高速计算。否则，它将使用 "cpu"。
*   最后，我们打印出当前使用的设备，以便确认我们的设置是否生效。



---

### **第二部分：数据处理与管道搭建**

在这一部分，我们将把原始的文本字符串，通过分词、构建词汇表、定义数据集和使用数据加载器等一系列步骤，转换为PyTorch模型可以直接使用的、批量化的张量（Tensors）。

#### **1. 数据源与预处理 (Data Source & Pre-processing)**

首先，我们需要一份用于训练的文本数据。为了让笔记清晰易懂，我们不使用庞大复杂的数据集，而是直接在代码中定义一小段文本。这样可以让我们完全聚焦于处理流程本身。在实际项目中，您可以轻松地将这里的文本替换为您从文件中读出的任何内容。

```python
# For demonstration purposes, we use a small, simple text.
# In a real-world scenario, you would load this from a file.
raw_text = """
It is not the critic who counts; not the man who points out how the strong man stumbles, 
or where the doer of deeds could have done them better. 
The credit belongs to the man who is actually in the arena, whose face is marred by dust and sweat and blood; 
who strives valiantly; who errs, who comes short again and again, 
because there is no effort without error and shortcoming; 
but who does actually strive to do the deeds; who knows the great enthusiasms, the great devotions; 
who spends himself in a worthy cause; 
who at the best knows in the end the triumph of high achievement, 
and who at the worst, if he fails, at least fails while daring greatly, 
so that his place shall never be with those cold and timid souls who neither know victory nor defeat.
"""

# Basic preprocessing: convert to lowercase to reduce vocabulary size
raw_text = raw_text.lower()
```

代码解释：
*   我们定义了一个多行字符串 `raw_text` 作为我们的全部训练数据。
*   `raw_text = raw_text.lower()`: 这是一个简单但非常有效的预处理步骤。通过将所有文本转换为小写，我们减少了词汇表的大小（例如，"The" 和 "the" 会被视为同一个词/字符集的一部分），这有助于模型更快地学习。

#### **2. 分词与构建词汇表 (Tokenization & Vocabulary)**

由于我们构建的是一个字符级别的模型，所以“分词”（Tokenization）的过程就是获取文本中所有独特的字符。然后，我们需要创建一个“词汇表”（Vocabulary），它本质上是两个映射字典：一个将每个独特字符映射到一个唯一的整数索引，另一个则反向映射。

```python
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}

encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

print(f"Vocabulary: {''.join(chars)}")
print(f"Vocabulary size: {vocab_size}")

# Example of encoding and decoding
encoded_sample = encode("hello world")
decoded_sample = decode(encoded_sample)
print(f"Sample text: 'hello world'")
print(f"Encoded sample: {encoded_sample}")
print(f"Decoded sample: '{decoded_sample}'")
```
代码解释：
*   `chars = sorted(list(set(raw_text)))`: 这一行代码完成了两件事。`set(raw_text)` 会找出文本中所有的不重复字符。`list()` 将其转换为列表，`sorted()` 对其进行排序。排序是为了保证每次运行代码时，字符到索引的映射都是固定的，这对于实验的可复现性至关重要。
*   `vocab_size = len(chars)`: 词汇表的大小就是我们独特字符的数量。这个数值对于定义模型嵌入层和输出层的维度至关重要。
*   `stoi` (string-to-integer) 和 `itos` (integer-to-string): 我们使用字典推导式创建了两个映射字典。`stoi` 用于将字符转换为数字，`itos` 用于将数字转换回字符。
*   `encode` 和 `decode`: 我们定义了两个辅助的lambda函数。`encode` 接收一个字符串，返回一个由整数索引组成的列表。`decode` 接收一个整数列表，返回原始的字符串。这是后续处理数据和检视模型输出时非常方便的工具。

#### **3. 创建数据集类 (PyTorch Dataset Class)**

现在，我们需要一种标准的方式来组织数据，告诉PyTorch如何获取单个训练样本。PyTorch通过 `torch.utils.data.Dataset` 类来实现这一点。我们需要创建一个子类，并实现两个核心方法：`__len__` 和 `__getitem__`。

对于语言模型训练，一个样本通常包含两部分：一个输入序列 `x` 和一个目标序列 `y`。`y` 是 `x` 向右移动一个位置的结果。例如，如果输入 `x` 是 "hello"，那么模型的目标 `y` 就是 "ello"。

```python
# Hyperparameters for data processing
block_size = 8 

class TextDataset(Dataset):
    def __init__(self, text, block_size):
        self.block_size = block_size
        self.data = torch.tensor(encode(text), dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

# Instantiate the dataset
dataset = TextDataset(raw_text, block_size)
print(f"Total number of samples in the dataset: {len(dataset)}")
sample_x, sample_y = dataset[0]
print(f"First sample input (x): {sample_x}")
print(f"First sample target (y): {sample_y}")
print(f"Decoded input: '{decode(sample_x.tolist())}'")
print(f"Decoded target: '{decode(sample_y.tolist())}'")
```

代码解释：
*   `block_size = 8`: 这是一个重要的超参数，也称为“上下文长度”（context length）或“序列长度”（sequence length）。它定义了模型在做一次预测时，最多能“看到”多长的历史文本。这里我们设为8，意味着模型将根据前8个字符来预测第9个字符。
*   `TextDataset(Dataset)`: 我们定义了一个继承自 `torch.utils.data.Dataset` 的类。
*   `__init__(self, text, block_size)`: 构造函数接收原始文本和 `block_size`。它做的最重要的一件事就是调用我们之前定义的 `encode` 函数，将**全部**文本一次性地转换为一个长长的PyTorch张量 `self.data`。
*   `__len__(self)`: 这个方法必须返回数据集中样本的总数。由于我们使用滑动窗口的方式创建样本，最后一个可能的起始位置是 `len(self.data) - block_size`。
*   `__getitem__(self, idx)`: 这是最核心的方法。当我们需要获取第 `idx` 个样本时，这个方法会被调用。
    *   `x = self.data[idx:idx+self.block_size]`: 它从编码后的数据中切片，提取出长度为 `block_size` 的一段作为输入 `x`。
    *   `y = self.data[idx+1:idx+self.block_size+1]`: 它提取从 `x` 的第二个位置开始，同样长度的一段作为目标 `y`。这样就构成了 (输入 -> 目标) 的训练对。

#### **4. 创建数据加载器 (PyTorch DataLoader)**

虽然我们有了`Dataset`，但在训练时，我们通常希望分批（batch）次地将数据喂给模型，并且在每个训练周期（epoch）开始时打乱数据顺序。`torch.utils.data.DataLoader` 正是为此而生。

```python
# Hyperparameters for data loading
batch_size = 4

train_loader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=True
)

# Let's inspect a single batch from the loader
data_iter = iter(train_loader)
first_batch_x, first_batch_y = next(data_iter)

print(f"Shape of a single batch input (X): {first_batch_x.shape}")
print(f"Shape of a single batch target (Y): {first_batch_y.shape}")
```

代码解释：
*   `batch_size = 4`: 这是另一个重要的超参数，定义了我们一次性将多少个样本打包在一起送入模型进行训练。
*   `DataLoader(...)`: 我们实例化了一个 `DataLoader`。
    *   `dataset=dataset`: 告诉加载器从哪个`Dataset`对象中取数据。
    *   `batch_size=batch_size`: 设置每个批次的大小。
    *   `shuffle=True`: 这是非常关键的一步。设置为 `True` 意味着在每个epoch开始前，`DataLoader` 都会随机打乱数据的顺序。这可以防止模型学到数据本身的排列顺序，增强了模型的泛化能力。
*   `data_iter = iter(train_loader)` 和 `first_batch_x, first_batch_y = next(data_iter)`: 这两行代码模拟了在训练循环中从`DataLoader`获取一个批次数据的过程。
*   我们打印出这个批次数据的形状。可以看到，输入 `X` 的形状是 `[4, 8]`（`[batch_size, block_size]`），这正是模型所期望的输入格式。

---

至此，我们已经成功地建立了一个完整且高效的数据处理管道。我们从一个原始的文本字符串开始，最终得到了一个可以持续提供打乱且批量化的`[输入, 目标]`张量对的`DataLoader`。这是模型训练前至关重要的一步。

好的，感谢您的鼓励！我们现在进入最核心、最激动人心的部分。在这里，我们将从零开始，一块一块地拼接起强大的Transformer模型。我会尽可能地拆解每一个概念，让其变得直观易懂。

---

### **第三部分：Transformer模型架构详解**

在这一部分，我们将深入探索构成Transformer模型的各个组件，并使用PyTorch将它们一一实现。我们的目标是构建一个仅包含解码器（Decoder-only）的Transformer，它非常适合文本生成任务。

#### **1. 核心理念回顾**

与循环神经网络（RNN）一次处理一个时间步的序列信息不同，Transformer的核心是 **自注意力机制（Self-Attention）**。这种机制允许模型在处理序列中的任何一个词（或字符）时，能够同时直接计算它与序列中所有其他词的关联强度，从而捕捉长距离依赖关系。这种并行计算的能力是Transformer相比RNN的主要优势之一，极大地提高了训练效率。

#### **2. 词嵌入与位置编码 (Token Embedding & Positional Encoding)**

计算机无法直接理解字符，因此第一步是需要将输入的字符索引转换为高维度的向量表示，这称为 **词嵌入（Token Embedding）**。PyTorch中的 `nn.Embedding` 层可以为我们完成这个任务。

然而，仅有词嵌入是不够的。自注意力机制本身不包含任何关于序列顺序的信息，它平等地看待所有位置的词。为了让模型理解 "A followed by B" 和 "B followed by A" 的区别，我们必须额外注入位置信息。这就是 **位置编码（Positional Encoding）** 的作用。我们将其与词嵌入相加，共同作为模型的输入。

```python
# Hyperparameters for the model
d_model = 64 # Dimension of the model embedding

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x
```

代码解释：
*   我们创建了一个名为 `PositionalEncoding` 的模块。它不包含任何可训练的参数，其作用是根据固定的数学公式（正弦和余弦函数）为每个位置生成一个独特的编码向量。
*   `__init__` 方法：
    *   `pe = torch.zeros(max_len, d_model)`: 创建一个足够大的、形状为 `[最大序列长度, 模型维度]` 的零矩阵，用于存储位置编码。
    *   `position` 和 `div_term`: 这两行代码正在构建计算正弦和余弦函数的公式组件。这种公式设计巧妙，能让模型轻易地学习到相对位置信息。
    *   `pe[:, 0::2]` 和 `pe[:, 1::2]`: 分别计算偶数维度和奇数维度的位置编码值。
    *   `self.register_buffer('pe', pe)`: 这是关键的一步。`register_buffer` 告诉PyTorch，`pe` 是模型的一个固定状态，它应该被包含在模型的`state_dict`中（这样保存和加载模型时它也会被保存），但它不是一个需要计算梯度的模型参数。
*   `forward(self, x)` 方法：
    *   它接收词嵌入后的张量 `x`（形状为 `[batch_size, seq_len, d_model]`）。
    *   `self.pe[:, :x.size(1), :]`: 从我们预先计算好的 `pe` 矩阵中，取出与当前输入序列长度相匹配的部分。
    *   `x = x + ...`: 将位置编码直接加到词嵌入上，从而将位置信息注入到输入向量中。

#### **3. 核心组件：带掩码的多头自注意力 (Masked Multi-Head Self-Attention)**

这是Transformer的心脏。它让模型在生成第 `t` 个字符时，能够关注到从第 `1` 个到第 `t` 个字符的所有信息，并动态地为它们分配不同的“注意力权重”。

*   **自注意力（Self-Attention）**：对于序列中的每一个输入向量，我们通过三个独立的线性变换，创建出三个新的向量：**查询（Query）**、**键（Key）** 和 **值（Value）**。通过计算一个查询向量和所有键向量的点积，我们可以得到该查询与每个键的相似度分数。这些分数经过缩放和Softmax归一化后，就成了注意力权重。最后，将这些权重与对应的值向量加权求和，就得到了该位置的最终输出。
*   **掩码（Masking）**：在文本生成任务中，模型在预测下一个词时，**不能看到未来的词**。因此，我们需要一个“掩码”来阻止这种情况。这个掩码是一个下三角矩阵，它会强制将所有未来位置的注意力分数设为一个极小的负数，这样在经过Softmax后，它们的权重就几乎为零。
*   **多头（Multi-Head）**：我们不只做一次注意力计算，而是将模型维度 `d_model` 分割成多个“头”（`n_heads`）。每个头独立地执行上述的自注意力计算。这允许模型在不同的“表示子空间”中，同时关注不同方面的信息。最后，再将所有头的输出拼接起来，通过一个线性层进行整合。

```python
# More model hyperparameters
n_heads = 4
d_k = d_model // n_heads # Dimension of key/query per head

class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.d_k = d_model // n_heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)

    def forward(self, x, mask):
        batch_size = x.size(0)
        
        q = self.q_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_linear(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = torch.softmax(scores, dim=-1)
        
        context = torch.matmul(attention_weights, v)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        output = self.out_linear(context)
        return output
```

代码解释：
*   `__init__`: 初始化了四个线性层。前三个用于从输入 `x` 生成Q, K, V，最后一个 (`out_linear`) 用于整合多头注意力的输出。
*   `forward(self, x, mask)`:
    *   `q = self.q_linear(x).view(...)`: 输入 `x` 经过线性变换后，通过 `.view()` 和 `.transpose()` 操作，被重塑为 `[batch_size, n_heads, seq_len, d_k]` 的形状，将不同的头分离开。K和V也做同样的操作。
    *   `scores = torch.matmul(...) / math.sqrt(self.d_k)`: 计算Q和K的点积，得到注意力分数。除以 `sqrt(d_k)` 是一个缩放步骤，可以防止梯度在训练初期过小。
    *   `scores.masked_fill(mask == 0, float('-inf'))`: 这就是应用掩码的地方。我们将掩码中值为0的位置（即未来位置）的注意力分数设置为负无穷。
    *   `attention_weights = torch.softmax(scores, dim=-1)`: 对分数进行softmax，得到0到1之间的权重。被掩码的位置权重会变为0。
    *   `context = torch.matmul(attention_weights, v)`: 将权重与V进行加权求和。
    *   `context.transpose(...).contiguous().view(...)`: 将多头的结果重新拼接成 `[batch_size, seq_len, d_model]` 的形状。
    *   `output = self.out_linear(context)`: 通过最后的线性层，得到该模块的最终输出。

#### **4. 其他关键组件：前馈网络与残差连接**

*   **位置相关前馈网络 (Position-wise Feed-Forward Network)**：在每个注意力层之后，都会跟一个简单的前馈网络。它对序列中的每一个位置的向量独立地进行一次非线性变换，增加了模型的表达能力。它通常由两个线性层和它们之间的一个ReLU激活函数组成。
*   **残差连接 (Residual Connection) 与层归一化 (Layer Normalization)**：这是训练深度网络的关键技巧。在每个子模块（如多头注意力和前馈网络）的周围，我们都使用一个残差连接，即把模块的输入直接加到模块的输出上 (`x + Sublayer(x)`)。这可以有效防止梯度消失，让网络更容易训练。在残差连接之后，我们再进行一次层归一化 (`LayerNorm`)，它能稳定训练过程，加速收敛。

```python
# More model hyperparameters
dropout_rate = 0.1

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff=2048, dropout=dropout_rate):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = torch.relu(self.linear_1(x))
        x = self.dropout(x)
        x = self.linear_2(x)
        return x

class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, dropout=dropout_rate):
        super().__init__()
        self.attention = MultiHeadAttention(n_heads, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn_output = self.attention(x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x
```

代码解释：
*   `FeedForward`: 简单实现了 `线性层 -> ReLU -> Dropout -> 线性层` 的结构。
*   `DecoderBlock`: 这是构成我们模型的基本单元。它清晰地展示了信息的流动过程：
    1.  输入 `x` 首先进入多头注意力层 (`self.attention`)。
    2.  注意力层的输出经过Dropout后，与原始输入 `x` 相加（残差连接），然后进行层归一化 (`self.norm1`)。
    3.  上一步的结果进入前馈网络 (`self.ff`)。
    4.  前馈网络的输出经过Dropout后，再次与它的输入相加（第二个残差连接），然后进行第二次层归一化 (`self.norm2`)。

#### **5. 组装完整的生成模型**

万事俱备，现在我们可以将所有组件组装成一个完整的语言模型了。

```python
# Final model hyperparameters
n_layers = 3 # Number of DecoderBlocks to stack

class LanguageModel(nn.Module):
    def __init__(self, vocab_size, d_model, n_layers, n_heads):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout_rate)
        
        self.decoder_blocks = nn.ModuleList([DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        
        self.final_norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(x.device)

        x = self.token_embedding(x)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        for block in self.decoder_blocks:
            x = block(x, mask)
            
        x = self.final_norm(x)
        logits = self.lm_head(x)
        
        return logits
```

代码解释：
*   `__init__`:
    *   `self.token_embedding`: 词嵌入层，将输入的字符索引转换为向量。
    *   `self.positional_encoding`: 我们之前实现的位置编码模块。
    *   `self.decoder_blocks = nn.ModuleList(...)`: 这是模型的“主体”。我们使用 `nn.ModuleList` 来堆叠 `n_layers` 个 `DecoderBlock`。
    *   `self.final_norm`: 在输出到最后一步之前，再进行一次层归一化。
    *   `self.lm_head`: 这是一个线性层，它的作用是将模型最终输出的 `d_model` 维度的向量，投影回 `vocab_size` 维度。这样，输出的每个位置上的向量，其每一个元素就对应着词汇表中一个字符的“得分”（logit）。
*   `forward(self, x)`:
    *   `mask = torch.tril(...)`: 在这里，我们为输入的序列动态地创建了 causal mask。`torch.tril` 会创建一个下三角矩阵。
    *   `x = self.token_embedding(x)`: 输入 `x` (形状 `[batch, seq_len]`) 首先通过嵌入层。
    *   `x = self.positional_encoding(x)`: 然后加上位置编码。
    *   `for block in self.decoder_blocks:`: 将数据依次送入我们堆叠的每一个`DecoderBlock`中进行处理。
    *   `logits = self.lm_head(self.final_norm(x))`: 最后的输出经过归一化和线性投影，得到最终的 `logits`（形状 `[batch, seq_len, vocab_size]`）。

---

我们已经成功地从最基础的组件开始，构建了一个完整且功能强大的Transformer语言模型！每一步的实现都对应着Transformer架构图中的一个部分。

### **第四部分：训练模块的构建**

在这一部分，我们将编写驱动模型学习的代码。这包括设置训练所需的超参数，实例化我们之前定义的模型、损失函数和优化器，并构建核心的训练与评估循环。

#### **1. 定义训练超参数与实例化组件**

在开始训练之前，我们需要定义一些关键的超参数，例如学习率和训练的总轮数。然后，我们将创建模型、损失函数和优化器的实例。

```python
# Training Hyperparameters
learning_rate = 3e-4
num_epochs = 100

# Instantiate the model
model = LanguageModel(
    vocab_size=vocab_size,
    d_model=d_model,
    n_layers=n_layers,
    n_heads=n_heads
)
model.to(device)

# Define the loss function and the optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Print model size
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"The model has {num_params:,} trainable parameters.")
```

代码解释：
*   `learning_rate`: 学习率控制了模型参数在每次更新时调整的幅度。这是一个需要仔细调整的关键超参数。`3e-4` (即0.0003) 是训练Transformer时一个常用且效果不错的初始值。
*   `num_epochs`: 定义了我们将完整地遍历训练数据集多少次。
*   `model = LanguageModel(...)`: 我们使用之前定义的模型超参数（`vocab_size`, `d_model`等）来创建`LanguageModel`类的一个实例。
*   `model.to(device)`: 这一步至关重要，它将模型的所有参数和缓冲区移动到我们之前设置的设备上（GPU或CPU）。为了让计算在GPU上进行，模型和数据必须在同一个设备上。
*   `criterion = nn.CrossEntropyLoss()`: 我们选择交叉熵损失作为我们的损失函数。这对于多分类问题（我们在这里的每一步都是在`vocab_size`个可能的字符中选择一个）是标准的选择。`nn.CrossEntropyLoss`在内部会自动帮我们处理`softmax`和计算负对数似然损失，因此我们模型的输出只需要是原始的`logits`即可。
*   `optimizer = torch.optim.Adam(...)`: 我们选择Adam优化器，它是一种高效且广泛使用的梯度下降算法。我们将模型的`model.parameters()`传递给它，告诉它需要更新哪些参数，并指定了学习率。
*   最后，我们计算并打印了模型的总参数量，这有助于我们了解模型的规模。

#### **2. 训练循环 (Training Loop)**

这是项目的心脏跳动的地方。我们将定义一个函数，它负责执行一个完整的训练轮次（epoch）。

```python
def train(model, data_loader, criterion, optimizer, device):
    model.train()
    epoch_loss = 0
    
    progress_bar = tqdm(data_loader, desc="Training")
    
    for batch_x, batch_y in progress_bar:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        
        logits = model(batch_x)
        
        B, T, C = logits.shape
        logits_flat = logits.view(B*T, C)
        targets_flat = batch_y.view(B*T)
        
        loss = criterion(logits_flat, targets_flat)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return epoch_loss / len(data_loader)
```

代码解释：
*   `def train(...)`: 我们将训练逻辑封装在一个函数中，使其清晰且可重用。
*   `model.train()`: 这是PyTorch的一个重要模式切换。调用它会告诉模型正处于“训练模式”。这会启用像Dropout这样的层，这些层在训练和评估时的行为是不同的。
*   `progress_bar = tqdm(...)`: 我们用`tqdm`包装`data_loader`，这样在迭代时就会显示一个漂亮的进度条。
*   `for batch_x, batch_y in progress_bar:`: 循环从数据加载器中获取每一个批次的数据。
*   `batch_x, batch_y = batch_x.to(device), batch_y.to(device)`: 将当前批次的数据也移动到与模型相同的设备上。
*   `optimizer.zero_grad()`: 在计算新一轮的梯度之前，必须清空上一轮的梯度。否则，梯度会累积。
*   `logits = model(batch_x)`: **前向传播**。将输入数据送入模型，得到模型的预测输出`logits`。
*   `logits_flat = logits.view(B*T, C)` 和 `targets_flat = batch_y.view(B*T)`: 这是非常关键的一步。`CrossEntropyLoss`期望的输入格式是：`logits`的形状为 `[N, C]`，`targets`的形状为 `[N]`，其中`N`是样本总数，`C`是类别数。我们模型的输出`logits`形状是 `[Batch, Time, Classes]`，目标`batch_y`的形状是 `[Batch, Time]`。因此，我们用`.view()`方法将它们“展平”，以符合损失函数的要求。
*   `loss = criterion(logits_flat, targets_flat)`: 计算模型预测和真实目标之间的损失。
*   `loss.backward()`: **反向传播**。PyTorch会自动计算损失相对于模型所有可训练参数的梯度。
*   `optimizer.step()`: **参数更新**。优化器根据计算出的梯度来更新模型的权重。
*   `epoch_loss += loss.item()`: `.item()`方法可以从一个只包含单个值的张量中提取出Python数值。我们累加每个批次的损失。
*   `return epoch_loss / len(data_loader)`: 返回这个轮次的平均损失。

#### **3. 评估循环 (Evaluation Loop)**

评估循环与训练循环非常相似，但有几个关键区别：它不计算梯度，也不更新模型参数。它的唯一目的是在当前模型状态下，衡量模型在数据上的表现。

*在实际项目中，我们通常会有一个独立的验证数据集（validation set）来执行评估，以获得模型泛化能力的无偏估计。为了简化，我们这里将在同一个训练数据上进行评估，但会遵循正确的评估流程。*

```python
def evaluate(model, data_loader, criterion, device):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            logits = model(batch_x)
            
            B, T, C = logits.shape
            logits_flat = logits.view(B*T, C)
            targets_flat = batch_y.view(B*T)
            
            loss = criterion(logits_flat, targets_flat)
            
            epoch_loss += loss.item()
            
    return epoch_loss / len(data_loader)
```

代码解释：
*   `model.eval()`: 切换到“评估模式”。这会禁用Dropout等层。
*   `with torch.no_grad()`: 这是一个上下文管理器，它会临时禁用所有梯度计算。这非常重要，因为在评估时我们不需要梯度，关闭它可以显著减少内存消耗并加速计算。
*   循环内部的逻辑与训练循环基本一致，都是进行前向传播和计算损失，但**没有** `optimizer.zero_grad()`、`loss.backward()` 和 `optimizer.step()` 这三个步骤。

#### **4. 整合与执行**

现在，我们把所有部分组合起来，编写主循环来执行整个训练过程。

```python
train_losses = []

for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
```

代码解释：
*   `train_losses = []`: 我们创建一个列表来存储每个轮次的训练损失，以便后续进行可视化。
*   `for epoch in range(num_epochs):`: 主循环，重复我们设定的轮次数。
*   在循环内部，我们依次调用 `train` 函数，并将返回的平均损失存入列表中。
*   我们设置了一个简单的打印逻辑，每10个轮次打印一次当前的训练损失，让我们能够实时监控训练的进展。

---

### **第五部分：实时训练过程可视化**

在这一部分，我们将使用`matplotlib`库来绘制模型的损失曲线。一个理想的损失曲线应该随着训练轮次（Epochs）的增加而平稳下降。我们将在训练循环的末尾添加绘图功能，以实现“实时”或至少是逐轮更新的训练监控。

#### **1. 实时绘图的挑战与策略**

在像Jupyter Notebook这样的交互式环境中，实时更新图表非常方便。但在一个普通的Python脚本中，每次都弹出一个新的绘图窗口会很麻烦。因此，我们的策略是：
1.  在训练开始前，创建一个图表对象。
2.  在每个epoch结束后，更新图表的数据并重新绘制。
3.  在整个训练过程结束后，将最终的图表保存为一张图片。

为了让代码更具通用性，我们将采用一种在脚本环境中也能良好运行的方式，即在训练循环结束后一次性绘制并展示/保存最终的损失曲线。

#### **2. 实现绘图函数**

首先，我们定义一个简单的函数，它接收损失历史记录并使用`matplotlib`进行绘图。

```python
def plot_losses(losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label='Training Loss')
    plt.title('Training Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
```

代码解释：
*   `plt.figure(figsize=(10, 5))`: 创建一个新的图窗，并设置其尺寸为10x5英寸，使其看起来更清晰。
*   `plt.plot(losses, label='Training Loss')`: 这是核心的绘图命令。它以epoch的索引（`matplotlib`会自动处理）为x轴，以我们记录的损失值为y轴，绘制出一条线图。`label`参数用于图例显示。
*   `plt.title(...)`, `plt.xlabel(...)`, `plt.ylabel(...)`: 设置图表的标题、x轴标签和y轴标签，让图表更具可读性。
*   `plt.legend()`: 显示图例，即我们在`plot`函数中定义的`label`。
*   `plt.grid(True)`: 在图表中添加网格线，方便观察数值。
*   `plt.show()`: 显示绘制好的图表。

#### **3. 整合到主训练流程中**

现在，我们将这个绘图功能整合到我们上一部分编写的主训练流程的末尾。我们会先完成所有的训练，然后调用这个函数来展示最终的结果。

这是完整的、带有可视化功能的训练执行代码块：

```python
# --- (This part is the same as in Part 4) ---
# Hyperparameters
learning_rate = 3e-4
num_epochs = 100 # Let's increase epochs for a better curve

# Instantiate model, criterion, optimizer
model = LanguageModel(vocab_size, d_model, n_layers, n_heads).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
train_losses = []
print("Starting training...")
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    train_losses.append(train_loss)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")

print("Training finished.")

# --- (This is the new part for visualization) ---
# Now, plot the results
plot_losses(train_losses)
```

代码解释：
*   我们沿用了第四部分的所有训练设置和训练循环。
*   在整个`for`循环结束后，`train_losses`列表中已经包含了每一轮训练的平均损失。
*   此时，我们调用 `plot_losses(train_losses)`，将这个列表传递给我们的绘图函数。
*   执行这段代码后，在训练过程的日志输出完毕后，会弹出一个窗口，清晰地展示损失值随训练轮次下降的曲线。

**预期结果：**
你将会看到一张图表，其Y轴代表交叉熵损失值，X轴代表训练的轮次（Epoch）。曲线的趋势应该是从左上角向右下角延伸，表示随着训练的进行，模型的预测越来越准，损失值越来越小。曲线的下降速度在初期会很快，后期会逐渐减慢，最终趋于平坦，这表明模型可能已经达到了收敛状态。

---

通过可视化，我们不再仅仅是看到一堆变化的数字，而是能够直观地感受到模型的“成长”过程。这是调试模型、判断训练状态不可或缺的工具。

---

### **第六部分：文本生成（推理）与束搜索**

训练完成后，模型内部的权重已经学会了文本数据的语言模式。现在，我们的任务是利用这些学到的知识来生成新的文本序列。这个过程通常被称为“推理”（Inference）或“解码”（Decoding）。

#### **1. 生成逻辑概述**

无论是贪心搜索还是束搜索，其基本流程都是相似的，可以概括为自回归（auto-regressive）过程：
1.  提供一个起始文本（prompt），如 "the man who"。
2.  将prompt编码为数字索引，并送入模型。
3.  模型会预测出prompt后面每一个位置的下一个字符的概率分布。我们只关心最后一个位置的预测，因为它是在给定整个prompt后对下一个字符的预测。
4.  根据某种策略（贪心或束搜索）从这个概率分布中选择一个字符。
5.  将选中的字符追加到prompt的末尾，形成新的、更长的prompt。
6.  重复步骤2-5，直到生成了足够长的文本或遇到了特殊的结束标记。

#### **2. 实现束搜索 (Beam Search)**

贪心搜索在每一步都选择概率最高的那个词，虽然简单快速，但容易陷入局部最优，导致生成的文本可能比较平庸或出现重复。

束搜索则是一种改进。它在每一步都会保留 `k` 个最可能的候选序列，这个 `k` 就是 **束宽（Beam Width）**。在下一步，它会基于这 `k` 个序列，探索所有可能的下一个字符，并从所有新生成的序列中，再次选出总概率最高的 `k` 个。这个过程不断迭代，直到所有 `k` 个序列都生成了结束符或达到了最大长度。

下面是束搜索的实现代码。它比贪心搜索复杂，但我们会详细解释每一步。

```python
def beam_search_generate(model, start_string, max_len, beam_width=3):
    model.eval()
    
    # --- Initialization ---
    encoded_start = torch.tensor(encode(start_string), dtype=torch.long, device=device).unsqueeze(0)
    
    # A list of tuples: (sequence, probability_score)
    # The score is initially log probability, so we start with 0.
    beams = [(encoded_start, 0.0)]
    
    with torch.no_grad():
        for _ in range(max_len):
            all_candidates = []
            
            # --- Expansion Step ---
            # For each beam (existing candidate sequence), predict the next token
            for seq, score in beams:
                # The model expects a batch, so the shape should be [1, sequence_length]
                input_tensor = seq
                
                logits = model(input_tensor)
                # We only care about the prediction for the very last token
                last_logits = logits[:, -1, :]
                
                # Apply softmax to get probabilities, and then log for scoring
                log_probs = torch.log_softmax(last_logits, dim=-1)
                
                # Get the top `beam_width` most likely next tokens and their log probabilities
                top_log_probs, top_indices = torch.topk(log_probs, beam_width, dim=-1)
                
                # Add the new candidates to our list
                for i in range(beam_width):
                    new_token = top_indices[0, i].unsqueeze(0).unsqueeze(0) # Shape [1, 1]
                    new_log_prob = top_log_probs[0, i].item()
                    
                    # Create the new sequence and calculate its total score
                    new_seq = torch.cat([seq, new_token], dim=1)
                    new_score = score + new_log_prob
                    all_candidates.append((new_seq, new_score))
            
            # --- Pruning Step ---
            # Sort all candidates by their score (higher is better)
            ordered_candidates = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            
            # Keep only the top `beam_width` candidates for the next iteration
            beams = ordered_candidates[:beam_width]
            
            # --- Check for Completion (Optional but good practice) ---
            # If the top beam ends with an end-of-sentence token, you could stop.
            # For this simple char-level model, we'll just run to max_len.

    # --- Final Selection ---
    # The best sequence is the one with the highest score at the end
    best_seq, _ = beams[0]
    generated_text = decode(best_seq.squeeze(0).tolist())
    
    return generated_text
```

代码解释：
*   `def beam_search_generate(...)`: 定义生成函数，接收模型、起始文本、最大生成长度和束宽作为参数。
*   `model.eval()`: 切换到评估模式，这非常重要。
*   **初始化**:
    *   `encoded_start = ...`: 将起始文本编码为张量。
    *   `beams = [(encoded_start, 0.0)]`: 初始化我们的束列表。它包含一个元组，元组的第一个元素是序列本身，第二个元素是该序列的累积对数概率得分。初始得分为0。
*   **主循环**: 循环 `max_len` 次来生成新字符。
    *   `all_candidates = []`: 在每一轮生成开始前，创建一个空列表来存放所有可能的新序列。
    *   **扩展步骤 (Expansion)**:
        *   `for seq, score in beams:`: 遍历当前束中的每一个候选序列。
        *   `logits = model(input_tensor)`: 将序列输入模型，得到预测结果。
        *   `last_logits = logits[:, -1, :]`: 我们只关心对最后一个时间步的预测。
        *   `log_probs = torch.log_softmax(...)`: 将`logits`转换为对数概率。我们使用对数概率而不是原始概率，因为多个小概率相乘容易导致数值下溢，而对数概率相加则更稳定。
        *   `top_log_probs, top_indices = torch.topk(...)`: 找出概率最高的 `beam_width` 个下一个字符的索引和它们的对数概率。
        *   `for i in range(beam_width):`: 对于每一个可能的下一个字符，我们都创建一个新的候选序列。
        *   `new_seq = torch.cat(...)`: 将新预测的字符拼接到旧序列的末尾。
        *   `new_score = score + new_log_prob`: 新序列的总分等于旧序列的分数加上新字符的对数概率。
        *   `all_candidates.append(...)`: 将这个新的（序列，分数）对添加到候选列表中。
    *   **剪枝步骤 (Pruning)**:
        *   `ordered_candidates = sorted(...)`: 在扩展完所有束之后，`all_candidates` 列表中现在会有 `beam_width * beam_width` 个候选序列。我们根据它们的分数（对数概率，值越大越好）进行降序排序。
        *   `beams = ordered_candidates[:beam_width]`: 我们只保留排序后的前 `beam_width` 个序列作为下一轮迭代的输入。所有其他的可能性都被“剪掉”了。
*   **最终选择**:
    *   `best_seq, _ = beams[0]`: 当循环结束后，`beams` 列表中的第一个序列就是总分数最高的那个。
    *   `generated_text = decode(...)`: 将其解码回人类可读的文本并返回。

#### **3. 执行生成**

现在，让我们使用这个函数，看看我们训练好的模型能生成什么样的文本。

```python
# Set a starting prompt
prompt = "the man who"

# Generate text using beam search
generated_output = beam_search_generate(
    model=model,
    start_string=prompt,
    max_len=200,
    beam_width=5 # A beam width of 5 is a good starting point
)

print("--- Generated Text ---")
print(generated_output)
```

代码解释：
*   我们定义一个 `prompt`。
*   调用我们刚刚编写的 `beam_search_generate` 函数，传入模型、prompt、想要生成的长度以及束宽。
*   打印函数返回的最终生成的文本。

**预期结果：**
输出的文本将以 "the man who" 开头，后面跟着模型生成的200个字符。由于我们使用的是字符级模型和相对较小的数据集，生成的文本可能不会完全符合语法规则，或者会出现一些拼写错误和重复。但是，它应该能够捕捉到原始文本的一些风格，比如单词的组合方式和空格的使用。束搜索的引入，会使其比简单的贪心搜索生成的文本更加连贯和多样化。

---

我们已经成功地让模型生成了文本，并且使用了强大的束搜索算法！这是对我们之前所有工作的最终检验。

### **第七部分：模型保存、加载与整合**

在这一部分，我们将学习如何将训练好的模型权重保存到文件中，以及如何在需要时将它们加载回来，从而避免重复训练。最后，我们会把整个项目的代码逻辑整合到一个清晰的脚本框架中。

#### **1. 保存模型 (Saving the Model)**

在PyTorch中，保存模型的推荐方式是只保存模型的可学习参数（即权重和偏置），而不是整个模型对象。这被称为保存模型的`state_dict`（状态字典）。这样做更轻量、更灵活，不容易在代码重构时出错。

我们通常在训练循环结束后，或者在验证集上达到最佳性能时保存模型。

```python
# Define a path to save the model
MODEL_SAVE_PATH = "transformer_text_generator.pth"

# Save the model's state dictionary
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print(f"Model saved to {MODEL_SAVE_PATH}")
```

代码解释：
*   `MODEL_SAVE_PATH`: 我们定义一个字符串变量来存储模型文件的路径和名称。`.pth` 或 `.pt` 是PyTorch模型文件常用的扩展名。
*   `model.state_dict()`: 这是一个PyTorch `nn.Module` 的内置方法，它返回一个包含了模型所有参数（权重`weight`和偏置`bias`）的Python字典。字典的键是层的名称，值是对应的参数张量。
*   `torch.save(object, path)`: 这是PyTorch通用的保存函数。我们将模型的状态字典作为要保存的对象，传递给它。

#### **2. 加载模型 (Loading the Model)**

加载模型的过程与保存相对应，分为两步：
1.  首先，我们需要创建一个与保存时结构完全相同的模型实例。
2.  然后，使用`load_state_dict()`方法将文件中保存的参数加载到这个新的模型实例中。

```python
# --- Imagine this is a new script, or you have restarted your session ---

# 1. First, instantiate the model with the same architecture
loaded_model = LanguageModel(
    vocab_size=vocab_size,
    d_model=d_model,
    n_layers=n_layers,
    n_heads=n_heads
)

# 2. Load the saved state dictionary
# We use map_location to ensure the model loads correctly on CPU if it was saved from a GPU
loaded_model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))

# 3. Move the model to the desired device
loaded_model.to(device)

# 4. Remember to set the model to evaluation mode for inference
loaded_model.eval()

print(f"Model loaded from {MODEL_SAVE_PATH}")

# You can now use the loaded_model for generation, just like the original one
prompt = "who is in the"
generated_text = beam_search_generate(loaded_model, prompt, max_len=100, beam_width=5)

print("\n--- Generation from Loaded Model ---")
print(generated_text)
```

代码解释：
*   `loaded_model = LanguageModel(...)`: 我们必须先创建一个模型的“骨架”。PyTorch需要知道这些加载进来的权重应该放在模型的哪个位置。因此，这个实例的结构必须和我们保存时使用的模型结构一模一样。
*   `torch.load(MODEL_SAVE_PATH, map_location=device)`: `torch.load` 函数从文件中读取对象。`map_location=device` 是一个非常有用的参数，它能确保即使你当前的环境没有GPU，也能成功加载一个在GPU上训练和保存的模型（它会智能地将张量映射到CPU上）。
*   `loaded_model.load_state_dict(...)`: 这个方法会将加载的字典中的参数，按照键名匹配的方式，复制到`loaded_model`的对应参数上。
*   `loaded_model.to(device)` 和 `loaded_model.eval()`: 加载完成后，不要忘记将模型移动到正确的设备，并切换到评估模式，这和我们直接使用训练好的模型进行推理时是完全一样的步骤。

#### **3. 项目整合 (Putting It All Together)**

现在，我们将之前所有部分的代码，按照逻辑顺序和良好的编程实践，整合到一个完整的脚本框架中。这展示了一个真实项目的文件结构会是什么样子。

```python
# ==============================================================================
# main_script.py
# ==============================================================================

# --- (Part 1: Imports and Global Settings) ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from tqdm import tqdm

# Settings
def set_seed(seed):
    # ... (function definition from Part 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_SAVE_PATH = "transformer_text_generator.pth"
DO_TRAINING = True # A flag to control whether to train or just load

# --- (Part 2: Data Processing) ---
# Hyperparameters
block_size = 8
batch_size = 4
# Raw Text Data
raw_text = """...""" # (Your text data from Part 2)
raw_text = raw_text.lower()
# Vocabulary
chars = sorted(list(set(raw_text)))
vocab_size = len(chars)
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])
# Dataset and DataLoader
class TextDataset(Dataset):
    # ... (class definition from Part 2)
    def __init__(self, text, block_size):
        self.block_size = block_size
        self.data = torch.tensor(encode(text), dtype=torch.long)
    def __len__(self):
        return len(self.data) - self.block_size
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.block_size]
        y = self.data[idx+1:idx+self.block_size+1]
        return x, y

dataset = TextDataset(raw_text, block_size)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# --- (Part 3: Model Architecture) ---
# Hyperparameters
d_model = 64
n_heads = 4
n_layers = 3
dropout_rate = 0.1
# Class definitions (PositionalEncoding, MultiHeadAttention, FeedForward, DecoderBlock, LanguageModel)
class PositionalEncoding(nn.Module):
    # ...
class MultiHeadAttention(nn.Module):
    # ...
class FeedForward(nn.Module):
    # ...
class DecoderBlock(nn.Module):
    # ...
class LanguageModel(nn.Module):
    # ...
    
# --- (Part 4 & 5: Training and Visualization) ---
# Hyperparameters
learning_rate = 3e-4
num_epochs = 100
# Function definitions (train, evaluate, plot_losses)
def train(model, data_loader, criterion, optimizer, device):
    # ...
def plot_losses(losses):
    # ...

# --- (Part 6: Generation) ---
# Function definition (beam_search_generate)
def beam_search_generate(model, start_string, max_len, beam_width=3):
    # ...

# --- Main Execution Block ---
if __name__ == "__main__":
    
    # Instantiate the model
    model = LanguageModel(vocab_size, d_model, n_layers, n_heads).to(device)
    
    if DO_TRAINING:
        print("--- Starting Model Training ---")
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        train_losses = []
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, criterion, optimizer, device)
            train_losses.append(train_loss)
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}")
        
        print("--- Training Finished ---")
        plot_losses(train_losses)
        
        # Save the trained model
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"Model saved to {MODEL_SAVE_PATH}")

    else:
        print("--- Loading Pre-trained Model ---")
        try:
            model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
            print("Model loaded successfully.")
        except FileNotFoundError:
            print(f"Error: Model file not found at {MODEL_SAVE_PATH}. Please train first by setting DO_TRAINING = True.")
            exit()
            
    # --- Perform Generation ---
    print("\n--- Generating Text ---")
    model.eval()
    prompt = "the credit belongs to"
    generated_text = beam_search_generate(model, prompt, max_len=300, beam_width=5)
    print(generated_text)

```

代码解释：
*   我们使用了一个 `DO_TRAINING` 布尔标志。当它为`True`时，脚本会执行完整的训练、绘图和保存流程。当它为`False`时，脚本会跳过训练，直接尝试加载已保存的模型文件。这使得我们可以轻松地在“训练模式”和“推理模式”之间切换。
*   `if __name__ == "__main__":`: 这是Python脚本的标准写法，确保只有当这个文件作为主程序运行时，才会执行内部的代码。
*   我们将所有的类和函数定义放在了主执行块之前，使得代码结构更加清晰。
*   在加载部分，我们加入了一个`try-except`块来处理模型文件不存在的异常情况，使程序更加健壮。

---

### **总结**

恭喜您！我们已经一起从零开始，走完了使用PyTorch构建、训练和使用一个文本生成Transformer模型的全部流程。

您现在拥有了一份详尽的、覆盖了从数据处理到模型部署的完整笔记。回顾一下，我们完成了：
1.  **项目准备**：搭建了环境并明确了目标。
2.  **数据管道**：将原始文本转换为了模型可用的批量化数据。
3.  **模型构建**：深入理解并亲手实现了Transformer的每一个核心组件。
4.  **模型训练**：编写了标准的训练和评估循环。
5.  **过程可视化**：学会了如何监控模型的学习进度。
6.  **文本生成**：实现了先进的束搜索算法来产出高质量文本。
7.  **模型持久化**：掌握了保存和加载模型，使工作成果可复用的关键技能。

这份笔记为您打下了一个坚实的基础。您可以基于此框架，去探索更庞大的数据集、更深层的模型架构、更复杂的解码策略，以及更广阔的自然语言生成世界。
