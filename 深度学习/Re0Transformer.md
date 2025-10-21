这份指南将具备以下特点：
*   **纯粹的PyTorch实现**：不使用任何高级封装（如`nn.Transformer`），我们将从`nn.Module`和`nn.Linear`等基础组件构建一切。
*   **从零处理数据**：不使用预训练词向量和外部NLP库（如Spacy）进行分词。我们将实现自己的字符级词汇表，并处理原始文本。
*   **任务导向**：目标是训练一个能够生成文本的模型。
*   **工程化结构**：代码将被组织在清晰的项目结构中，而不是一个单一的脚本。
*   **分步、详尽的阐述**：我将先提供总纲，然后逐部分输出。每个代码部分都将包含 **(1) 代码流程详述**，**(2) 带注释的代码实现**，以及 **(3) 对代码的逐行/逐块解释**。

---

### **从零开始使用PyTorch实现Transformer：总纲**

#### **第零部分：项目搭建与数据准备**
*   **0.1 项目结构**：定义一个清晰、可扩展的文件和目录结构。
*   **0.2 数据集**：选择并获取一个简单、易于处理的公开文本数据集（我们将使用 "tiny-shakespeare"）。
*   **0.3 字符级词汇表**：构建一个从零开始的词汇表类，负责将字符映射到整数索引，反之亦然。
*   **0.4 数据集与数据加载器**：
    *   创建自定义的PyTorch `Dataset` 类，用于将长文本切分为输入/输出样本对。
    *   配置 `DataLoader`，并实现一个自定义的 `collate_fn` 来处理批处理中的序列填充（Padding）。
*   **0.5 配置文件**：创建一个 `config.py` 文件来统一管理所有超参数。

#### **第一部分：构建Transformer的核心组件**
*   **1.1 输入层：词嵌入与位置编码**
    *   实现 `TokenEmbedding` 模块。
    *   实现 `PositionalEncoding` 模块，使用sin/cos函数生成固定的位置向量。
*   **1.2 多头自注意力机制 (Multi-Head Attention)**
    *   实现一个完整的、通用的多头注意力模块，它将是编码器和解码器的核心。
*   **1.3 位置全连接前馈网络 (Position-wise Feed-Forward Network)**
    *   实现FFN模块。
*   **1.4 编码器层 (Encoder Layer)**
    *   将多头自注意力和FFN与残差连接、层归一化组合成一个完整的编码器层。
*   **1.5 解码器层 (Decoder Layer)**
    *   组合带掩码的多头自注意力、编码器-解码器注意力、FFN以及辅助模块，构成一个完整的解码器层。

#### **第二部分：组装完整的Transformer模型**
*   **2.1 编码器 (Encoder)**：堆叠N个编码器层。
*   **2.2 解码器 (Decoder)**：堆叠N个解码器层。
*   **2.3 完整的Transformer模型**：
    *   将编码器、解码器、输入层和最终的输出线性层整合到一个 `Transformer` 类中。
    *   实现模型的前向传播逻辑和生成后续位置掩码的辅助函数。

#### **第三部分：模型训练**
*   **3.1 训练脚本 (`train.py`) 的结构**：
    *   设置设备（CPU/GPU）、加载数据、初始化模型。
    *   定义优化器和损失函数（带填充忽略的交叉熵）。
*   **3.2 训练循环**：
    *   实现完整的训练循环逻辑，包括数据迭代、模型前向传播、损失计算、反向传播和参数更新。
    *   添加训练过程的可视化（打印损失）。
*   **3.3 模型保存**：在训练结束后保存模型权重。

#### **第四部分：文本生成（推理）**
*   **4.1 生成脚本 (`generate.py`) 的结构**：
    *   加载已保存的模型和词汇表。
*   **4.2 实现生成函数**：
    *   实现自回归（auto-regressive）的解码过程。
    *   使用贪心搜索（Greedy Search）策略从概率分布中选择下一个词元。
    *   展示如何使用训练好的模型生成新的文本。
    好的，我们正式开始这个项目。

---

### **第零部分：项目搭建与数据准备**

在编写任何模型代码之前，我们需要建立一个稳健的项目结构，并准备好我们的数据管道。这是确保项目可维护、可复现和可扩展的关键第一步。我们将从零开始，包括下载数据、构建我们自己的字符级词汇表，以及设置PyTorch的`Dataset`和`DataLoader`。

#### **0.1 项目结构**

##### **代码流程详述**
一个良好的项目结构能够将不同功能的代码分离开来，使得逻辑更加清晰。我们将创建以下目录和文件：
1.  **`transformer-from-scratch/`**: 项目的根目录。
2.  **`config.py`**: 一个中心化的配置文件，用于存放所有的超参数，如批次大小、学习率、模型维度等。这使得调整实验参数变得非常容易。
3.  **`data/`**: 一个用于存放原始数据集文件的目录。
4.  **`utils.py`**: 存放辅助工具，我们自定义的词汇表类将放在这里。
5.  **`dataset.py`**: 存放数据处理相关的代码，包括PyTorch的`Dataset`子类和创建`DataLoader`的函数。
6.  **`model.py`**: 存放所有与Transformer模型结构相关的`nn.Module`子类。
7.  **`train.py`**: 训练模型的主脚本。
8.  **`generate.py`**: 使用训练好的模型进行文本生成（推理）的脚本。

执行以下命令来创建这个结构：
```bash
mkdir transformer-from-scratch
cd transformer-from-scratch
mkdir data
touch config.py utils.py dataset.py model.py train.py generate.py
```

---

#### **0.2 数据集**

##### **代码流程详述**
我们将使用"tiny-shakespeare"数据集，这是一个包含了莎士比亚部分作品的纯文本文件。它足够小，可以在单张消费级GPU上快速训练，同时又足够复杂，能够展示模型的学习能力。
我们将使用`wget`命令从网上下载这个数据集，并将其存放在`data/`目录中。

##### **代码实现**
在您的终端中，确保您位于`transformer-from-scratch`根目录下，然后执行：
```bash
wget -O data/tiny_shakespeare.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
```

##### **代码解释**
*   `wget`: 一个常用的命令行工具，用于从网络上下载文件。
*   `-O data/tiny_shakespeare.txt`: 这个选项指定了输出文件的路径和名称。我们将下载的文件直接保存到`data`目录下，并命名为`tiny_shakespeare.txt`。
*   `https://.../input.txt`: 这是"tiny-shakespeare"数据集的原始URL。

---

#### **0.3 字符级词汇表**

##### **代码流程详述**
由于我们不使用外部库进行分词，最直接的方法是创建一个字符级别的词汇表。这意味着模型的基本单位是单个字符（字母、标点、空格等），而不是单词。
我们的词汇表类 `CharacterVocabulary` 需要实现以下功能：
1.  在初始化时，读取整个文本文件，找出所有不重复的字符。
2.  创建两个核心的映射字典：
    *   `stoi` (string-to-index): 将每个字符映射到一个唯一的整数索引。
    *   `itos` (index-to-string): 将整数索引映射回对应的字符。
3.  包含一些特殊的词元（token）：
    *   `<PAD>`: 填充符，用于将批次中不同长度的序列填充到相同长度。其索引通常为0。
    *   `<SOS>`: 序列起始符 (Start of Sequence)，在生成任务中标志着解码的开始。
    *   `<EOS>`: 序列结束符 (End of Sequence)，标志着一个序列的结束。
    *   `<UNK>`: 未知字符符。虽然在字符级模型中，测试集不太可能出现训练集没有的字符，但包含它是一个好习惯。
4.  提供`encode`和`decode`方法，用于在字符序列和整数索引序列之间进行转换。

##### **代码实现 (`utils.py`)**
```python
# utils.py

class CharacterVocabulary:
    """
    Manages the mapping between characters and integer indices.
    """
    def __init__(self, text):
        # Special tokens
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>"
        self.eos_token = "<EOS>"
        self.unk_token = "<UNK>"
        
        # Find all unique characters in the text
        self.chars = sorted(list(set(text)))
        
        # Create the vocabulary and mappings
        self.vocab = [self.pad_token, self.sos_token, self.eos_token, self.unk_token] + self.chars
        
        # String-to-index and index-to-string mappings
        self.stoi = {char: i for i, char in enumerate(self.vocab)}
        self.itos = {i: char for i, char in enumerate(self.vocab)}

        # Get indices for special tokens
        self.pad_idx = self.stoi[self.pad_token]
        self.sos_idx = self.stoi[self.sos_token]
        self.eos_idx = self.stoi[self.eos_token]
        self.unk_idx = self.stoi[self.unk_token]

    def __len__(self):
        # Returns the total size of the vocabulary
        return len(self.vocab)

    def encode(self, text):
        # Converts a string of text into a list of integer indices
        return [self.stoi.get(char, self.unk_idx) for char in text]

    def decode(self, indices):
        # Converts a list of integer indices back into a string
        return "".join([self.itos.get(i, self.unk_token) for i in indices])

```

##### **代码解释**
*   **`__init__(self, text)`**:
    *   `self.pad_token`, `...`: 定义了我们将要使用的四个特殊字符的字符串表示。
    *   `self.chars = sorted(list(set(text)))`: 这一行是核心。`set(text)`找出所有唯一的字符，`list()`将其转换为列表，`sorted()`确保每次运行时字符的顺序都是一致的，这对于模型的可复现性至关重要。
    *   `self.vocab = [...] + self.chars`: 构建完整的词汇表列表。我们将特殊字符放在最前面，这样它们的索引（0, 1, 2, 3）是固定的。
    *   `self.stoi = ...`, `self.itos = ...`: 使用字典推导式，高效地创建字符到索引和索引到字符的映射。
    *   `self.pad_idx = ...`: 获取并存储特殊字符的索引，方便后续在代码中直接使用，而不用每次都去查询字典。
*   **`__len__(self)`**:
    *   返回词汇表的总大小。这在定义模型的嵌入层和输出层时非常重要。
*   **`encode(self, text)`**:
    *   这是一个列表推导式，遍历输入`text`中的每个字符。
    *   `self.stoi.get(char, self.unk_idx)`: 尝试在`stoi`字典中查找字符`char`。如果找到了，返回其索引；如果没找到（例如，在推理时遇到一个训练时从未见过的字符），则返回`<UNK>`的索引。
*   **`decode(self, indices)`**:
    *   同样使用列表推导式，遍历输入的整数索引列表`indices`。
    *   `self.itos.get(i, self.unk_token)`: 查找索引`i`对应的字符，如果找不到则返回`<UNK>`字符。
    *   `"".join(...)`: 将解码出的字符列表重新组合成一个字符串。

---

#### **0.4 数据集与数据加载器**

##### **代码流程详述**
现在我们需要一个方法来将原始的长文本字符串转换成模型可以消费的、一批批的张量（tensors）。这需要两样东西：
1.  **`torch.utils.data.Dataset`**: 一个自定义的PyTorch `Dataset`类。它的工作是告诉PyTorch如何从数据源中获取**单个**样本。对于文本生成任务，一个样本通常是一对`(输入序列, 目标序列)`。我们将通过在长文本上滑动一个固定大小的窗口来创建这些样本。例如，如果窗口大小（`block_size`）是8，文本是"hello world"，第一个样本的输入就是"hello wo"，目标就是"ello wor"。
2.  **`torch.utils.data.DataLoader`**: `DataLoader`会包装我们的`Dataset`，并自动地从中抽取小批量（mini-batches）数据，进行打乱（shuffle）等操作。

##### **代码实现 (`dataset.py`)**
```python
# dataset.py

import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    """
    Custom PyTorch Dataset for creating input/target sequences from a long text.
    """
    def __init__(self, text, vocab, block_size):
        self.vocab = vocab
        self.block_size = block_size
        
        # Encode the entire text into integer indices
        self.data = self.vocab.encode(text)

    def __len__(self):
        # The number of possible sequences we can create
        # We subtract block_size because the last possible sequence starts at len(data) - block_size
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Grab a chunk of (block_size + 1) characters
        chunk = self.data[idx : idx + self.block_size + 1]
        
        # The first block_size characters are the input
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        
        # The last block_size characters (shifted by one) are the target
        y = torch.tensor(chunk[1:], dtype=torch.long)
        
        return x, y

def create_dataloader(text, vocab, block_size, batch_size, shuffle=True):
    """
    Creates a DataLoader for the text generation task.
    """
    dataset = TextDataset(text, vocab, block_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0, # Can be increased if data loading is a bottleneck
    )
    return dataloader
```

##### **代码解释**
*   **`TextDataset` 类**:
    *   **`__init__(self, text, vocab, block_size)`**:
        *   存储传入的`vocab`对象和`block_size`（序列长度）。
        *   `self.data = self.vocab.encode(text)`: 立即将全部文本编码为一长串整数列表，后续将从这个列表中切片。
    *   **`__len__(self)`**:
        *   返回数据集中样本的总数。如果总数据长度为N，序列长度为B，那么最后一个有效的起始位置是 `N - 1 - B`。为了简单，我们用 `N - B` 作为长度，这确保了即使从最后一个可能的起始点 `N - B - 1` 取 `B+1` 个字符，也不会越界。
    *   **`__getitem__(self, idx)`**:
        *   这是`Dataset`的核心。当`DataLoader`请求第`idx`个样本时，此方法被调用。
        *   `chunk = self.data[idx : idx + self.block_size + 1]`: 从编码后的数据中切出一个长度为`block_size + 1`的片段。
        *   `x = torch.tensor(chunk[:-1], ...)`: 将片段的前`block_size`个元素作为输入`x`。
        *   `y = torch.tensor(chunk[1:], ...)`: 将片段的后`block_size`个元素（即`x`向右移动一位）作为目标`y`。这正是语言模型的训练方式：给定前面的字符，预测下一个字符。
        *   `torch.tensor(..., dtype=torch.long)`: 将Python列表转换为PyTorch张量，数据类型为`long`（64位整数），这是嵌入层所要求的索引类型。
*   **`create_dataloader` 函数**:
    *   这是一个工厂函数，封装了创建`DataLoader`的逻辑。
    *   `dataset = TextDataset(...)`: 实例化我们刚刚定义的`Dataset`。
    *   `dataloader = DataLoader(...)`: 实例化PyTorch的`DataLoader`。
        *   `dataset`: 要从中加载数据的`Dataset`对象。
        *   `batch_size`: 每个批次包含多少个样本。
        *   `shuffle=True`: 在每个epoch开始时，打乱数据的顺序。这对于训练的稳定性和泛化性非常重要。

---

#### **0.5 配置文件**

##### **代码流程详述**
我们将所有重要的超参数集中在一个文件中，方便管理和修改。

##### **代码实现 (`config.py`)**
```python
# config.py

# --- Data Parameters ---
DATA_PATH = "data/tiny_shakespeare.txt"
# For simplicity, we'll use a small portion of the data for quick training.
# Set to 1.0 to use the full dataset.
TRAIN_DATA_RATIO = 0.9 

# --- Model Hyperparameters ---
BATCH_SIZE = 64         # How many independent sequences will we process in parallel?
BLOCK_SIZE = 256        # What is the maximum context length for predictions?
D_MODEL = 512           # The dimension of the embedding and the model's hidden states.
N_HEAD = 8              # The number of attention heads.
N_LAYER = 6             # The number of Transformer blocks (Encoder/Decoder layers).
DROPOUT = 0.1           # The dropout rate.

# --- Training Hyperparameters ---
LEARNING_RATE = 3e-4
NUM_EPOCHS = 5
DEVICE = "cuda" # "cuda" if torch.cuda.is_available() else "cpu"

```

##### **代码解释**
*   每个参数都有清晰的命名和注释，解释了它的作用。
*   `TRAIN_DATA_RATIO`: 我们添加了一个参数来划分训练集和验证集。
*   `DEVICE`: 自动检测是否有可用的GPU，这使得代码更具可移植性。
*   将这些参数放在一个单独的文件中，意味着当你想尝试不同的模型大小（如改变`D_MODEL`或`N_LAYER`）或训练设置时，你只需要修改这一个文件，而无需深入到模型或训练脚本的内部逻辑中。

---
好的，我们继续构建Transformer模型的核心组件。

---

### **第一部分：构建Transformer的核心组件**

在这一部分，我们将逐一实现构成Transformer架构的各个基本模块。我们将严格遵循模块化的思想，确保每个类都只负责一项明确的功能。所有的模型代码都将写入`model.py`文件中。

#### **1.1 输入层：词嵌入与位置编码**

##### **代码流程详述**
输入层负责将输入的整数索引序列转换为包含语义和位置信息的密集向量序列。这由两个子模块完成：
1.  **`TokenEmbedding`**: 这是一个标准的PyTorch `nn.Embedding`层。它将每个词元索引映射到一个可学习的`d_model`维向量。
2.  **`PositionalEncoding`**: 这个模块负责创建固定的、基于sin/cos函数的位置编码矩阵。它不是一个可学习的层，但在模型中扮演着至关重要的角色。
    *   在`__init__`中，它会预先计算一个足够大的位置编码矩阵（例如，支持的最大序列长度为5000）。这个矩阵会被注册为模型的缓冲区（`register_buffer`），这意味着它会随着模型一起被移动到GPU，但不会被视为模型参数进行梯度更新。
    *   在`forward`方法中，它接收词嵌入的输出，并将对应长度的位置编码切片加到词嵌入上。我们还会加入一个`nn.Dropout`层来增加正则化。

##### **代码实现 (`model.py`)**
```python
# model.py

import torch
import torch.nn as nn
import math

class TokenEmbedding(nn.Module):
    """
    Converts token indices to dense embedding vectors.
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # The original paper scales the embeddings by sqrt(d_model).
        return self.embedding(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Injects positional information into the input embeddings.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a positional encoding matrix of shape (max_len, d_model)
        pe = torch.zeros(max_len, d_model)
        
        # Create a tensor for the positions (0, 1, 2, ..., max_len-1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        # Calculate the division term for the sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Add a batch dimension and register as a buffer
        pe = pe.unsqueeze(0) # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        # Add positional encoding to the input tensor
        # self.pe is sliced to match the input sequence length
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

```

##### **代码解释**
*   **`TokenEmbedding` 类**:
    *   `__init__`: 初始化一个`nn.Embedding`层。`vocab_size`是词汇表大小，`d_model`是嵌入向量的维度。
    *   `forward`:
        *   `self.embedding(x)`: PyTorch的嵌入层会自动将输入的索引张量`x`（形状 `[batch_size, seq_len]`）转换为嵌入向量张量（形状 `[batch_size, seq_len, d_model]`）。
        *   `* math.sqrt(self.d_model)`: 遵循原版Transformer论文的做法，将嵌入向量乘以模型维度的平方根。这有助于在残差连接中平衡嵌入向量和位置编码的量级。
*   **`PositionalEncoding` 类**:
    *   `__init__`:
        *   `self.dropout`: 初始化一个Dropout层，用于正则化。
        *   `pe = torch.zeros(...)`: 创建一个空的矩阵，用于存放位置编码。
        *   `position = torch.arange(...).unsqueeze(1)`: 创建一个列向量 `[[0], [1], ..., [max_len-1]]`，代表序列中的位置。
        *   `div_term = torch.exp(...)`: 这是对公式 $10000^{2i/d_{model}}$ 的高效计算。通过在对数空间中计算 `(2i/d_model) * log(10000)` 然后取指数，可以避免直接计算大数值的幂，从而保持数值稳定性。
        *   `pe[:, 0::2] = ...`, `pe[:, 1::2] = ...`: 利用PyTorch的切片功能，一次性计算所有偶数维度和奇数维度的sin/cos值。`position * div_term` 利用了广播机制，将 `[max_len, 1]` 的位置向量和 `[d_model/2]` 的`div_term`向量相乘，得到 `[max_len, d_model/2]` 的结果。
        *   `self.register_buffer('pe', pe)`: 将`pe`张量注册为模型的缓冲区。这意味着`pe`是模型状态的一部分（会随`model.to(device)`移动），但它不是一个需要计算梯度的参数。
    *   `forward`:
        *   `x.size(1)`: 获取输入序列的实际长度`seq_len`。
        *   `self.pe[:, :x.size(1), :]`: 从预先计算好的`pe`矩阵中，切出与当前输入序列长度相匹配的部分。
        *   `x = x + ...`: 将位置编码加到词嵌入上。
        *   `self.dropout(x)`: 应用dropout后返回结果。

---

#### **1.2 多头自注意力机制 (Multi-Head Attention)**

##### **代码流程详述**
这是Transformer最核心的组件。我们将实现一个通用的多头注意力模块，它可以同时用于编码器的自注意力、解码器的带掩码自注意力以及编码器-解码器注意力。
1.  **初始化 (`__init__`)**:
    *   接收`d_model`和`n_head`作为参数。
    *   断言`d_model`必须能被`n_head`整除。
    *   创建四个关键的线性层：`q_proj`, `k_proj`, `v_proj`用于将输入投影到Q, K, V空间，以及`out_proj`用于将多头拼接后的结果进行最终投影。
    *   保存`n_head`和`d_k`（每个头的维度）。
2.  **前向传播 (`forward`)**:
    *   接收`q`, `k`, `v`以及一个可选的`mask`作为输入。
    *   获取批次大小`batch_size`。
    *   将输入的`q`, `k`, `v`分别通过对应的线性投影层。
    *   **拆分多头**: 将投影后的Q, K, V张量从形状 `[batch_size, seq_len, d_model]` 变换为 `[batch_size, n_head, seq_len, d_k]`。这通过`.view()`和`.transpose()`操作完成，以便每个头可以独立进行计算。
    *   **计算注意力**:
        *   计算分数：$Q \cdot K^T$。
        *   应用缩放因子。
        *   如果提供了`mask`，则将mask中为`False`（或0）的位置的分数设置为一个极大的负数。
        *   应用Softmax得到注意力权重。
        *   应用Dropout。
        *   将权重与V相乘得到每个头的输出。
    *   **合并多头**: 将多头输出从 `[batch_size, n_head, seq_len, d_k]` 变换回 `[batch_size, seq_len, d_model]`。这需要`.transpose()`和`.contiguous().view()`操作。
    *   将合并后的结果通过最终的输出投影层`out_proj`。

##### **代码实现 (`model.py`)**
```python
# model.py (continued)

class MultiHeadAttention(nn.Module):
    """
    Implements the Multi-Head Attention mechanism.
    """
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"

        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head # Dimension of each head's key/query/value

        # Linear projections for Q, K, V and the final output
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # q, k, v shape: (batch_size, seq_len, d_model)
        # mask shape: (batch_size, 1, seq_len, seq_len) or (batch_size, 1, 1, seq_len)
        
        batch_size = q.size(0)

        # 1. Perform linear projections and split into heads
        # .view() reshapes the tensor.
        # .transpose() swaps dimensions.
        # Resulting shape for Q, K, V: (batch_size, n_head, seq_len, d_k)
        Q = self.q_proj(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        K = self.k_proj(k).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)
        V = self.v_proj(v).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)

        # 2. Calculate attention scores
        # K.transpose(-2, -1) results in shape (batch_size, n_head, d_k, seq_len)
        # scores shape: (batch_size, n_head, seq_len, seq_len)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 3. Apply mask (if provided)
        if mask is not None:
            # The mask is broadcasted to match the scores tensor's shape
            scores = scores.masked_fill(mask == 0, -1e9)

        # 4. Apply softmax and dropout
        # attn_weights shape: (batch_size, n_head, seq_len, seq_len)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. Multiply by V to get the output
        # output shape: (batch_size, n_head, seq_len, d_k)
        output = torch.matmul(attn_weights, V)

        # 6. Concatenate heads and apply final linear projection
        # .transpose() followed by .contiguous() and .view() to merge heads
        # output shape: (batch_size, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.out_proj(output)

        return output
```

##### **代码解释**
*   **`__init__`**:
    *   `assert d_model % n_head == 0`: 确保可以平均分配维度给每个头。
    *   `self.d_k = d_model // n_head`: 计算每个头的维度。
    *   `nn.Linear(d_model, d_model)`: 我们为Q, K, V各创建一个线性层。注意，这里输出维度仍然是`d_model`，拆分到各个头是在`forward`中通过`view`操作完成的，这是一种常见的实现方式。
*   **`forward`**:
    *   `Q = self.q_proj(q).view(batch_size, -1, self.n_head, self.d_k).transpose(1, 2)`: 这是一个关键的变形操作链。
        1.  `self.q_proj(q)`: 形状 `[B, S, D]` (B=batch, S=seq_len, D=d_model)。
        2.  `.view(B, -1, H, d_k)`: 将`D`维度拆分为`H`（头数）和`d_k`（头维度）。`-1`让PyTorch自动推断序列长度。形状变为 `[B, S, H, d_k]`。
        3.  `.transpose(1, 2)`: 交换序列长度和头数维度，得到 `[B, H, S, d_k]`。这个形状对于并行的头计算非常方便。
    *   `scores = torch.matmul(Q, K.transpose(-2, -1))`: 计算Q和K的点积。`K.transpose(-2, -1)`将K的最后两个维度（`S`和`d_k`）交换，得到 `[B, H, d_k, S]`。`[B, H, S, d_k]` @ `[B, H, d_k, S]` -> `[B, H, S, S]`。
    *   `scores = scores.masked_fill(mask == 0, -1e9)`: `masked_fill`是一个PyTorch函数。它会检查`mask == 0`的条件，在所有条件为`True`的位置，用`-1e9`（一个非常小的数）填充`scores`张量。
    *   `attn_weights = torch.softmax(scores, dim=-1)`: 沿着最后一个维度（键的维度）计算softmax，确保对于每个查询，其对所有键的注意力权重之和为1。
    *   `output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)`: 这是拆分头的逆操作。
        1.  `.transpose(1, 2)`: 换回头和序列长度维度，得到 `[B, S, H, d_k]`。
        2.  `.contiguous()`: 在进行`view`之前，需要确保张量在内存中是连续的。`transpose`操作可能会导致内存不连续。
        3.  `.view(B, -1, D)`: 将`H`和`d_k`维度重新合并为`D`（`d_model`），得到最终的 `[B, S, D]`。
    *   `output = self.out_proj(output)`: 通过最后的线性层，整合所有头的信息。

---
好的，我们继续构建剩下的核心组件，并开始将它们组装成编码器和解码器层。

---

#### **1.3 位置全连接前馈网络 (Position-wise Feed-Forward Network)**

##### **代码流程详述**
这个模块相对简单，它的作用是对注意力层的输出进行进一步的非线性处理。
1.  **初始化 (`__init__`)**:
    *   接收`d_model`、一个更大的中间层维度`d_ff`（通常是`4 * d_model`）以及`dropout`率。
    *   创建两个线性层：第一个`linear1`将维度从`d_model`扩展到`d_ff`，第二个`linear2`将维度从`d_ff`缩减回`d_model`。
    *   在它们之间放置一个Dropout层和一个ReLU激活函数。
2.  **前向传播 (`forward`)**:
    *   按顺序将输入`x`通过`linear1` -> `relu` -> `dropout` -> `linear2`。

##### **代码实现 (`model.py`)**
```python
# model.py (continued)

class PositionWiseFeedForward(nn.Module):
    """
    Implements the Position-wise Feed-Forward Network.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
```

##### **代码解释**
*   **`__init__`**:
    *   `self.linear1 = nn.Linear(d_model, d_ff)`: 定义第一个线性层，实现维度的扩展。
    *   `self.linear2 = nn.Linear(d_ff, d_model)`: 定义第二个线性层，实现维度的缩减。
    *   `self.relu = nn.ReLU()`: 定义ReLU激活函数。
*   **`forward`**:
    *   代码直接反映了数据流：`x`首先通过`linear1`和`relu`进行非线性变换和特征提取，然后通过`dropout`进行正则化，最后通过`linear2`将特征投影回原始的`d_model`维度，以便于后续的残差连接。

---

#### **1.4 编码器层 (Encoder Layer)**

##### **代码流程详述**
现在我们将前面构建的`MultiHeadAttention`和`PositionWiseFeedForward`模块，与残差连接（Add）和层归一化（Norm）组合起来，形成一个完整的编码器层。
1.  **初始化 (`__init__`)**:
    *   接收`d_model`, `n_head`, `d_ff`, `dropout`等超参数。
    *   实例化一个`MultiHeadAttention`模块作为自注意力子层。
    *   实例化一个`PositionWiseFeedForward`模块作为前馈网络子层。
    *   创建两个`nn.LayerNorm`模块，分别用于两个子层的输出。
    *   创建一个`nn.Dropout`模块，用于残差连接。
2.  **前向传播 (`forward`)**:
    *   接收输入`x`和一个可选的`mask`（对于编码器，这个mask通常用于屏蔽`<PAD>`填充位）。
    *   **第一个子层 (Self-Attention)**:
        *   计算自注意力输出。注意，Q, K, V都来自同一个输入`x`。
        *   将注意力输出通过dropout层。
        *   将dropout后的结果与原始输入`x`相加（第一个残差连接）。
        *   将相加后的结果通过第一个层归一化模块。
    *   **第二个子层 (Feed-Forward)**:
        *   将第一个子层的输出送入前馈网络。
        *   将FFN的输出通过dropout层。
        *   将dropout后的结果与FFN的**输入**相加（第二个残差连接）。
        *   将相加后的结果通过第二个层归一化模块。
    *   返回最终结果。

##### **代码实现 (`model.py`)**
```python
# model.py (continued)

class EncoderLayer(nn.Module):
    """
    A single layer of the Transformer Encoder.
    Consists of a multi-head self-attention sublayer and a position-wise feed-forward sublayer.
    """
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, seq_len, d_model)
        # mask shape: (batch_size, 1, 1, seq_len) for padding mask
        
        # --- Self-Attention Sublayer ---
        # 1. Compute self-attention
        attn_output = self.self_attn(q=x, k=x, v=x, mask=mask)
        # 2. Apply dropout and residual connection
        x = x + self.dropout(attn_output)
        # 3. Apply layer normalization
        x = self.norm1(x)

        # --- Feed-Forward Sublayer ---
        # 1. Compute feed-forward output
        ff_output = self.feed_forward(x)
        # 2. Apply dropout and residual connection
        x = x + self.dropout(ff_output)
        # 3. Apply layer normalization
        x = self.norm2(x)
        
        return x
```

##### **代码解释**
*   **`__init__`**:
    *   `self.self_attn = ...`: 实例化多头注意力模块。
    *   `self.feed_forward = ...`: 实例化前馈网络模块。
    *   `self.norm1`, `self.norm2`: 实例化两个独立的层归一化模块。为每个子层使用独立的归一化层是一种标准实践。
*   **`forward`**:
    *   `attn_output = self.self_attn(q=x, k=x, v=x, mask=mask)`: 调用自注意力模块。因为是**自**注意力，所以Q, K, V都来自于同一个源`x`。
    *   `x = x + self.dropout(attn_output)`: 这是残差连接的核心。我们将子层（注意力）的输出先通过dropout，然后加回到子层的**原始输入**`x`上。
    *   `x = self.norm1(x)`: 对残差连接的结果进行层归一化。
    *   接下来的Feed-Forward子层遵循完全相同的模式：`FFN -> Dropout -> Add -> Norm`。注意，第二个残差连接加的是FFN的输入（即第一个子层的输出）。

---

#### **1.5 解码器层 (Decoder Layer)**

##### **代码流程详述**
解码器层比编码器层复杂，因为它有三个子层。
1.  **初始化 (`__init__`)**:
    *   实例化**两个**`MultiHeadAttention`模块：一个用于带掩码的自注意力（`self_attn`），另一个用于编码器-解码器注意力（`cross_attn`）。
    *   实例化一个`PositionWiseFeedForward`模块。
    *   创建**三个**`nn.LayerNorm`模块，每个子层一个。
    *   创建一个`nn.Dropout`模块。
2.  **前向传播 (`forward`)**:
    *   接收解码器输入`x`、编码器输出`encoder_output`，以及两个掩码：`src_mask`（源序列的填充掩码）和`tgt_mask`（目标序列的前瞻掩码和填充掩码的组合）。
    *   **第一个子层 (Masked Self-Attention)**:
        *   与编码器层类似，但调用自注意力时传入`tgt_mask`来防止看到未来。
        *   应用Dropout、残差连接和层归一化。
    *   **第二个子层 (Encoder-Decoder Attention)**:
        *   调用`cross_attn`模块。关键在于：**Q**来自前一个子层的输出，而**K**和**V**都来自`encoder_output`。
        *   传入`src_mask`来屏蔽源序列中的填充位。
        *   应用Dropout、残差连接和层归一化。
    *   **第三个子层 (Feed-Forward)**:
        *   与编码器层完全相同，对第二个子层的输出进行处理。
        *   应用Dropout、残差连接和层归一化。
    *   返回最终结果。

##### **代码实现 (`model.py`)**
```python
# model.py (continued)

class DecoderLayer(nn.Module):
    """
    A single layer of the Transformer Decoder.
    Consists of three sublayers: masked multi-head self-attention, 
    encoder-decoder multi-head attention, and position-wise feed-forward.
    """
    def __init__(self, d_model: int, n_head: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.feed_forward = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, tgt_seq_len, d_model)
        # encoder_output shape: (batch_size, src_seq_len, d_model)
        # src_mask shape: (batch_size, 1, 1, src_seq_len)
        # tgt_mask shape: (batch_size, 1, tgt_seq_len, tgt_seq_len)

        # --- Masked Self-Attention Sublayer ---
        attn_output = self.self_attn(q=x, k=x, v=x, mask=tgt_mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)

        # --- Encoder-Decoder Attention Sublayer ---
        # Query from decoder, Key and Value from encoder
        cross_attn_output = self.cross_attn(q=x, k=encoder_output, v=encoder_output, mask=src_mask)
        x = x + self.dropout(cross_attn_output)
        x = self.norm2(x)

        # --- Feed-Forward Sublayer ---
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm3(x)
        
        return x
```

##### **代码解释**
*   **`__init__`**:
    *   `self.self_attn`和`self.cross_attn`是两个独立的`MultiHeadAttention`实例，它们将学习不同的权重。
    *   `self.norm1`, `self.norm2`, `self.norm3`: 对应三个子层的三个独立的层归一化模块。
*   **`forward`**:
    *   `attn_output = self.self_attn(q=x, k=x, v=x, mask=tgt_mask)`: 第一个子层对解码器自身的输入`x`进行自注意力计算，并使用`tgt_mask`来屏蔽未来信息。
    *   `cross_attn_output = self.cross_attn(q=x, k=encoder_output, v=encoder_output, mask=src_mask)`: 这是第二个子层，是解码器与编码器交互的地方。
        *   `q=x`: 查询来自于解码器（经过第一个子层处理后的`x`）。
        *   `k=encoder_output`, `v=encoder_output`: 键和值都来自于编码器的最终输出。
        *   `mask=src_mask`: 使用源序列的填充掩码，确保注意力不会计算在源句子的填充位上。
    *   剩下的部分，包括残差连接和层归一化，以及第三个FFN子层，都遵循与编码器层相同的逻辑。

---
好的，现在我们已经拥有了所有的基础组件，是时候将它们组装成一个完整的、端到端的Transformer模型了。

---

### **第二部分：组装完整的Transformer模型**

在这一部分，我们将使用上一部分创建的`EncoderLayer`和`DecoderLayer`来构建完整的编码器和解码器。然后，我们会将编码器、解码器以及输入/输出层整合到一个顶层的`Transformer`类中。

#### **2.1 编码器 (Encoder)**

##### **代码流程详述**
编码器的作用就是将N个`EncoderLayer`堆叠在一起。
1.  **初始化 (`__init__`)**:
    *   接收词汇表大小、`d_model`、`n_layer`（编码器层数）、`n_head`、`d_ff`和`dropout`等超参数。
    *   创建输入层，由`TokenEmbedding`和`PositionalEncoding`组成。
    *   使用`nn.ModuleList`来存放N个`EncoderLayer`。`nn.ModuleList`是一个可以像Python列表一样索引的模块容器，它能确保所有子模块都被PyTorch正确注册。
2.  **前向传播 (`forward`)**:
    *   接收源序列索引`src`和源序列掩码`src_mask`。
    *   将`src`通过输入层（词嵌入 + 位置编码）。
    *   使用一个循环，将数据依次传递过`nn.ModuleList`中的每一个`EncoderLayer`。
    *   返回最后一层的输出。

##### **代码实现 (`model.py`)**
```python
# model.py (continued)

class Encoder(nn.Module):
    """
    The Encoder part of the Transformer model.
    Consists of an embedding layer and N stacked EncoderLayers.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            EncoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        # src shape: (batch_size, src_seq_len)
        # src_mask shape: (batch_size, 1, 1, src_seq_len)
        
        # 1. Apply token embedding and positional encoding
        x = self.token_embedding(src)
        x = self.positional_encoding(x)
        
        # 2. Pass through N encoder layers
        for layer in self.layers:
            x = layer(x, src_mask)
            
        # 3. Apply a final layer normalization
        x = self.norm(x)
        
        return x
```

##### **代码解释**
*   **`__init__`**:
    *   `self.token_embedding`, `self.positional_encoding`: 实例化我们之前定义的输入层模块。
    *   `self.layers = nn.ModuleList([...])`: 这是一个列表推导式，它会创建`n_layer`个`EncoderLayer`实例，并将它们存放在`nn.ModuleList`中。
    *   `self.norm = nn.LayerNorm(d_model)`: 在原版Transformer论文中，编码器和解码器的最后还会额外应用一次层归一化。我们遵循这个设计。
*   **`forward`**:
    *   `x = self.token_embedding(src)`: 将输入的索引序列转换为词嵌入。
    *   `x = self.positional_encoding(x)`: 注入位置信息。
    *   `for layer in self.layers: ...`: 这是一个简洁的循环，它按顺序将`x`传递过堆叠的每一个编码器层，每一层的输出成为下一层的输入。
    *   `x = self.norm(x)`: 应用最后的层归一化。

---

#### **2.2 解码器 (Decoder)**

##### **代码流程详述**
解码器的结构与编码器非常相似，也是堆叠N个`DecoderLayer`。
1.  **初始化 (`__init__`)**:
    *   与编码器类似，接收所有必要的超参数。
    *   创建输入层（`TokenEmbedding` + `PositionalEncoding`）。
    *   使用`nn.ModuleList`来存放N个`DecoderLayer`。
2.  **前向传播 (`forward`)**:
    *   接收目标序列索引`tgt`、编码器输出`encoder_output`、源序列掩码`src_mask`和目标序列掩码`tgt_mask`。
    *   将`tgt`通过输入层。
    *   使用一个循环，将数据和掩码依次传递过每一个`DecoderLayer`。
    *   返回最后一层的输出。

##### **代码实现 (`model.py`)**
```python
# model.py (continued)

class Decoder(nn.Module):
    """
    The Decoder part of the Transformer model.
    Consists of an embedding layer and N stacked DecoderLayers.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.token_embedding = TokenEmbedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, n_head, d_ff, dropout) for _ in range(n_layer)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, tgt: torch.Tensor, encoder_output: torch.Tensor, src_mask: torch.Tensor, tgt_mask: torch.Tensor) -> torch.Tensor:
        # tgt shape: (batch_size, tgt_seq_len)
        # encoder_output shape: (batch_size, src_seq_len, d_model)
        # src_mask shape: (batch_size, 1, 1, src_seq_len)
        # tgt_mask shape: (batch_size, 1, tgt_seq_len, tgt_seq_len)
        
        # 1. Apply token embedding and positional encoding
        x = self.token_embedding(tgt)
        x = self.positional_encoding(x)
        
        # 2. Pass through N decoder layers
        for layer in self.layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)
            
        # 3. Apply a final layer normalization
        x = self.norm(x)
        
        return x
```

##### **代码解释**
*   解码器的实现与编码器高度对称，这体现了Transformer设计的优雅之处。
*   主要区别在于`forward`方法接收的参数更多，因为它需要编码器的输出（`encoder_output`）以及两种掩码（`src_mask`, `tgt_mask`），并将它们正确地传递给每一个`DecoderLayer`。

---

#### **2.3 完整的Transformer模型**

##### **代码流程详述**
现在，我们将所有部分整合到一个最终的`Transformer`类中。
1.  **初始化 (`__init__`)**:
    *   接收所有模型超参数。
    *   实例化一个`Encoder`模块。
    *   实例化一个`Decoder`模块。
    *   创建一个最终的线性层`output_proj`，用于将解码器的输出映射到词汇表大小的logits。
2.  **辅助函数 (`_generate_mask`)**:
    *   创建一个私有辅助方法来生成掩码。这个方法将负责创建源序列的填充掩码和目标序列的前瞻+填充掩码。
        *   **源掩码**: 只需要屏蔽`<PAD>`位。
        *   **目标掩码**: 需要结合两方面的信息：(1) 屏蔽`<PAD>`位；(2) 创建一个下三角矩阵来屏蔽未来的词元。
3.  **前向传播 (`forward`)**:
    *   接收源序列`src`和目标序列`tgt`。
    *   调用`_generate_mask`方法创建所有需要的掩码。
    *   将`src`和`src_mask`传递给编码器，得到`encoder_output`。
    *   将`tgt`、`encoder_output`以及两种掩码传递给解码器，得到`decoder_output`。
    *   将`decoder_output`通过最终的线性投影层，得到logits。
    *   返回logits。

##### **代码实现 (`model.py`)**
```python
# model.py (continued)

class Transformer(nn.Module):
    """
    The complete Transformer model architecture.
    """
    def __init__(self, vocab_size: int, d_model: int, n_layer: int, n_head: int, d_ff: int, dropout: float):
        super().__init__()
        self.encoder = Encoder(vocab_size, d_model, n_layer, n_head, d_ff, dropout)
        self.decoder = Decoder(vocab_size, d_model, n_layer, n_head, d_ff, dropout)
        self.output_proj = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        # Initialize parameters with Glorot / fan_avg.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        # src shape: (batch_size, src_seq_len)
        # tgt shape: (batch_size, tgt_seq_len)
        
        # 1. Create masks
        # For our text generation task, src and tgt are the same.
        # However, we build a general Transformer that could be used for translation.
        # In this specific case, src_mask and tgt_padding_mask will be identical.
        src_padding_mask = (src != 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, S)
        tgt_padding_mask = (tgt != 0).unsqueeze(1).unsqueeze(2) # (B, 1, 1, T)
        
        seq_len = tgt.size(1)
        look_ahead_mask = torch.tril(torch.ones(seq_len, seq_len, device=tgt.device)).bool() # (T, T)
        
        # Combine padding mask and look-ahead mask for the target
        tgt_mask = tgt_padding_mask & look_ahead_mask # (B, 1, T, T)

        # 2. Pass inputs through encoder
        encoder_output = self.encoder(src, src_padding_mask)
        
        # 3. Pass encoder output and target through decoder
        decoder_output = self.decoder(tgt, encoder_output, src_padding_mask, tgt_mask)
        
        # 4. Final linear projection
        output = self.output_proj(decoder_output)
        
        return output
```

##### **代码解释**
*   **`__init__`**:
    *   `self.encoder = Encoder(...)`, `self.decoder = Decoder(...)`: 实例化编码器和解码器。
    *   `self.output_proj = nn.Linear(d_model, vocab_size)`: 创建最终的输出层。
    *   `self._initialize_weights()`: 调用一个辅助函数来初始化模型的权重。使用Xavier/Glorot初始化是一种常见的做法，有助于模型在训练初期稳定收敛。
*   **`forward`**:
    *   **Mask Creation**:
        *   `src_padding_mask = (src != 0).unsqueeze(1).unsqueeze(2)`: 这是创建填充掩码的关键。
            1. `(src != 0)`: 假设我们的`<PAD>`索引是0。这个操作会创建一个布尔张量，`<PAD>`位是`False`，其他位是`True`。形状 `[B, S]`。
            2. `.unsqueeze(1).unsqueeze(2)`: 增加两个维度，使其形状变为 `[B, 1, 1, S]`，以便能与多头注意力内部的 `[B, H, S, S]` 分数矩阵进行广播。
        *   `look_ahead_mask = torch.tril(...)`: `torch.tril`创建一个下三角矩阵。这确保了每个位置只能关注到自己和之前的位置。
        *   `tgt_mask = tgt_padding_mask & look_ahead_mask`: 通过逻辑与操作，将填充掩码和前瞻掩码合并。最终的`tgt_mask`在需要屏蔽的位置（填充位 或 未来位）为`False`。
    *   **Data Flow**:
        *   `encoder_output = self.encoder(...)`: 调用编码器。
        *   `decoder_output = self.decoder(...)`: 调用解码器。
        *   `output = self.output_proj(...)`: 计算最终的logits。
    *   **关于文本生成的说明**: 在我们的任务中，源序列和目标序列实际上是同一个文本块的不同部分（`x`和`y`）。因此，`src`和`tgt`的填充掩码会是相同的。我们这里构建了一个更通用的框架，如果未来要用于机器翻译，代码无需大的改动。

---
好的，现在我们已经构建了完整的模型，接下来将编写训练脚本来训练它。

---

### **第三部分：模型训练**

在这一部分，我们将创建一个名为 `train.py` 的主脚本。这个脚本将负责：
1.  加载和准备数据。
2.  初始化我们构建的Transformer模型。
3.  定义损失函数和优化器。
4.  执行训练循环，迭代数据、计算损失、更新模型参数。
5.  保存训练好的模型权重。

#### **3.1 训练脚本 (`train.py`) 的结构**

##### **代码流程详述**
我们的训练脚本将遵循一个标准的PyTorch训练流程。
1.  **导入**: 导入所有必要的库，包括`torch`、我们自己写的`config`、`utils`、`dataset`和`model`。
2.  **主训练函数 `train()`**:
    *   **设置**:
        *   从`config`中获取设备（CPU/GPU）。
        *   设置随机种子以保证实验的可复现性。
    *   **数据加载**:
        *   读取`tiny_shakespeare.txt`文件。
        *   根据文本创建`CharacterVocabulary`实例。
        *   将文本数据划分为训练集和验证集。
        *   使用`create_dataloader`函数为训练集和验证集创建`DataLoader`。
    *   **模型初始化**:
        *   根据`config`中的超参数实例化`Transformer`模型。
        *   将模型移动到指定的设备。
    *   **损失函数与优化器**:
        *   定义损失函数。我们将使用`nn.CrossEntropyLoss`。一个关键点是设置`ignore_index`参数，使其在计算损失时忽略掉填充（`<PAD>`）词元。
        *   定义优化器。我们将使用AdamW，这是一种对Adam优化器的改进版本，在Transformer的训练中表现良好。
    *   **训练循环**:
        *   外层循环遍历所有`epoch`。
        *   内层循环是`train_one_epoch`函数，负责处理一个epoch的训练数据。
        *   （可选，为简化起见，我们暂时省略验证循环，但会指出其位置）。
    *   **模型保存**:
        *   训练结束后，使用`torch.save`保存模型的`state_dict`（即所有可学习的参数）。

##### **代码实现 (`train.py`)**
```python
# train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import config
from utils import CharacterVocabulary
from dataset import TextDataset, create_dataloader
from model import Transformer

import time
import math

def train_one_epoch(model, dataloader, loss_fn, optimizer, device):
    """
    Performs one full training pass over the dataset.
    """
    model.train() # Set the model to training mode
    total_loss = 0.0
    
    # Iterate over batches
    for i, (src, tgt) in enumerate(dataloader):
        # In our text generation task, src and tgt are derived from the same sequence.
        # tgt_input is the target sequence shifted right (starts with <SOS>)
        # tgt_output is the target sequence (ends with <EOS>)
        # For this implementation, we simplify: src is input, tgt is the target to predict.
        # The model's forward pass will handle the "teacher forcing" aspect internally.
        
        src, tgt = src.to(device), tgt.to(device)

        # The target for the model's forward pass is the sequence shifted right.
        # The actual labels for loss calculation is the original target sequence.
        # Let's assume tgt is structured as [y1, y2, ..., yN]
        # The input to the decoder should be [<SOS>, y1, y2, ..., yN-1]
        # The target for the loss function should be [y1, y2, ..., yN, <EOS>]
        # Our current dataset implementation simplifies this:
        # src = text[0...N-1], tgt = text[1...N]
        
        # Zero the gradients
        optimizer.zero_grad()
        
        # Forward pass
        logits = model(src, tgt) # In our simple case, tgt is used for teacher forcing
        
        # Calculate loss
        # We need to reshape logits and tgt for CrossEntropyLoss
        # Logits: (batch_size, seq_len, vocab_size) -> (batch_size * seq_len, vocab_size)
        # Tgt: (batch_size, seq_len) -> (batch_size * seq_len)
        loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        
        # Update weights
        optimizer.step()
        
        total_loss += loss.item()

        if i % 200 == 0 and i > 0:
            print(f"| Batch {i}/{len(dataloader)} | Loss: {loss.item():.4f}")

    return total_loss / len(dataloader)


def main():
    """
    Main function to orchestrate the training process.
    """
    # --- 1. Setup ---
    torch.manual_seed(42)
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # --- 2. Data Loading ---
    print("Loading data and creating vocabulary...")
    with open(config.DATA_PATH, 'r', encoding='utf-8') as f:
        text = f.read()
    
    vocab = CharacterVocabulary(text)
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")

    # Split data into training and validation sets
    n = len(text)
    train_text = text[:int(n * config.TRAIN_DATA_RATIO)]
    # val_text = text[int(n * config.TRAIN_DATA_RATIO):] # We'll skip validation for simplicity

    train_dataloader = create_dataloader(
        train_text, vocab, config.BLOCK_SIZE, config.BATCH_SIZE
    )
    # val_dataloader = create_dataloader(val_text, vocab, config.BLOCK_SIZE, config.BATCH_SIZE)

    # --- 3. Model Initialization ---
    print("Initializing model...")
    model = Transformer(
        vocab_size=vocab_size,
        d_model=config.D_MODEL,
        n_layer=config.N_LAYER,
        n_head=config.N_HEAD,
        d_ff=4 * config.D_MODEL, # Standard practice
        dropout=config.DROPOUT
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    # --- 4. Loss Function and Optimizer ---
    loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)

    # --- 5. Training Loop ---
    print("Starting training...")
    for epoch in range(1, config.NUM_EPOCHS + 1):
        start_time = time.time()
        
        avg_train_loss = train_one_epoch(model, train_dataloader, loss_fn, optimizer, device)
        
        end_time = time.time()
        epoch_mins, epoch_secs = divmod(end_time - start_time, 60)
        
        print("-" * 50)
        print(f"| End of Epoch: {epoch:02} | Time: {int(epoch_mins)}m {int(epoch_secs)}s |")
        print(f"| Train Loss: {avg_train_loss:.4f} | Train PPL: {math.exp(avg_train_loss):.4f} |")
        print("-" * 50)

    # --- 6. Model Saving ---
    print("Training finished. Saving model...")
    torch.save(model.state_dict(), "transformer_model.pth")
    # Also save the vocabulary for inference
    import pickle
    with open("vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("Model and vocabulary saved.")


if __name__ == "__main__":
    main()
```

##### **代码解释**
*   **`train_one_epoch` 函数**:
    *   `model.train()`: 这是一个非常重要的调用。它告诉PyTorch模型正处于训练模式。这会激活Dropout层和（如果有的话）BatchNorm层。
    *   `for i, (src, tgt) in enumerate(dataloader)`: 循环从`DataLoader`中获取一批数据。`src`和`tgt`就是我们`TextDataset`中返回的`(x, y)`对。
    *   `src, tgt = src.to(device), tgt.to(device)`: 将数据张量移动到配置的设备（GPU或CPU）。
    *   `optimizer.zero_grad()`: 在计算新梯度之前，必须清除上一批次计算的旧梯度。
    *   `logits = model(src, tgt)`: 这是核心的前向传播步骤。**注意**：在我们的简单文本生成设置中，`src`是解码器的输入（teacher forcing），`tgt`是用于计算损失的标签。在`model.forward`中，`src`和`tgt`都被用来生成掩码，但只有`tgt`被解码器实际处理。为了代码清晰，我们可以认为`model(src, tgt)`的`src`参数在这里被忽略了，因为解码器只关心`tgt`和它自己生成的掩码。一个更通用的实现可能会将`tgt`拆分为`tgt_input`和`tgt_label`。
    *   `loss = loss_fn(logits.view(-1, logits.size(-1)), tgt.view(-1))`:
        *   `nn.CrossEntropyLoss`期望的输入形状是 `(N, C)`，其中N是样本数，C是类别数。
        *   我们的`logits`形状是 `[B, S, V]` (B=batch, S=seq_len, V=vocab_size)。
        *   我们的`tgt`形状是 `[B, S]`。
        *   `.view(-1, ...)`将它们展平为 `[B*S, V]` 和 `[B*S]`，这正是损失函数需要的形状。
    *   `loss.backward()`: 计算损失相对于模型所有参数的梯度。
    *   `torch.nn.utils.clip_grad_norm_(...)`: 这是一个防止梯度爆炸的常用技巧。它会将梯度的范数（norm）限制在一个最大值（这里是0.5）内。
    *   `optimizer.step()`: 根据计算出的梯度，使用AdamW算法更新模型的权重。
    *   `total_loss += loss.item()`: `.item()`从张量中提取出Python标量值。我们累加损失以便计算整个epoch的平均损失。
*   **`main` 函数**:
    *   `torch.manual_seed(42)`: 设置随机种子，确保每次运行代码时，权重的随机初始化和数据的打乱顺序都是一样的，这对于调试和复现结果至关重要。
    *   `with open(...)`: 打开并读取文本文件。
    *   `vocab = CharacterVocabulary(text)`: 创建词汇表。
    *   `model = Transformer(...)`: 实例化模型。
    *   `loss_fn = nn.CrossEntropyLoss(ignore_index=vocab.pad_idx)`: **关键点**。`ignore_index`告诉损失函数，当目标标签`tgt`中的值为`vocab.pad_idx`时，不要计算该位置的损失。这确保了我们不会因为模型对填充位的预测而惩罚模型。
    *   `optimizer = torch.optim.AdamW(...)`: AdamW通常比标准Adam在Transformer上效果更好。
    *   `print(f"| Train PPL: {math.exp(avg_train_loss):.4f} |")`: 打印困惑度（Perplexity），它是交叉熵损失的指数，是评估语言模型性能的常用指标，更具可解释性。
    *   `torch.save(model.state_dict(), ...)`: 保存模型的**状态字典**，它只包含模型的参数（权重和偏置），而不包含整个模型结构。这是推荐的保存方式。
    *   `pickle.dump(vocab, f)`: 我们必须保存词汇表对象，因为在推理时，我们需要用它来将用户输入的文本编码为模型能理解的索引。

好的，现在模型已经训练完毕并保存了权重，我们将进入最后一步：使用训练好的模型来生成新的文本。

---

### **第四部分：文本生成（推理）**

在这一部分，我们将创建 `generate.py` 脚本。这个脚本将加载我们之前保存的模型权重和词汇表，并实现一个函数来执行自回归（auto-regressive）文本生成。这意味着模型将逐个字符地生成文本，并将每个新生成的字符作为下一步的输入。

#### **4.1 生成脚本 (`generate.py`) 的结构**

##### **代码流程详述**
1.  **导入**: 导入必要的库，包括`torch`、`pickle`（用于加载词汇表）、`config`以及我们的`model`模块。
2.  **主生成函数 `generate()`**:
    *   **设置**:
        *   从`config`中获取设备。
    *   **加载工件 (Artifacts)**:
        *   使用`pickle`加载之前保存的`vocab.pkl`文件，重建词汇表对象。
        *   获取词汇表大小。
    *   **模型加载**:
        *   根据`config`中的超参数重新实例化`Transformer`模型。
        *   使用`model.load_state_dict()`加载我们训练好的权重`transformer_model.pth`。
        *   将模型移动到指定的设备，并调用`model.eval()`。
    *   **生成循环**:
        *   定义一个起始文本（prompt），例如 "O Romeo, Romeo, wherefore art thou Romeo?"。
        *   调用一个核心的生成函数（我们称之为`run_generation`），传入模型、词汇表、起始文本和要生成的最大长度。
        *   打印生成的文本。

#### **4.2 实现生成函数**

##### **代码流程详述 (`run_generation` 函数)**
这是推理过程的核心。它模拟了模型在没有"标准答案"的情况下如何一步步地构建输出序列。
1.  **初始化**:
    *   将模型设置为评估模式 `model.eval()`。这会关闭Dropout等训练特有的层。
    *   使用词汇表将起始文本（prompt）编码为整数索引，并将其转换为PyTorch张量。这个张量就是我们解码过程的初始`tgt`序列。
2.  **自回归循环**:
    *   循环指定的次数（`max_len`）。
    *   在循环的每一步：
        *   **准备输入**: 确保输入序列的长度不超过模型的`BLOCK_SIZE`。如果超过，就截断它，只保留最后的`BLOCK_SIZE`个词元。
        *   **前向传播**: 将当前的序列张量送入模型。**注意**：在推理时，我们只有一个序列，所以`src`和`tgt`是同一个东西。模型的前向传播将计算出序列中**最后一个**位置的logits，这代表了对下一个词元的预测。
        *   **获取下一个词元的预测**: 从输出的logits中，只选择最后一个时间步的logits `[-1, :]`。
        *   **应用Softmax**: 将这些logits通过Softmax函数转换为概率分布。
        *   **选择下一个词元 (Greedy Search)**: 使用`torch.argmax`从概率分布中选出概率最高的那个词元的索引。
        *   **更新序列**: 将新选出的词元索引拼接到当前序列的末尾。
        *   **检查结束条件**: 如果生成的词元是`<EOS>`，则停止生成。
3.  **解码与返回**:
    *   循环结束后，使用词汇表的`decode`方法将整个整数索引序列转换回人类可读的文本字符串。
    *   返回生成的文本。

##### **代码实现 (`generate.py`)**
```python
# generate.py

import torch
import pickle

import config
from model import Transformer
from utils import CharacterVocabulary

def run_generation(model, vocab, prompt, max_len, device):
    """
    Generates text using the trained Transformer model.
    """
    model.eval() # Set the model to evaluation mode

    # Encode the starting prompt
    input_indices = vocab.encode(prompt)
    
    # Convert to a tensor and add a batch dimension
    # Shape: (1, seq_len)
    tgt = torch.tensor(input_indices, dtype=torch.long, device=device).unsqueeze(0)

    # Generation loop
    for _ in range(max_len):
        # Ensure the input sequence doesn't exceed the model's block size
        # We take the last `BLOCK_SIZE` tokens as context
        tgt_for_model = tgt[:, -config.BLOCK_SIZE:]

        # Perform a forward pass with no gradient calculation
        with torch.no_grad():
            # The model expects both src and tgt. In inference, they are the same.
            logits = model(tgt_for_model, tgt_for_model)
        
        # Get the logits for the very last token
        # Logits shape: (1, current_seq_len, vocab_size)
        # We only care about the prediction for the next token
        last_logits = logits[:, -1, :] # Shape: (1, vocab_size)
        
        # Apply softmax to get probabilities
        probs = torch.softmax(last_logits, dim=-1)
        
        # Get the index of the token with the highest probability (Greedy Search)
        next_token_idx = torch.argmax(probs, dim=-1) # Shape: (1)
        
        # Check if the model generated the End-Of-Sequence token
        if next_token_idx.item() == vocab.eos_idx:
            break
            
        # Append the predicted token to the sequence
        # Shape becomes (1, current_seq_len + 1)
        tgt = torch.cat([tgt, next_token_idx.unsqueeze(0)], dim=1)

    # Decode the generated sequence of indices back to a string
    generated_text = vocab.decode(tgt.squeeze(0).tolist())
    
    return generated_text


def main():
    """
    Main function to load the model and run text generation.
    """
    # --- 1. Setup ---
    device = torch.device(config.DEVICE)
    print(f"Using device: {device}")

    # --- 2. Load Vocabulary and Model ---
    print("Loading vocabulary and model...")
    try:
        with open("vocab.pkl", "rb") as f:
            vocab = pickle.load(f)
        vocab_size = len(vocab)
        
        model = Transformer(
            vocab_size=vocab_size,
            d_model=config.D_MODEL,
            n_layer=config.N_LAYER,
            n_head=config.N_HEAD,
            d_ff=4 * config.D_MODEL,
            dropout=config.DROPOUT
        ).to(device)
        
        model.load_state_dict(torch.load("transformer_model.pth", map_location=device))
        print("Model and vocabulary loaded successfully.")
    except FileNotFoundError:
        print("Error: `transformer_model.pth` or `vocab.pkl` not found.")
        print("Please run `train.py` first to train and save the model.")
        return

    # --- 3. Generate Text ---
    prompt = "O Romeo, Romeo, wherefore art thou Romeo?"
    max_len_to_generate = 500
    
    print("\n--- Starting Generation ---")
    print(f"Prompt: {prompt}")
    
    generated_text = run_generation(model, vocab, prompt, max_len_to_generate, device)
    
    print("\n--- Generated Text ---")
    print(generated_text)
    print("\n------------------------")


if __name__ == "__main__":
    main()
```

##### **代码解释**
*   **`run_generation` 函数**:
    *   `model.eval()`: 同样非常重要。它将模型切换到评估模式，关闭Dropout等。
    *   `tgt = torch.tensor(...).unsqueeze(0)`: 将编码后的prompt列表转换为张量，并使用`.unsqueeze(0)`添加一个批次维度，因为我们的模型期望的输入形状是 `[batch_size, seq_len]`。
    *   `for _ in range(max_len)`: 循环生成最多`max_len`个新字符。
    *   `tgt_for_model = tgt[:, -config.BLOCK_SIZE:]`: 这是一个重要的细节。如果生成的序列比模型的`BLOCK_SIZE`长，我们不能将整个序列都喂给模型，因为位置编码只定义到了`BLOCK_SIZE`。我们只取最后的`BLOCK_SIZE`个字符作为上下文。
    *   `with torch.no_grad()`: 这是一个上下文管理器，它告诉PyTorch在接下来的代码块中不要计算梯度。这可以显著减少内存消耗并加速计算，因为推理过程不需要反向传播。
    *   `logits = model(tgt_for_model, tgt_for_model)`: 调用模型的前向传播。在推理时，`src`和`tgt`是相同的，即模型根据已有的序列来预测下一个。
    *   `last_logits = logits[:, -1, :]`: 我们只对预测**下一个**字符感兴趣，所以我们只从logits张量 `[1, S, V]` 中取出最后一个时间步 `S-1` (在Python中索引为`-1`) 的结果。
    *   `next_token_idx = torch.argmax(probs, dim=-1)`: `argmax`返回指定维度上最大值的索引。这里我们找到了概率最高的词元的索引。
    *   `tgt = torch.cat([tgt, next_token_idx.unsqueeze(0)], dim=1)`: `torch.cat`用于拼接张量。我们将新预测出的词元索引（需要调整形状以匹配）拼接到`tgt`张量的末尾，为下一轮循环做准备。
    *   `vocab.decode(tgt.squeeze(0).tolist())`:
        1.  `.squeeze(0)`: 移除批次维度，将形状从 `[1, S]` 变为 `[S]`。
        2.  `.tolist()`: 将PyTorch张量转换为Python列表。
        3.  `vocab.decode(...)`: 调用我们之前写的解码函数，将索引列表变回字符串。
*   **`main` 函数**:
    *   `try...except FileNotFoundError`: 添加了错误处理，如果用户没有先训练模型，会给出友好的提示。
    *   `model.load_state_dict(torch.load("transformer_model.pth", map_location=device))`:
        *   `torch.load(...)`: 加载保存在磁盘上的模型权重。
        *   `map_location=device`: 这是一个好习惯，它确保了即使模型是在GPU上训练和保存的，也能在只有CPU的机器上成功加载。
        *   `model.load_state_dict(...)`: 将加载的权重应用到我们新创建的模型实例上。

现在，您可以运行 `python generate.py` 来查看您的Transformer模型的创作了！您可以通过修改`prompt`和`max_len_to_generate`来与模型进行不同的交互。

---
