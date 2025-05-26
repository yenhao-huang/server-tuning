
## Model Training Summary

* **Loss Function**: `CrossEntropyLoss`
* **Optimizer**: `Adam`
* **Input Shape**: `(batch_size, 61, 103)`
* **#Classes**: `6`

---

### Model 1: DNN

* **Model Type**: 4 FC layers
* **#Parameters**: **\~14 M**
* **Model Size**: **53.36 MB**

#### 🔧 Architecture

```=py
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
         Flatten-1               [B, 6283]                   0
          Linear-2               [B, 2048]             12,872,192
       LeakyReLU-3               [B, 2048]                   0
          Linear-4               [B, 512]              1,049,088
       LeakyReLU-5               [B, 512]                    0
          Linear-6               [B, 128]                65,664
       LeakyReLU-7               [B, 128]                    0
          Linear-8               [B, 6]                     774
================================================================
```

---

### Model2: CNN

* **Model Type**: Conv1D across Counters
* **#Parameters**: **\~40 K**
* **Model Size**: **160 KB**

#### 🔧 Architecture

```=py
----------------------------------------------------------------
      Conv1d-1 (×5 kernels)       [B, 1, 61, 1]             520
      LeakyReLU-2 (×5)            [B, 1, 61, 1]               0
      Squeeze(dim=3)              [B, 1, 61]                  0
      Concat (dim=2)              [B, 1, 305]                 0
      Squeeze(dim=1)              [B, 305]                    0
      Linear (305 → 128)          [B, 128]               39,168
      LeakyReLU                   [B, 128]                    0
      Linear (128 → 6)            [B, 6]                    774
================================================================
```

> Note: Each kernel performs a full-width convolution over `n_counters=103`, extracting 1 feature per time step (total 61 time steps).

---

### Model 3: RNN (LSTM)

* **Model Type**: 2-layer LSTM + Temporal Pooling
* **#Parameters**: **\~244 K**
* **Model Size**: **\~1 MB**

#### 🔧 Architecture

```=py
----------------------------------------------------------------
             LSTM-1             [B, 61, 200]           243,600
         Transpose-2            [B, 200, 61]                 0
             Tanh-3             [B, 200, 61]                 0
         MaxPool1d-4            [B, 200, 1]                  0
             Tanh-5             [B, 200]                     0
           Linear-6             [B, 6]                   1,206
================================================================
```

> `hidden_dim = 200`, with 2 stacked LSTM layers. Max-pooling across time dimension compresses temporal features.