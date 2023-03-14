# Lightning extension: Hivemind












Collaborative Training tries to solve the need for top-tier multi-GPU servers by allowing you to train across unreliable machines,
such as local machines or even preemptible cloud compute across the internet.

Under the hood, we use [Hivemind](https://github.com/learning-at-home/hivemind) which provides de-centralized training across the internet.

To use Collaborative Training, you need to first this extension.

```bash
git clone https://github.com/Lightning-AI/lightning-Hivemind.git
cd lightning-Hivemind
pip install .
```

The `HivemindStrategy` accumulates gradients from all processes that are collaborating until they reach a `target_batch_size`. By default, we use the batch size
of the first batch to determine what each local machine batch contributes towards the `target_batch_size`. Once the `target_batch_size` is reached, an optimizer step
is made on all processes.

When using `HivemindStrategy` note that you cannot use gradient accumulation (`accumulate_grad_batches`). This is because Hivemind manages accumulation internally.

```py
from pytorch_lightning import Trainer
from lightning_hivemind.strategy import HivemindStrategy

trainer = Trainer(strategy=HivemindStrategy(target_batch_size=8192), accelerator="gpu", devices=1)
```

Followed by:

```bash
python train.py
# Other machines can connect running the same command:
# INITIAL_PEERS=... python train.py
# or passing the peers to the strategy:"
# HivemindStrategy(initial_peers=...)"
```

A helper message is printed once your training begins, which shows you how to start training on other machines using the same code.
