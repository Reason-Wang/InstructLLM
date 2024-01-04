# Understanding the Instruction Mixture for Large Language Model Fine-tuning

## Benchmark Performance and Alignment Skills

### Benchmarks

#### LLaMA-2-7B

| Data |     ARC      | Wino-grande  |     PIQA     |     MMLU     |     Race     |  Hella-Swag  |   Average    | Human<br>@1  | Eval<br>@10  |
| :--- | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| None |    43.09     |    69.53     |    77.97     |    40.81     |    39.23     |    57.20     |    54.64     |    13.72     |    21.34     |
| A    |    47.78     |    67.64     |    78.24     |    42.19     | <u>44.50</u> |  **61.09**   |    56.91     |    13.48     |    17.07     |
| C    |    46.08     |    69.46     | <u>78.50</u> |    40.99     |    41.05     | <u>60.96</u> |    56.17     |    16.22     |  **24.39**   |
| P    | <u>49.57</u> |  **71.43**   |  **79.00**   |  **45.98**   |    43.45     |    59.44     |  **58.15**   |     4.63     |     7.93     |
| AC   |    47.10     |    66.93     |    78.13     |    40.42     |    44.21     |    59.70     |    56.08     |  **17.50**   |      25      |
| AP   |    48.38     |    70.01     |    78.07     |    43.84     |    42.87     |    58.46     |    56.94     |    13.84     |    17.68     |
| CP   |    47.95     | <u>71.27</u> |    78.40     | <u>44.91</u> |    44.40     |    60.69     | <u>57.94</u> | <u>16.77</u> |    20.12     |
| ACP  |  **49.66**   |    68.03     |    77.86     |    43.52     |  **44.59**   |    58.73     |    57.07     |    15.98     | <u>23.78</u> |

#### LLaMA-2-13B

| Data |     ARC      | Wino-grande  |     PIQA     |     MMLU     |     Race     |  Hella-Swag  |   Average    | Human<br>@1  | Eval<br>@10  |
| :--- | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: | :----------: |
| None |    48.55     |    71.90     |    79.16     |  **52.12**   |    40.67     |    60.12     |    58.75     |    15.43     |    26.22     |
| A    |    54.10     |    71.19     |    80.03     |    47.86     |  **47.08**   |  **65.58**   |    60.97     |    15.06     |    20.73     |
| C    |    49.66     |    73.40     |  **80.79**   | <u>51.50</u> |    45.36     |    63.63     |    60.72     |    17.87     |    24.39     |
| P    |    54.27     | <u>74.19</u> |    80.03     |    50.30     | <u>45.55</u> |    62.46     | <u>61.13</u> |     0.30     |     1.83     |
| AC   |    51.62     |    68.75     | <u>80.58</u> |    48.68     |    44.40     |    62.97     |    59.50     |    17.07     | <u>27.44</u> |
| AP   | <u>54.79</u> |    71.74     |    80.30     |    51.15     |    45.17     |    62.72     |    60.98     |     8.29     |    14.63     |
| CP   |  **55.38**   |  **74.59**   |    80.52     |    51.42     | <u>45.55</u> | <u>63.85</u> |  **61.89**   | <u>18.23</u> |      25      |
| ACP  |    54.44     |    71.51     |    80.03     |    49.98     |  **47.08**   |    63.14     |    61.03     |  **20.24**   |  **32.93**   |

### Alignment Skills

#### LLaMA-2-7B

| Data |    Corr.    |    Fact.    |    Comm.    |   Compr.    |   Compl.    |  Insight.   |    Read.    |    Conc.    |    Avg.     |
| :--- | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| A    |    47.6     |  **55.4**   |    58.8     | <u>54.8</u> | <u>48.0</u> |  **50.4**   |  **88.0**   |    81.6     | <u>60.6</u> |
| C    |    48.8     |    52.0     |    58.4     |    52.0     |    40.2     |    46.2     |    83.8     |    78.4     |    57.4     |
| P    |    47.2     |    40.0     |    48.8     |    38.4     |    29.0     |    30.4     |    64.4     |    68.6     |    45.8     |
| AC   | <u>49.0</u> | <u>54.4</u> |  **59.6**   |  **56.4**   |  **48.2**   | <u>49.8</u> | <u>86.6</u> |  **85.6**   |  **61.2**   |
| AP   |    48.4     |    51.4     |    57.6     |    52.6     |    45.0     |    46.0     |    84.2     |    80.8     |    58.2     |
| CP   |    47.0     |    49.6     |    54.2     |    48.8     |    36.2     |    41.8     |    78.2     |    77.2     |    54.2     |
| ACP  |  **50.4**   |    53.0     | <u>59.0</u> |    53.8     |    47.2     |    46.8     |    85.0     | <u>81.8</u> |    59.6     |

#### LLaMA-2-13B

| Data |    Corr.    |    Fact.    |    Comm.    |   Compr.    |   Compl.    |  Insight.   |    Read.    |    Conc.    |    Avg.     |
| :--- | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: | :---------: |
| A    |    53.6     | <u>58.8</u> | <u>63.8</u> | <u>60.0</u> | <u>47.6</u> |  **55.2**   |  **89.2**   | <u>84.0</u> | <u>64.0</u> |
| C    |  **57.2**   | <u>58.8</u> |    61.0     |    57.8     |    43.8     |    52.4     |    85.6     |    82.2     |    62.4     |
| P    |    49.4     |    42.4     |    51.8     |    42.0     |    28.2     |    32.0     |    66.8     |    70.4     |    47.8     |
| AC   | <u>55.6</u> |  **61.0**   |  **66.6**   |  **61.2**   |  **51.4**   | <u>54.0</u> | <u>88.4</u> |  **86.6**   |  **65.6**   |
| AP   |    53.0     |    55.4     |    60.6     |    56.2     |    47.0     |    48.0     |    85.0     |    83.4     |    61.0     |
| CP   |    53.0     |    53.2     |    57.4     |    53.4     |    39.0     |    45.2     |    81.2     |    82.6     |    58.2     |
| ACP  |    51.6     |    55.6     |    61.8     |    57.0     |    47.0     |    48.6     |    87.0     |    83.0     |    61.4     |

## Training

The following is a command using `deepspeed` with 4 GPUs, training LLaMA-2-7B on Alpaca dataset.

```bash
deepspeed --num_gpus=4 train.py \
  --model_name_or_path meta-llama/Llama-2-7b \
  --deepspeed src/deepspeed_z3_config.json \
  --architecture causal \
  --output_dir /ckpts/Llama-2-7b-A \
  --save_strategy no \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --num_p3_data 0 \
  --num_code_data 0 \
  --num_instruction_data 20000 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 16 \
  --num_train_epochs 2 \
  --gradient_checkpointing False \
  --bf16 \
  --logging_steps 10
```
