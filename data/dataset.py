import torch
import transformers
from torch.utils.data import Dataset

from data.utils import load_bias_data

# PROMPT_DICT = {
#     "prompt_input": (
#         "Below is an instruction that describes a task, paired with an input that provides further context. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
#     ),
#     "prompt_no_input": (
#         "Below is an instruction that describes a task. "
#         "Write a response that appropriately completes the request.\n\n"
#         "### Instruction:\n{instruction}\n\n### Response:"
#     ),
# }

PROMPT_DICT = {
    "prompt_input": (
        "{instruction}\n\n {input}\n\n"
    ),
    "prompt_no_input": (
        "{instruction}\n\n"
    ),
}


class Seq2SeqDataset(Dataset):
    def __init__(self, sources, targets):
        super(Seq2SeqDataset, self).__init__()

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):
        return self.sources[item], self.targets[item]


class Seq2SeqCollator(object):
    def __init__(self, tokenizer, intruction_length, output_length):
        self.tokenizer = tokenizer
        self.intruction_length = intruction_length
        self.output_length = output_length

    def __call__(self, batch):
        sources = [ex[0] for ex in batch]
        targets = [ex[1] for ex in batch]

        inputs = self.tokenizer(
            sources,
            max_length=self.intruction_length,
            return_tensors='pt',
            padding=True,
            truncation=True
        )

        labels = self.tokenizer(
            targets,
            max_length=self.output_length,
            return_tensors='pt',
            padding=True,
            truncation=True
        ).input_ids

        labels[labels == self.tokenizer.pad_token_id] = -100
        inputs['labels'] = labels

        return inputs


class CausalLMDataset(Dataset):
    def __init__(self, tokenizer, sources, targets, max_length):
        super(CausalLMDataset, self).__init__()
        self.tokenizer = tokenizer
        self.sources = sources
        self.targets = targets
        self.max_length = max_length
        # self.instruction_prompt = "Instruction: {instruction} Response: "
        # self.response_prompt = "{response}"
        self.has_print = False

    def _tokenize(self, text):
        return self.tokenizer(text, truncation=True, max_length=self.max_length)

    def __len__(self):
        return len(self.sources)

    def __getitem__(self, item):
        full_prompt = self.sources[item] + ' ' + self.targets[item]
        user_prompt = self.sources[item]

        # set a prompt for inputs
        # full_prompt = self.instruction_prompt.format(instruction=self.sources[item]) + self.response_prompt.format(response=self.targets[item])
        # user_prompt = self.response_prompt.format(response=self.targets[item])

        if not self.has_print:
            print(full_prompt, user_prompt)
            self.has_print = True

        tokenized_full_prompt = self._tokenize(full_prompt)
        if (tokenized_full_prompt["input_ids"][-1] != self.tokenizer.eos_token_id
            and len(tokenized_full_prompt["input_ids"]) < self.max_length):
            tokenized_full_prompt["input_ids"].append(self.tokenizer.eos_token_id)
            tokenized_full_prompt["attention_mask"].append(1)

        tokenized_user_prompt = self._tokenize(user_prompt)["input_ids"]
        user_prompt_len = len(tokenized_user_prompt)
        labels = [-100 if i < user_prompt_len else token_id for i, token_id in enumerate(tokenized_full_prompt["input_ids"])]

        return torch.tensor(tokenized_full_prompt["input_ids"]), \
            torch.tensor(tokenized_full_prompt["attention_mask"]), \
            torch.tensor(labels)


class CausalLMCollator(object):
    def __init__(self, tokenizer, max_length, padding_side='right'):
        super().__init__()
        self.tokenizer = tokenizer
        self.tokenizer.pad_token_id = tokenizer.eos_token_id
        self.max_length = max_length
        self.padding_side = padding_side

    def __call__(self, instances):
        input_ids = [e[0] for e in instances]
        attention_masks = [e[1] for e in instances]
        labels = [e[2] for e in instances]

        if self.padding_side == 'left':
            # pad all inputs from left side, this can help batch generation
            reversed_input_ids = [ids.flip(0) for ids in input_ids]
            reversed_attention_masks = [mask.flip(0) for mask in attention_masks]
            reversed_labels = [label.flip(0) for label in labels]

            padded_input_ids = torch.nn.utils.rnn.pad_sequence(reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            padded_input_ids = padded_input_ids.flip(1)
            padded_attention_masks = torch.nn.utils.rnn.pad_sequence(reversed_attention_masks, batch_first=True, padding_value=0)
            padded_attention_masks = padded_attention_masks.flip(1)
            padded_labels = torch.nn.utils.rnn.pad_sequence(reversed_labels, batch_first=True, padding_value=-100)
            padded_labels = padded_labels.flip(1)
        elif self.padding_side == 'right':
            padded_input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)
            padded_attention_masks = torch.nn.utils.rnn.pad_sequence(attention_masks, batch_first=True, padding_value=0)
            padded_labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        else:
            raise RuntimeError("Padding side must 'left' or 'right'.")

        return {"input_ids": padded_input_ids, "attention_mask": padded_attention_masks, "labels": padded_labels}

    def _mask(self, lens, max_length):
        mask = torch.arange(max_length).expand(len(lens), max_length) < torch.tensor(lens).unsqueeze(1)
        return mask


if __name__ == '__main__':
    sources = ['List 5 reasons why learn to code.', 'what is it?']
    targets = ['Improve communication skills.', 'Not like a human.']
    tokenizer = transformers.AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-590M")
    dataset = CausalLMDataset(tokenizer, sources, targets, max_length=512)
    collator = CausalLMCollator(tokenizer, max_length=512)
    print(dataset[1])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=4
    )
    for data in dataloader:
        print(data)

    instructions, responses = load_bias_data()
    dataset = CausalLMDataset(tokenizer, instructions, responses, max_length=512)
    print(dataset[1])
    dataloader = torch.utils.data.DataLoader(
        dataset,
        collate_fn=collator,
        batch_size=4
    )
    for data in dataloader:
        print(data)
        break



