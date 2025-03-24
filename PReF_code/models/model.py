import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class Model(torch.nn.Module):
    def __init__(self, n_pairs, n_users, feature_dim=16, normalize=False, bias=False, model_name="Qwen/Qwen2.5-0.5B"):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, torch_dtype="auto", device_map="auto")
        self.tokenizer.padding_side = "left"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = self.model.device
        self.linear_head = torch.nn.Linear(self.model.config.hidden_size, feature_dim, dtype=torch.bfloat16)
        self.linear_head.to(self.model.device)
        
        self.bias = bias
        if bias:
            users = torch.randn(n_users, feature_dim-1, dtype=torch.bfloat16)
        else:
            users = torch.randn(n_users, feature_dim, dtype=torch.bfloat16)
        self.normalize = normalize
        if normalize:
            users = users / torch.linalg.norm(users, axis=-1, keepdim=True)
        self.users = torch.nn.Parameter(users, requires_grad=True)

    def _forward(self, inputs):
        tokenizer_output = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)
        token_ids = tokenizer_output.input_ids.to(self.device)
        attention_mask = tokenizer_output.attention_mask.to(self.device)
        outputs = self.model(token_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden = outputs.hidden_states[-1][:, -1:, :]
        hidden = hidden.squeeze(1)
        pairs = self.linear_head(hidden)
        return pairs
        
    def forward(self, input1, input2, return_features=False):
        pairs1 = self._forward(input1)
        pairs2 = self._forward(input2)

        if self.normalize:
            pairs1, pairs2 = pairs1/torch.linalg.norm(pairs1, axis=-1, keepdim=True), pairs2/torch.linalg.norm(pairs2, axis=-1, keepdim=True)
        
        if self.bias:
            users = torch.cat([torch.ones(self.users.shape[0], 1, dtype=torch.bfloat16, device=self.users.device), self.users], dim=1)
        else:
            users = self.users
        preference_parameters = torch.matmul(pairs1 - pairs2, users.transpose(1,0))
        if return_features:
            return preference_parameters, pairs1 - pairs2
        return preference_parameters

    def loss(self, preference_observations, input1, input2):
        pred_preferences = self.forward(input1, input2)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_preferences.reshape(1,-1),
            preference_observations.reshape(1,-1),
            reduction='none'
        )
        return loss

    def set_users(self, data):
        if self.normalize:
            data = data / torch.linalg.norm(data, axis=-1, keepdim=True)
        if self.bias: #remove the first column of data
            data = data[:, 1:]
        self.users.data = data.cuda().float().to(torch.bfloat16) 