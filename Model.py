import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)

# Training and optimisation
class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        # Pytorch optimisation step
        self.optimizer = optim.Adam(model.parameters(), lr = self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, new_state, game_over):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype = torch.long)
        reward = torch.tensor(reward, dtype = torch. float)
        new_state = torch.tensor(new_state, dtype = torch.float)
        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            new_state = torch.unsqueeze(new_state, 0)
            game_over = (game_over, )

        # Getting the predicted Q values with the current state
        predict = self.model(state)

        # new_Q = reward + gamma * max(next_predicted Q value) -> executed only if not game_over
        target = predict.clone()
        for index in range(len(game_over)):
            new_Q = reward[index]
            if not game_over[index]:
                new_Q = reward[index] + self.gamma * torch.max(self.model(new_state[index]))

            target[index][torch.argmax(action[index]).item()] = new_Q

        self.optimizer.zero_grad()
        loss = self.criterion(target, predict)
        loss.backward()
        
        self.optimizer.step()
