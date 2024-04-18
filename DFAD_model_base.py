import torch.nn as nn
import torch

class DFADModel(nn.Module):
    def __init__(self):
        super(DFADModel, self).__init__()

        dropout_rate = 0.335
        leaky_relu_slope = 0.01 

        
        self.layers = nn.Sequential(
            nn.Linear(1536, 1536),
            nn.BatchNorm1d(1536),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            # nn.Sigmoid(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(1536, 1536),
            nn.BatchNorm1d(1536),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.Dropout(p=dropout_rate),

            nn.Linear(1536, 1536),
            nn.BatchNorm1d(1536),
            # nn.ReLU(),
            nn.LeakyReLU(negative_slope=leaky_relu_slope),
            nn.Dropout(p=dropout_rate),

        )

        # 
        self.output_layer = nn.Linear(1536, 1)

        #he initialization
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.output_layer.weight, mode='fan_in', nonlinearity='relu')


    def forward(self, inputs, text_inputs):
        # Example of concatenating inputs and text_inputs along the last dimension
        x = torch.cat((inputs, text_inputs), dim=-1)  # Adjust dim as needed

        x = self.layers(x)
        output = self.output_layer(x)
        return output


if __name__ == '__main__':

    model = DFADModel()  
    print(model)  