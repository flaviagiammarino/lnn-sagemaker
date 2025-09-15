# Copyright 2019-2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
from __future__ import absolute_import

import warnings

warnings.filterwarnings("ignore")

import os
import io
import pandas as pd
import numpy as np
import torch

from sagemaker_inference import default_inference_handler
from sagemaker_pytorch_serving_container.modules import Model

# extract the environment configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_data_parallel = torch.cuda.device_count() > 1
kwargs = {"num_workers": 1, "pin_memory": True} if torch.cuda.is_available() else {}

class Dataset(torch.utils.data.Dataset):
    '''
    Define a custom dataset for processing variable length sequences.
    '''
    def __init__(self, t, x, transform=None):
        self.t = t
        self.x = x
        self.transform = transform
    
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()
        return [
            torch.from_numpy(self.t[idx]).float(),
            torch.from_numpy(self.x[idx]).float(),
        ]


def _get_test_dataloader(data, timespans, input_length, output_length, **kwargs):
    '''
    Build the dataloader using the custom dataset.
    '''
    t = []
    x = []
    for i in range(input_length, len(data) + output_length, output_length):
        if timespans is not None:
            t.append(timespans[i - input_length: i])
        else:
            t.append(np.ones(input_length))
        x.append(data[i - input_length: i, :])
    return torch.utils.data.DataLoader(
        dataset=Dataset(t, x),
        shuffle=False,
        batch_size=1,
        **kwargs
    )


class DefaultPytorchInferenceHandler(default_inference_handler.DefaultInferenceHandler):
    
    def default_model_fn(self, model_dir):
        '''
        Load the model.
        '''
        with open(os.path.join(model_dir, "model.pth"), "rb") as f:
            checkpoint = torch.load(f, map_location="cpu")
        params = checkpoint["params"]
        state_dict = checkpoint["model_state_dict"]
        model = Model(**params)
        model.load_state_dict(state_dict)
        model.to(device)
        if use_data_parallel:
            model = torch.nn.DataParallel(model).to(device)
        return model
    
    def default_input_fn(self, input_data, content_type):
        '''
        Load the data.
        '''
        return pd.read_csv(io.StringIO(input_data)).astype(float)
    
    def default_predict_fn(self, data, model):
        '''
        Generate the predictions.
        '''
        # extract the inputs
        if use_data_parallel:
            timespan_names = model.module.timespan_names
            input_names = model.module.input_names
            output_names = model.module.output_names
            min_ = model.module.min_
            max_ = model.module.max_
            input_length = model.module.input_length
            output_length = model.module.output_length
            num_inputs = model.module.num_inputs
            num_outputs = model.module.num_outputs
        else:
            timespan_names = model.timespan_names
            input_names = model.input_names
            output_names = model.output_names
            min_ = model.min_
            max_ = model.max_
            input_length = model.input_length
            output_length = model.output_length
            num_inputs = model.num_inputs
            num_outputs = model.num_outputs

        # extract the timespans
        if len(timespan_names):
            timespans = data.loc[:, timespan_names].values
        else:
            timespans = None
        
        # reorder the columns
        data = data[input_names + output_names].values
        
        # scale the data
        data = (data - min_) / (max_ - min_)
        
        # create the dataloader
        dataloader = _get_test_dataloader(
            data,
            timespans,
            input_length,
            output_length,
            **kwargs
        )
        
        # generate the model predictions
        mu = torch.from_numpy(np.nan * np.ones((input_length, num_outputs))).float().to(device)
        sigma = torch.from_numpy(np.nan * np.ones((input_length, num_outputs))).float().to(device)
        for t, x in dataloader:
            t, x = t.to(device), x.to(device)
            with torch.no_grad():
                mu_, sigma_ = model(x, t)
            mu = torch.cat([mu, mu_.reshape(mu_.size(0) * mu_.size(1), mu_.size(2))], dim=0)
            sigma = torch.cat([sigma, sigma_.reshape(sigma_.size(0) * sigma_.size(1), sigma_.size(2))], dim=0)

        # transform the model predictions back to the original scale
        mu = min_[:, - num_outputs:] + (max_[:, - num_outputs:] - min_[:, - num_outputs:]) * mu.detach().cpu().numpy()
        sigma = (max_[:, - num_outputs:] - min_[:, - num_outputs:]) * sigma.detach().cpu().numpy()

        # organize the model predictions in a data frame
        prediction = {}
        for i in range(num_outputs):
            prediction[output_names[i] + '_mean'] = mu[:, i]
            prediction[output_names[i] + '_std'] = sigma[:, i]
        prediction = pd.DataFrame(prediction)
        
        return prediction
    
    def default_output_fn(self, prediction, accept):
        '''
        Return the predictions.
        '''
        csv_buffer = io.StringIO()
        prediction.to_csv(csv_buffer, index=False)
        return csv_buffer.getvalue()