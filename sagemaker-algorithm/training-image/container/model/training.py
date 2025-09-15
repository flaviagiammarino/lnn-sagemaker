from __future__ import absolute_import

import warnings
warnings.filterwarnings("ignore")

import torch
import argparse
import logging
import tarfile
import os
import sys
import random
from collections import OrderedDict
from sagemaker_training import environment
import pandas as pd
import numpy as np

from modules import Model

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))

def _set_seed(use_cuda):
    '''
    Fix the random seed, for reproducibility.
    '''
    random_seed = 0
    np.random.seed(random_seed)
    random.seed(random_seed)
    torch.manual_seed(random_seed)
    if use_cuda:
        torch.cuda.manual_seed(random_seed)


def _get_data(data_dir):
    '''
    Load the data.
    '''
    return pd.concat([pd.read_csv(os.path.join(data_dir, f)) for f in os.listdir(data_dir)], axis=0, ignore_index=True)


def _get_dataloader(data, timespans, input_length, output_length, sequence_stride, num_outputs, batch_size, **kwargs):
    '''
    Build the training dataloader.
    '''
    t = []
    x = []
    y = []
    
    for i in range(input_length, len(data) - output_length, sequence_stride):
        if timespans is not None:
            t.append(timespans[i - input_length: i])
        else:
            t.append(np.ones(input_length))
        x.append(data[i - input_length: i, :])
        y.append(data[i: i + output_length, - num_outputs:])
    
    t = np.array(t, dtype=np.float32)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    return torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(
            torch.from_numpy(t).float(),
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float()
        ),
        shuffle=True,
        batch_size=batch_size,
        **kwargs
    )


def _get_validation_dataloader(data, timespans, input_length, output_length, num_outputs, batch_size, **kwargs):
    '''
    Build the validation dataloader.
    '''
    t = []
    x = []
    y = []
    
    for i in range(input_length, len(data) - output_length, output_length):
        if timespans is not None:
            t.append(timespans[i - input_length: i])
        else:
            t.append(np.ones(input_length))
        x.append(data[i - input_length: i, :])
        y.append(data[i: i + output_length, - num_outputs:])
    
    t = np.array(t, dtype=np.float32)
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    
    return torch.utils.data.DataLoader(
        dataset=torch.utils.data.TensorDataset(
            torch.from_numpy(t).float(),
            torch.from_numpy(x).float(),
            torch.from_numpy(y).float()
        ),
        shuffle=False,
        batch_size=batch_size,
        **kwargs
    )


def _get_scalers(x):
    '''
    Calculate the scaling parameters.
    '''
    min_ = np.nanmin(x, axis=0, keepdims=True)
    max_ = np.nanmax(x, axis=0, keepdims=True)
    return min_, max_


def _mean_squared_error(y_true, y_pred):
    '''
    Calculate the mean squared error.
    '''
    return torch.mean(torch.square(y_true - y_pred))


def _mean_absolute_error(y_true, y_pred):
    '''
    Calculate the mean absolute error.
    '''
    return torch.mean(torch.abs(y_true - y_pred))


def _negative_log_likelihood(y, mu, sigma):
    '''
    Calculate the negative log-likelihood.
    '''
    return torch.mean(torch.sum(0.5 * torch.tensor(np.log(2 * np.pi)) + 0.5 * torch.log(sigma ** 2) + 0.5 * ((y - mu) ** 2) / (sigma ** 2), dim=-1))


def _train(model, dataloader, optimizer, scheduler, device):
    '''
    Run a training step.
    '''
    model.train()
    for t, x, y in dataloader:
        t, x, y = t.to(device), x.to(device), y.to(device)
        optimizer.zero_grad()
        mu, sigma = model(x, t)
        loss = _negative_log_likelihood(y, mu, sigma)
        loss.backward()
        optimizer.step()
    scheduler.step()


def _validate(model, dataloader, device):
    '''
    Run a validation step.
    '''
    model.eval()
    y_true = []
    y_pred = []
    for t, x, y in dataloader:
        t, x, y = t.to(device), x.to(device), y.to(device)
        with torch.no_grad():
            yhat, _ = model(x, t)
        y_true.append(y.reshape(y.shape[0] * y.shape[1], y.shape[2]))
        y_pred.append(yhat.reshape(yhat.shape[0] * yhat.shape[1], yhat.shape[2]))
    y_true = torch.cat(y_true, dim=0)
    y_pred = torch.cat(y_pred, dim=0)
    mse = _mean_squared_error(y_true, y_pred).item()
    mae = _mean_absolute_error(y_true, y_pred).item()
    return mse, mae


def fine_tune(args):
    '''
    Continue training an existing model.
    '''
    # extract the environment configuration
    use_cuda = torch.cuda.is_available()
    use_data_parallel = torch.cuda.device_count() > 1
    is_multi_channel = args.test_data_dir is not None
    
    if use_cuda:
        device = torch.device("cuda:0")
        kwargs = {"num_workers": 1, "pin_memory": True}
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
        kwargs = {}
    
    # load the pre-trained model
    print("\n")
    print("--------------------------------------")
    print("Loading the pre-trained model.")
    file = tarfile.open(os.path.join(args.model, 'model.tar.gz'))
    file.extractall(args.model)
    file.close()
    with open(os.path.join(args.model, "model.pth"), "rb") as f:
        checkpoint = torch.load(f, map_location="cpu")
    params = checkpoint["params"]
    model_state_dict = checkpoint["model_state_dict"]
    optimizer_state_dict = checkpoint["optimizer_state_dict"]
    model = Model(**params)
    model.load_state_dict(model_state_dict)
    model.to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    if use_data_parallel:
        model = torch.nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.parameters())
    optimizer.load_state_dict(optimizer_state_dict)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    print("\n")
    print("--------------------------------------")
    print("\n")
    print("--------------------------------------")
    print(f"Initial learning rate: {optimizer_state_dict['param_groups'][0]['lr']}")
    print("--------------------------------------")

    # extract the hyperparameters
    timespan_names = params["timespan_names"]
    input_names = params["input_names"]
    output_names = params["output_names"]
    min_ = params["min_"]
    max_ = params["max_"]
    num_inputs = params["num_inputs"]
    num_outputs = params["num_outputs"]
    input_length = params["input_length"]
    output_length = params["output_length"]
    
    # load the training data
    train_data = _get_data(args.train_data_dir)
    print("\n")
    print("--------------------------------------")
    print(f"Training on {train_data.shape[0]} samples.")
    print("--------------------------------------")
    print("\n")
    print("--------------------------------------")
    print(f"Timespans: {timespan_names}")
    print(f"Features: {input_names}")
    print(f"Targets: {output_names}")
    print("--------------------------------------")
    print("\n")

    # extract the timespans
    if len(timespan_names):
        train_timespans = train_data.loc[:, timespan_names].values
    else:
        train_timespans = None
    
    # reorder the columns
    train_data = train_data[input_names + output_names].values

    # scale the data
    train_data = (train_data - min_) / (max_ - min_)

    if is_multi_channel:
    
        # load the test data
        test_data = _get_data(args.test_data_dir)
        print("\n")
        print("--------------------------------------")
        print(f"Validating on {test_data.shape[0]} samples.")
        print("--------------------------------------")
        print("\n")
    
        # extract the timespans
        if len(timespan_names):
            test_timespans = test_data.loc[:, timespan_names].values
        else:
            test_timespans = None

        # reorder the columns
        test_data = test_data[input_names + output_names].values
    
        # scale the data
        test_data = (test_data - min_) / (max_ - min_)

    # build the dataloaders
    _set_seed(use_cuda)
    loader = _get_dataloader(
        train_data,
        train_timespans,
        input_length,
        output_length,
        args.sequence_stride,
        num_outputs,
        args.batch_size,
        **kwargs
    )

    train_loader = _get_validation_dataloader(
        train_data,
        train_timespans,
        input_length,
        output_length,
        num_outputs,
        args.batch_size,
        **kwargs
    )

    if is_multi_channel:
        test_loader = _get_validation_dataloader(
            test_data,
            test_timespans,
            input_length,
            output_length,
            num_outputs,
            args.batch_size,
            **kwargs
        )
        
    # train the model
    print("\n")
    print("--------------------------------------")
    print("Training the model.")
    _set_seed(use_cuda)
    for epoch in range(args.epochs):
        # train the model
        _train(model, loader, optimizer, scheduler, device)
        # validate the model
        train_mse, train_mae = _validate(model, train_loader, device)
        if is_multi_channel:
            valid_mse, valid_mae = _validate(model, test_loader, device)
            print(
                f'epoch: {format(1 + epoch, ".0f")} '
                f'train_mse: {format(train_mse, ",.8f")} '
                f'train_mae: {format(train_mae, ",.8f")} '
                f'valid_mse: {format(valid_mse, ",.8f")} '
                f'valid_mae: {format(valid_mae, ",.8f")}'
            )
        else:
            print(
                f'epoch: {format(1 + epoch, ".0f")} '
                f'train_mse: {format(train_mse, ",.8f")} '
                f'train_mae: {format(train_mae, ",.8f")}'
            )

    # score the model
    print("\n")
    print("--------------------------------------")
    print("Scoring the model.")
    print("train:mse " + format(train_mse, ',.8f'))
    print("train:mae " + format(train_mae, ',.8f'))
    if is_multi_channel:
        print("valid:mse " + format(valid_mse, ',.8f'))
        print("valid:mae " + format(valid_mae, ',.8f'))
    print("--------------------------------------")
    print("\n")

    # save the model
    model.eval()
    path = os.path.join(args.model_dir, "model.pth")
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    if use_data_parallel:
        model_state_dict = OrderedDict({k.replace("module.", ""): v for k, v in model_state_dict.items()})
    checkpoint = {
        "params": params,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
    }
    torch.save(checkpoint, path)


def train(args):
    '''
    Train the model.
    '''
    # extract the environment configuration
    use_cuda = torch.cuda.is_available()
    use_data_parallel = torch.cuda.device_count() > 1
    is_multi_channel = args.test_data_dir is not None
    
    if use_cuda:
        device = torch.device("cuda:0")
        kwargs = {"num_workers": 1, "pin_memory": True}
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    else:
        device = torch.device("cpu")
        kwargs = {}
    
    # load the training data
    train_data = _get_data(args.train_data_dir)
    print("\n")
    print("--------------------------------------")
    print(f"Training on {train_data.shape[0]} samples.")
    print("--------------------------------------")
    print("\n")
    
    # extract the variable names
    timespan_names = [s for s in train_data.columns if s == "ts"]
    input_names = [s for s in train_data.columns if s.startswith("x")]
    output_names = [s for s in train_data.columns if s.startswith("y")]
    print("\n")
    print("--------------------------------------")
    print(f"Timespans: {timespan_names}")
    print(f"Features: {input_names}")
    print(f"Targets: {output_names}")
    print("--------------------------------------")
    print("\n")
    
    # calculate the number of variables
    num_inputs = len(input_names)
    num_outputs = len(output_names)
    
    # extract the timespans
    if len(timespan_names):
        train_timespans = train_data.loc[:, timespan_names].values
    else:
        train_timespans = None
        
    # reorder the columns
    train_data = train_data[input_names + output_names].values
    
    # calculate the scaling parameters
    min_, max_ = _get_scalers(train_data)
    
    # scale the data
    train_data = (train_data - min_) / (max_ - min_)
    
    if is_multi_channel:
    
        # load the test data
        test_data = _get_data(args.test_data_dir)
        print("\n")
        print("--------------------------------------")
        print(f"Validating on {test_data.shape[0]} samples.")
        print("--------------------------------------")
        print("\n")
        
        # extract the timespans
        if len(timespan_names):
            test_timespans = test_data.loc[:, timespan_names].values
        else:
            test_timespans = None

        # reorder the columns
        test_data = test_data[input_names + output_names].values
        
        # scale the data
        test_data = (test_data - min_) / (max_ - min_)
    
    # extract the hyperparameters
    params = dict(
        timespan_names=timespan_names,
        input_names=input_names,
        output_names=output_names,
        min_=min_,
        max_=max_,
        num_inputs=num_inputs,
        num_outputs=num_outputs,
        input_length=args.context_length,
        output_length=args.prediction_length,
        hidden_size=args.hidden_size,
        layers=args.backbone_layers,
        units=args.backbone_units,
        activation=args.backbone_activation,
        dropout=args.backbone_dropout,
        minimal=int(args.minimal) == 1,
        no_gate=int(args.no_gate) == 1,
        use_mixed=int(args.use_mixed) == 1,
        use_ltc=int(args.use_ltc) == 1,
    )
    
    # build the model
    print("\n")
    print("--------------------------------------")
    print("Building the model.")
    _set_seed(use_cuda)
    model = Model(**params)
    model.to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")
    if use_data_parallel:
        model = torch.nn.DataParallel(model).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, args.lr_decay)
    print("--------------------------------------")
    print("\n")

    # build the dataloaders
    _set_seed(use_cuda)
    loader = _get_dataloader(
        train_data,
        train_timespans,
        args.context_length,
        args.prediction_length,
        args.sequence_stride,
        num_outputs,
        args.batch_size,
        **kwargs
    )

    train_loader = _get_validation_dataloader(
        train_data,
        train_timespans,
        args.context_length,
        args.prediction_length,
        num_outputs,
        args.batch_size,
        **kwargs
    )

    if is_multi_channel:
        test_loader = _get_validation_dataloader(
            test_data,
            test_timespans,
            args.context_length,
            args.prediction_length,
            num_outputs,
            args.batch_size,
            **kwargs
        )
        
    # train the model
    print("\n")
    print("--------------------------------------")
    print("Training the model.")
    _set_seed(use_cuda)
    for epoch in range(args.epochs):
        # train the model
        _train(model, loader, optimizer, scheduler, device)
        # validate the model
        train_mse, train_mae = _validate(model, train_loader, device)
        if is_multi_channel:
            valid_mse, valid_mae = _validate(model, test_loader, device)
            print(
                f'epoch: {format(1 + epoch, ".0f")}, '
                f'train_mse: {format(train_mse, ",.8f")} '
                f'train_mae: {format(train_mae, ",.8f")} '
                f'valid_mse: {format(valid_mse, ",.8f")} '
                f'valid_mae: {format(valid_mae, ",.8f")}'
            )
        else:
            print(
                f'epoch: {format(1 + epoch, ".0f")}, '
                f'train_mse: {format(train_mse, ",.8f")} '
                f'train_mae: {format(train_mae, ",.8f")}'
            )
            
    # score the model
    print("\n")
    print("--------------------------------------")
    print("Scoring the model.")
    print(f"train:mse {format(train_mse, ',.8f')}")
    print(f"train:mae {format(train_mae, ',.8f')}")
    if is_multi_channel:
        print(f"valid:mse {format(valid_mse, ',.8f')}")
        print(f"valid:mae {format(valid_mae, ',.8f')}")
    print("--------------------------------------")
    print("\n")
    
    # save the model
    model.eval()
    path = os.path.join(args.model_dir, "model.pth")
    model_state_dict = model.state_dict()
    optimizer_state_dict = optimizer.state_dict()
    if use_data_parallel:
        model_state_dict = OrderedDict({k.replace("module.", ""): v for k, v in model_state_dict.items()})
    checkpoint = {
        "params": params,
        "model_state_dict": model_state_dict,
        "optimizer_state_dict": optimizer_state_dict,
    }
    torch.save(checkpoint, path)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--context-length",
        type=int,
    )

    parser.add_argument(
        "--prediction-length",
        type=int,
    )
    
    parser.add_argument(
        "--sequence-stride",
        type=int,
    )
    
    parser.add_argument(
        "--backbone-layers",
        type=int,
    )
    
    parser.add_argument(
        "--backbone-units",
        type=int,
    )
    
    parser.add_argument(
        "--backbone-activation",
        type=str,
    )
    
    parser.add_argument(
        "--backbone-dropout",
        type=float,
    )
    
    parser.add_argument(
        "--hidden-size",
        type=int,
    )
    
    parser.add_argument(
        "--minimal",
        type=int,
    )
    
    parser.add_argument(
        "--no-gate",
        type=int,
    )
    
    parser.add_argument(
        "--use-ltc",
        type=int,
    )
    
    parser.add_argument(
        "--use-mixed",
        type=int,
    )
    
    parser.add_argument(
        "--lr",
        type=float,
    )
    
    parser.add_argument(
        "--lr-decay",
        type=float,
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
    )
    
    env = environment.Environment()

    if len(env.hosts) > 1:
        raise ValueError("Distributed training is not supported.")

    parser.add_argument(
        "--model-dir",
        type=str,
        default=env.model_dir
    )

    parser.add_argument(
        "--train-data-dir",
        type=str,
        default=env.channel_input_dirs["training"]
    )

    parser.add_argument(
        "--test-data-dir",
        type=str,
        default=env.channel_input_dirs["validation"] if "validation" in env.channel_input_dirs else None
    )

    parser.add_argument(
        "--model",
        type=str,
        default=env.channel_input_dirs["model"] if "model" in env.channel_input_dirs else None
    )
    
    args = parser.parse_args()
    
    if args.model is not None:
        fine_tune(args)
    else:
        train(args)
