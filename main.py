from pathlib import Path
import click
import asyncio
import logging
from peptide_pipepline import *
from dummy import *

@click.command()
@click.option('--log', default='INFO', help='Logging level')
@click.option('--data', required=True, prompt='Dataset file path:', help='Dataset file path')
@click.option('--save', required=True, prompt='Output model path:', help='Output model path')
@click.option('--model', required=True, prompt=f'Model type ({", ".join(m.name for m in models)}):', help='Model name')
@click.option('-batch', default=32, help='Batch size')
@click.option('--epochs', default=10, help='Number of epochs')
@click.option('--lr', default=0.001, help='Learning rate')
def train(log, data, save, model, batch, epochs, lr):
    logging.basicConfig(level=getattr(logging, log.upper()))
    logging.info(f"Processing input file: {data}")
    logging.info(f"Writing output to: {save}")
    logging.info(f"Using model: {model}")
    logging.info(f"Using batch size: {batch}")
    logging.info(f"Training for {epochs} epochs")



@click.command()
@click.argument('model')

@click.option('--log', default='INFO', help='Logging level')
@click.option('-json_params', default=None, help='Path to JSON file with model parameters')
@click.option('--top_k', default=5, help='Number of top predictions to return')
@click.option('-size', default=8, help='Size of the sequence to predict')
    
def run(model, log, json_params, top_k, size):
    
    logging.info(f"Running model: {model_name_from_path(model)}")
    logging.info(f"Using logging level: {log}")
    logging.info(f"Using JSON parameters from: {json_params}")
    logging.info(f"Returning top {top_k} predictions")
    logging.info(f"Predicting sequences of size: {size}")

    