import click
import asyncio
import time
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich.live import Live

from dummy import models, model_name_from_path, Biologist, DataLoader, Generator, Orchestrator

console = Console()

@click.group()
def cli():
    """Multi-Expert Peptide Pipeline CLI"""
    pass

@cli.command()
@click.option('--data', required=True, help='Dataset file path')
def load_data(data):
    """Load data using DataLoader"""
    loader = DataLoader()
    with console.status(f"[bold green]Loading data from {data}...", spinner="dots"):
        loader.load(data)
        time.sleep(1) # Simulate work
    console.print(Panel(f"[bold blue]Successfully loaded data from {data}[/bold blue]", title="DataLoader Status"))

@cli.command()
@click.option('--input_file', required=True, help='Input file for biologist')
def process_biologist(input_file):
    """Process data using Biologist expert"""
    bio = Biologist()
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Biologist processing...", total=100)
        while not progress.finished:
            progress.update(task, advance=20)
            time.sleep(0.3)
    
    bio.process(input_file)
    table = Table(title="Biologist Processing Results")
    table.add_column("Input", style="magenta")
    table.add_column("Status", style="green")
    table.add_row(input_file, "Completed Successfully")
    console.print(table)

@cli.command()
@click.option('--params', default="default_params", help='Generation parameters')
def generate_peptides(params):
    """Generate peptides using Generator"""
    gen = Generator()
    with console.status("[bold yellow]Generating peptides...", spinner="earth"):
        peptides = gen.generate(params)
        time.sleep(1.5)
    
    table = Table(title=f"Generated Peptides (Params: {params})")
    table.add_column("ID", justify="right", style="cyan", no_wrap=True)
    table.add_column("Sequence", style="magenta")
    
    for i, p in enumerate(peptides, 1):
        table.add_row(str(i), p)
    
    console.print(table)

@cli.command()
def orchestrate():
    """Run the full Orchestrator pipeline"""
    orch = Orchestrator()
    steps = ["Initializing", "Loading Data", "Biologist Analysis", "Sequence Generation", "Finalizing"]
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
    ) as progress:
        overall_task = progress.add_task("[bold green]Overall Pipeline", total=len(steps))
        
        for step in steps:
            step_task = progress.add_task(f"[white]{step}...", total=100)
            for _ in range(10):
                time.sleep(0.1)
                progress.update(step_task, advance=10)
            progress.remove_task(step_task)
            progress.update(overall_task, advance=1)
            
    orch.run_pipeline()
    console.print(Panel(f"[bold green]Full Pipeline Complete[/bold green]", title="Orchestrator Final Status"))

@cli.command()
@click.option('--log', default='INFO', help='Logging level')
@click.option('--data', required=True, prompt='Dataset file path:', help='Dataset file path')
@click.option('--save', required=True, prompt='Output model path:', help='Output model path')
@click.option('--model', required=True, prompt=f'Model type ({", ".join(m.name for m in models)}):', help='Model name')
@click.option('-batch', default=32, help='Batch size')
@click.option('--epochs', default=10, help='Number of epochs')
@click.option('--lr', default=0.001, help='Learning rate')
def train(log, data, save, model, batch, epochs, lr):
    """Train a model"""
    console.print(f"[bold]Training with following parameters:[/bold]")
    table = Table()
    table.add_column("Parameter")
    table.add_column("Value")
    table.add_row("Log", log)
    table.add_row("Data", data)
    table.add_row("Save", save)
    table.add_row("Model", model)
    table.add_row("Batch", str(batch))
    table.add_row("Epochs", str(epochs))
    table.add_row("LR", str(lr))
    console.print(table)

@cli.command()
@click.argument('model')
@click.option('--log', default='INFO', help='Logging level')
@click.option('-json_params', default=None, help='Path to JSON file with model parameters')
@click.option('--top_k', default=5, help='Number of top predictions to return')
@click.option('-size', default=8, help='Size of the sequence to predict')
def run(model, log, json_params, top_k, size):
    """Run model prediction"""
    console.print(f"[bold green]Running model: {model_name_from_path(model)}[/bold green]")
    table = Table()
    table.add_column("Parameter")
    table.add_column("Value")
    table.add_row("Log Level", log)
    table.add_row("JSON Params", str(json_params))
    table.add_row("Top K", str(top_k))
    table.add_row("Size", str(size))
    console.print(table)

@cli.command()
def tui():
    """Launch the Textual TUI interface"""
    from tui import PeptideApp
    app = PeptideApp()
    app.run()

if __name__ == "__main__":
    cli()