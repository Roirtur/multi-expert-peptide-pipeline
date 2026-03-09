import click
import time
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel

from dummy import Biologist, DataLoader, Generator, Orchestrator, Chemist

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
@click.option('')
@click.option('--verbose', is_flag=True, help='Run the full Orchestrator pipeline')
def orchestrate():
    """Run the full Orchestrator pipeline"""
    orch = Orchestrator(Generator(), Chemist(), Biologist())
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
    orch.run(iterations=5, population_size=50, top_k=5)
    console.print(Panel(f"[bold green]Full Pipeline Complete[/bold green]", title="Orchestrator Final Status"))

@cli.command()
def tui():
    """Launch the Textual TUI interface"""
    from tui.app import PeptideApp
    app = PeptideApp()
    app.run()

if __name__ == "__main__":
    cli()
