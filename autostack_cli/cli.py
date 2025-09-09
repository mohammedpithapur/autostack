import typer
from rich import print
from rich.console import Console
from rich.panel import Panel

from autostack_cli.commands import build

# Initialize Typer app
app = typer.Typer(
    name="autostack",
    help="AI-powered CLI for generating full-stack SaaS applications",
    add_completion=False,
)

# Create console for rich output
console = Console()

def version_callback(value: bool):
    """Print version information."""
    if value:
        print(Panel.fit(
            "[bold blue]AutoStack[/bold blue] [yellow]v0.1.0[/yellow]",
            title="Version",
            border_style="blue",
        ))
        raise typer.Exit()

@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version information.",
        callback=version_callback,
        is_eager=True,
    ),
):
    pass

app.command()(build.start)

if __name__ == "__main__":
    app() 
