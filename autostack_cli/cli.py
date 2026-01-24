import typer
from rich import print
from rich.console import Console
from rich.panel import Panel

from autostack_cli.commands import build
from autostack_cli.services.local_llm_service import local_llm

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

@app.command()
def setup_local():
    """
    Setup local LLM with llama.cpp (Interactive wizard)
    
    This command helps you:
    - Install llama.cpp
    - Download a GGUF model
    - Configure and start the server
    """
    console.print("\n[bold cyan]═══ AutoStack Local LLM Setup Wizard ═══[/bold cyan]\n")
    
    # Check system specs
    specs = local_llm.get_system_specs()
    console.print(f"[cyan]System Information:[/cyan]")
    console.print(f"  RAM: {specs['total_ram_gb']:.1f} GB")
    console.print(f"  Free Disk: {specs['free_disk_gb']:.1f} GB")
    console.print(f"  CPU Cores: {specs['cpu_count']}\n")
    
    # Check if llama.cpp is installed
    if not local_llm.is_llama_cpp_installed():
        console.print("[yellow]Step 1: Download llama.cpp[/yellow]")
        local_llm.install_llama_cpp_guide()
        console.print("\n[cyan]Once installed, run this command again.[/cyan]\n")
        raise typer.Exit()
    
    console.print("[green]✓ llama.cpp is installed[/green]\n")
    
    # Check if server is running
    if not local_llm._check_llama_cpp_installation():
        console.print("[yellow]Step 2: Start llama.cpp Server[/yellow]")
        console.print("\n[cyan]Open a new terminal and run:[/cyan]")
        console.print("[bold]llama-server -m model.gguf -ngl 33[/bold]\n")
        console.print("Replace 'model.gguf' with your actual model file path.\n")
        
        console.print("[cyan]Once the server is running (you'll see 'Ready to accept connections'),[/cyan]")
        console.print("[cyan]run this command again.[/cyan]\n")
        raise typer.Exit()
    
    console.print("[green]✓ llama.cpp server is running[/green]\n")
    
    # Show available models
    console.print("[yellow]Step 3: Select Your Model[/yellow]\n")
    local_llm.display_model_recommendations()
    
    console.print("\n[cyan]Available installed models:[/cyan]")
    models = local_llm.list_installed_models()
    
    if models:
        for i, model in enumerate(models, 1):
            size_info = f" ({local_llm.format_size(model['size'])})" if model.get('size') else ""
            console.print(f"  {i}. {model['name']}{size_info}")
    else:
        console.print("  No models detected. Download a GGUF model from HuggingFace:")
        console.print("  https://huggingface.co/models?sort=trending&search=gguf\n")
        console.print("[cyan]Recommended models for download:[/cyan]")
        
        tier = local_llm.get_recommended_tier()
        for model in local_llm.MODEL_RECOMMENDATIONS[tier]:
            console.print(f"  • {model['name']} ({model['size']})")
            console.print(f"    {model['description']}\n")
    
    console.print("[green]✓ Setup complete![/green]")
    console.print("\nYou can now run: [bold cyan]autostack start[/bold cyan]")
    console.print("And select option [bold]1[/bold] to use local models.\n")

@app.command()
def local_models():
    """
    Quick guide for using local LLM models with llama.cpp
    """
    console.print("\n[bold cyan]═══ Using Local Models with AutoStack ═══[/bold cyan]\n")
    
    console.print("[bold]Quick Setup:[/bold]\n")
    console.print("1. Download llama.cpp:")
    console.print("   https://github.com/ggerganov/llama.cpp/releases\n")
    
    console.print("2. Download a GGUF model (example):")
    console.print("   https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF\n")
    
    console.print("3. Start llama.cpp server:")
    console.print("   [cyan]llama-server -m tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf -ngl 33[/cyan]\n")
    
    console.print("4. In another terminal, run:")
    console.print("   [cyan]autostack start[/cyan]\n")
    
    console.print("5. Choose option [bold]1[/bold] for Local Model\n")
    
    console.print("[bold]Recommended Models by System:[/bold]\n")
    specs = local_llm.get_system_specs()
    tier = local_llm.get_recommended_tier()
    
    console.print(f"Your System: {specs['total_ram_gb']:.1f}GB RAM → [bold]{tier.upper()}[/bold] tier\n")
    
    for model in local_llm.MODEL_RECOMMENDATIONS[tier]:
        console.print(f"• [cyan]{model['name']}[/cyan] ({model['size']})")
        console.print(f"  {model['description']}\n")
    
    console.print("[bold]Commands:[/bold]")
    console.print("• [cyan]autostack setup-local[/cyan]  - Interactive setup wizard")
    console.print("• [cyan]autostack start[/cyan]        - Build your project")
    console.print("• [cyan]autostack --help[/cyan]       - Show all commands\n")

if __name__ == "__main__":
    app()
