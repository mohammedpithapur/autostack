"""
Simplified service for local LLM models using llama-cpp-python.
"""
import subprocess
import sys
import psutil
from pathlib import Path
from typing import Optional, Dict
from rich.console import Console
from rich.prompt import Prompt, Confirm
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class LocalLLMService:
    """Service for managing local LLM models via llama-cpp-python."""
    
    # Model recommendations with HuggingFace repo info
    MODEL_RECOMMENDATIONS = {
        "low": [  # < 8GB RAM
            {
                "name": "TinyLlama 1.1B",
                "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "size": "0.6 GB",
                "description": "Fast, decent quality - great for testing"
            },
            {
                "name": "Phi-2 (2.7B)",
                "repo_id": "TheBloke/phi-2-GGUF",
                "filename": "phi-2.Q4_K_M.gguf",
                "size": "1.6 GB",
                "description": "Better quality, still fast"
            },
        ],
        "medium": [  # 8-16GB RAM
            {
                "name": "TinyLlama 1.1B",
                "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "size": "0.6 GB",
                "description": "Fast, good for quick iterations"
            },
            {
                "name": "Mistral 7B",
                "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                "size": "4.1 GB",
                "description": "Excellent code generation"
            },
        ],
        "high": [  # > 16GB RAM
            {
                "name": "TinyLlama 1.1B",
                "repo_id": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF",
                "filename": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
                "size": "0.6 GB",
                "description": "Fast development iterations"
            },
            {
                "name": "Mistral 7B",
                "repo_id": "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
                "filename": "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                "size": "4.1 GB",
                "description": "High quality code generation"
            },
        ]
    }
    
    def __init__(self):
        """Initialize local LLM service."""
        self.model = None
        self.current_model_name = None
        self.models_dir = Path.home() / ".autostack" / "models"
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def get_system_specs(self) -> Dict:
        """Get system specifications."""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                "total_ram_gb": memory.total / (1024 ** 3),
                "available_ram_gb": memory.available / (1024 ** 3),
                "free_disk_gb": disk.free / (1024 ** 3),
                "cpu_count": psutil.cpu_count(),
            }
        except Exception:
            return {
                "total_ram_gb": 8,
                "available_ram_gb": 4,
                "free_disk_gb": 50,
                "cpu_count": 4,
            }
    
    def get_recommended_tier(self) -> str:
        """Determine which model tier to recommend."""
        specs = self.get_system_specs()
        total_ram = specs["total_ram_gb"]
        
        if total_ram < 8:
            return "low"
        elif total_ram < 16:
            return "medium"
        else:
            return "high"
    
    def _check_llama_cpp_python(self) -> bool:
        """Check if llama-cpp-python is installed."""
        try:
            import llama_cpp
            return True
        except ImportError:
            return False
    
    def install_llama_cpp_python(self) -> bool:
        """Install llama-cpp-python package."""
        console.print("\n[cyan]Installing llama-cpp-python...[/cyan]")
        console.print("[yellow]This may take a few minutes...[/yellow]\n")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "llama-cpp-python", "--upgrade"
            ])
            console.print("[green]✓ Successfully installed llama-cpp-python[/green]")
            return True
        except Exception as e:
            console.print(f"[red]Failed to install: {e}[/red]")
            return False
    
    def select_and_load_model(self) -> Optional[str]:
        """Interactive model selection and loading."""
        # Check if llama-cpp-python is installed
        if not self._check_llama_cpp_python():
            console.print("\n[yellow]llama-cpp-python is required for local models[/yellow]")
            if Confirm.ask("Install llama-cpp-python now?", default=True):
                if not self.install_llama_cpp_python():
                    return None
            else:
                return None
        
        # Get system specs
        specs = self.get_system_specs()
        tier = self.get_recommended_tier()
        
        console.print(f"\n[bold cyan]═══ Local Model Setup ═══[/bold cyan]")
        console.print(f"System RAM: {specs['total_ram_gb']:.1f} GB\n")
        
        # Show recommended models
        recommended = self.MODEL_RECOMMENDATIONS[tier]
        console.print("[bold]Recommended models:[/bold]\n")
        
        for i, model in enumerate(recommended, 1):
            console.print(f"{i}. [cyan]{model['name']}[/cyan] ({model['size']})")
            console.print(f"   {model['description']}\n")
        
        # Get user choice
        while True:
            try:
                choice = Prompt.ask("Select a model", default="1")
                idx = int(choice) - 1
                
                if 0 <= idx < len(recommended):
                    selected = recommended[idx]
                    break
                else:
                    console.print(f"[red]Please enter 1-{len(recommended)}[/red]")
            except ValueError:
                console.print("[red]Please enter a number[/red]")
        
        # Download and load the model
        console.print(f"\n[cyan]Preparing {selected['name']}...[/cyan]")
        model_path = self.download_model_from_hf(
            selected['repo_id'], 
            selected['filename']
        )
        
        if model_path:
            if self.load_model(model_path):
                self.current_model_name = selected['name']
                return f"local:{selected['name']}"
        
        return None
    
    def download_model_from_hf(self, repo_id: str, filename: str) -> Optional[Path]:
        """Download model from HuggingFace."""
        try:
            from huggingface_hub import hf_hub_download
            
            console.print(f"[cyan]Downloading from HuggingFace: {repo_id}[/cyan]")
            console.print("[yellow]This may take a few minutes...[/yellow]\n")
            
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                task = progress.add_task("Downloading...", total=None)
                
                model_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    cache_dir=str(self.models_dir)
                )
                
                progress.update(task, completed=True)
            
            console.print(f"[green]✓ Model downloaded[/green]\n")
            return Path(model_path)
            
        except ImportError:
            console.print("[red]huggingface-hub package is required[/red]")
            if Confirm.ask("Install huggingface-hub?", default=True):
                try:
                    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface-hub"])
                    return self.download_model_from_hf(repo_id, filename)
                except:
                    console.print("[red]Failed to install huggingface-hub[/red]")
                    return None
            return None
        except Exception as e:
            console.print(f"[red]Failed to download: {e}[/red]")
            return None
    
    def load_model(self, model_path: Path) -> bool:
        """Load a model from disk."""
        try:
            from llama_cpp import Llama
            
            console.print(f"[cyan]Loading model...[/cyan]")
            
            # Increase context window to handle larger prompts
            # Use 16K context for code generation
            self.model = Llama(
                model_path=str(model_path),
                n_ctx=16384,  # 16K context window
                n_gpu_layers=0,  # CPU only
                verbose=False
            )
            
            console.print("[green]✓ Model loaded successfully[/green]\n")
            return True
            
        except Exception as e:
            console.print(f"[red]Failed to load model: {e}[/red]")
            return False
    
    def generate_code(
        self,
        prompt: str,
        model: str,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Generate code using the loaded model."""
        if not self.model:
            raise ValueError("No model loaded. Please select a model first.")
        
        try:
            # Build messages
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            # Generate response with reasonable token limits
            # Leave room for prompt tokens in the 16K context
            response = self.model.create_chat_completion(
                messages=messages,
                temperature=0.2,
                max_tokens=8192,  # 8K for response, leaves room for prompt
            )
            
            return response["choices"][0]["message"]["content"]
            
        except Exception as e:
            raise ValueError(f"Error generating code: {str(e)}")


# Create singleton instance
local_llm = LocalLLMService()
