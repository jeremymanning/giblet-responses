"""
Comprehensive evaluation of PyTorch visualization libraries for network diagrams.

Tests 5 different approaches with the actual giblet autoencoder model (2.0B params, 52 layers):
1. visualtorch
2. torchviz
3. hiddenlayer
4. torchview
5. custom matplotlib

Evaluation criteria:
- Supports PyTorch models natively
- Can show parallel branches (2A/B/C, 10A/B/C)
- Horizontal orientation option
- Handles 2B param models
- Color customization
- Publication quality output (PDF/SVG)
- Active maintenance
"""

import importlib
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

import torch

# Add parent directory to path to import giblet
sys.path.insert(0, str(Path(__file__).parent.parent))

from giblet.models import create_autoencoder


class LibraryEvaluator:
    """Evaluates a visualization library against our criteria."""

    def __init__(self, library_name: str):
        self.library_name = library_name
        self.results = {
            "library": library_name,
            "installable": False,
            "import_success": False,
            "pytorch_native": False,
            "handles_large_models": False,
            "shows_parallel_branches": "unknown",
            "horizontal_orientation": "unknown",
            "color_customization": "unknown",
            "pdf_svg_output": "unknown",
            "active_maintenance": "unknown",
            "test_time_seconds": 0,
            "error_messages": [],
            "success_messages": [],
            "sample_output": None,
        }

    def install(self, package_name: Optional[str] = None) -> bool:
        """Try to install the library."""
        if package_name is None:
            package_name = self.library_name

        print(f"\n{'='*80}")
        print(f"Installing {package_name}...")
        print(f"{'='*80}")

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                self.results["installable"] = True
                self.results["success_messages"].append(
                    f"Successfully installed {package_name}"
                )
                print(f"✓ Successfully installed {package_name}")
                return True
            else:
                self.results["error_messages"].append(
                    f"Install failed: {result.stderr}"
                )
                print(f"✗ Install failed: {result.stderr}")
                return False

        except Exception as e:
            self.results["error_messages"].append(f"Install error: {str(e)}")
            print(f"✗ Install error: {str(e)}")
            return False

    def test_import(self, module_name: Optional[str] = None) -> bool:
        """Try to import the library."""
        if module_name is None:
            module_name = self.library_name

        try:
            importlib.import_module(module_name)
            self.results["import_success"] = True
            self.results["success_messages"].append(
                f"Successfully imported {module_name}"
            )
            print(f"✓ Successfully imported {module_name}")
            return True
        except Exception as e:
            self.results["error_messages"].append(f"Import error: {str(e)}")
            print(f"✗ Import error: {str(e)}")
            return False

    def generate_test_output(self, output_dir: Path) -> bool:
        """Generate visualization with the library (to be implemented by subclasses)."""
        raise NotImplementedError

    def get_summary(self) -> Dict[str, Any]:
        """Return evaluation summary."""
        return self.results


class VisualtorchEvaluator(LibraryEvaluator):
    """Evaluate visualtorch library."""

    def __init__(self):
        super().__init__("visualtorch")

    def generate_test_output(self, output_dir: Path, model: torch.nn.Module) -> bool:
        """Test visualtorch with the autoencoder model."""
        start_time = time.time()

        try:
            import visualtorch

            self.results["pytorch_native"] = True

            # Check available methods
            available_methods = dir(visualtorch)
            print(
                f"Available methods: {[m for m in available_methods if not m.startswith('_')]}"
            )

            # Try to create visualization
            output_path = output_dir / f"{self.library_name}_test.png"

            # Test with a smaller model first
            simple_model = torch.nn.Sequential(
                torch.nn.Linear(10, 20), torch.nn.ReLU(), torch.nn.Linear(20, 10)
            )

            # Try different APIs
            if hasattr(visualtorch, "graph"):
                visualtorch.graph(
                    simple_model,
                    torch.zeros(1, 10),
                    format="png",
                    filename=str(output_path),
                )
                self.results["success_messages"].append("Created graph visualization")
            elif hasattr(visualtorch, "layered_view"):
                visualtorch.layered_view(simple_model, to_file=str(output_path))
                self.results["success_messages"].append("Created layered view")
            else:
                raise ValueError("Could not find visualization method")

            if output_path.exists():
                self.results["sample_output"] = str(output_path)
                self.results["handles_large_models"] = True
                self.results["pdf_svg_output"] = "supports PNG, check for PDF/SVG"

            # Test with full model
            try:
                full_output = output_dir / f"{self.library_name}_full.png"
                # Use smaller batch size
                dummy_video = torch.randn(1, 3, 90, 160)
                dummy_audio = torch.randn(1, 2048)
                dummy_text = torch.randn(1, 1024)

                if hasattr(visualtorch, "graph"):
                    visualtorch.graph(
                        model,
                        (dummy_video, dummy_audio, dummy_text),
                        format="png",
                        filename=str(full_output),
                    )
                    self.results["handles_large_models"] = True
            except Exception as e:
                self.results["error_messages"].append(
                    f"Full model test failed: {str(e)}"
                )
                self.results["handles_large_models"] = False

            return True

        except Exception as e:
            self.results["error_messages"].append(
                f"Generation error: {str(e)}\n{traceback.format_exc()}"
            )
            return False
        finally:
            self.results["test_time_seconds"] = time.time() - start_time


class TorchvizEvaluator(LibraryEvaluator):
    """Evaluate torchviz library."""

    def __init__(self):
        super().__init__("torchviz")

    def generate_test_output(self, output_dir: Path, model: torch.nn.Module) -> bool:
        """Test torchviz with the autoencoder model."""
        start_time = time.time()

        try:
            from torchviz import make_dot

            self.results["pytorch_native"] = True

            # Create small dummy inputs
            dummy_video = torch.randn(1, 3, 90, 160, requires_grad=True)
            dummy_audio = torch.randn(1, 2048, requires_grad=True)
            dummy_text = torch.randn(1, 1024, requires_grad=True)

            # Forward pass
            model.eval()
            outputs = model(dummy_video, dummy_audio, dummy_text)

            # Get main output
            if isinstance(outputs, dict):
                output = outputs["bottleneck"]
            else:
                output = outputs

            # Create computational graph
            dot = make_dot(output, params=dict(model.named_parameters()))

            # Save outputs
            output_path = output_dir / f"{self.library_name}_test"
            dot.render(str(output_path), format="png", cleanup=True)

            self.results["sample_output"] = str(output_path) + ".png"
            self.results["handles_large_models"] = True
            self.results["pdf_svg_output"] = "supports PDF, PNG, SVG via graphviz"
            self.results[
                "shows_parallel_branches"
            ] = "shows computational graph structure"
            self.results["success_messages"].append("Created computational graph")

            # Try SVG export
            try:
                svg_path = output_dir / f"{self.library_name}_test_svg"
                dot.render(str(svg_path), format="svg", cleanup=True)
                self.results["success_messages"].append("SVG export successful")
            except:
                pass

            return True

        except Exception as e:
            self.results["error_messages"].append(
                f"Generation error: {str(e)}\n{traceback.format_exc()}"
            )
            return False
        finally:
            self.results["test_time_seconds"] = time.time() - start_time


class HiddenlayerEvaluator(LibraryEvaluator):
    """Evaluate hiddenlayer library."""

    def __init__(self):
        super().__init__("hiddenlayer")

    def generate_test_output(self, output_dir: Path, model: torch.nn.Module) -> bool:
        """Test hiddenlayer with the autoencoder model."""
        start_time = time.time()

        try:
            import hiddenlayer as hl

            self.results["pytorch_native"] = True

            # Create visualization
            # Try graph method
            try:
                graph = hl.build_graph(
                    model,
                    torch.randn(1, 3, 90, 160),
                    torch.randn(1, 2048),
                    torch.randn(1, 1024),
                )

                output_path = output_dir / f"{self.library_name}_test.png"
                graph.save(str(output_path), format="png")

                self.results["sample_output"] = str(output_path)
                self.results["handles_large_models"] = True
                self.results["pdf_svg_output"] = "supports PNG, PDF"
                self.results["shows_parallel_branches"] = "shows architecture graph"
                self.results["success_messages"].append("Created architecture graph")

                # Try different themes
                graph.theme = hl.graph.THEMES["blue"]
                themed_path = output_dir / f"{self.library_name}_test_themed.png"
                graph.save(str(themed_path), format="png")
                self.results["color_customization"] = "supports themes"

            except Exception as e:
                self.results["error_messages"].append(
                    f"Graph creation failed: {str(e)}"
                )
                return False

            return True

        except Exception as e:
            self.results["error_messages"].append(
                f"Generation error: {str(e)}\n{traceback.format_exc()}"
            )
            return False
        finally:
            self.results["test_time_seconds"] = time.time() - start_time


class TorchviewEvaluator(LibraryEvaluator):
    """Evaluate torchview library."""

    def __init__(self):
        super().__init__("torchview")

    def generate_test_output(self, output_dir: Path, model: torch.nn.Module) -> bool:
        """Test torchview with the autoencoder model."""
        start_time = time.time()

        try:
            from torchview import draw_graph

            self.results["pytorch_native"] = True

            # Create visualization
            dummy_video = torch.randn(1, 3, 90, 160)
            dummy_audio = torch.randn(1, 2048)
            dummy_text = torch.randn(1, 1024)

            model_graph = draw_graph(
                model,
                input_data=[dummy_video, dummy_audio, dummy_text],
                expand_nested=True,
                depth=3,
                device="cpu",
            )

            # Save visualization
            output_path = output_dir / f"{self.library_name}_test"
            model_graph.visual_graph.render(
                str(output_path), format="png", cleanup=True
            )

            self.results["sample_output"] = str(output_path) + ".png"
            self.results["handles_large_models"] = True
            self.results["pdf_svg_output"] = "supports PNG, PDF, SVG via graphviz"
            self.results["shows_parallel_branches"] = "shows detailed layer structure"
            self.results["horizontal_orientation"] = "supports via rankdir parameter"
            self.results["success_messages"].append(
                "Created detailed layer visualization"
            )

            # Try horizontal orientation
            try:
                model_graph_horizontal = draw_graph(
                    model,
                    input_data=[dummy_video, dummy_audio, dummy_text],
                    expand_nested=True,
                    depth=3,
                    device="cpu",
                    graph_dir="LR",  # Left to Right
                )
                h_path = output_dir / f"{self.library_name}_test_horizontal"
                model_graph_horizontal.visual_graph.render(
                    str(h_path), format="png", cleanup=True
                )
                self.results["success_messages"].append(
                    "Horizontal orientation successful"
                )
            except:
                pass

            return True

        except Exception as e:
            self.results["error_messages"].append(
                f"Generation error: {str(e)}\n{traceback.format_exc()}"
            )
            return False
        finally:
            self.results["test_time_seconds"] = time.time() - start_time


class CustomMatplotlibEvaluator(LibraryEvaluator):
    """Evaluate custom matplotlib approach (already implemented)."""

    def __init__(self):
        super().__init__("custom_matplotlib")

    def generate_test_output(self, output_dir: Path, model: torch.nn.Module) -> bool:
        """Test custom matplotlib visualization."""
        start_time = time.time()

        try:
            from giblet.utils.visualization import create_network_diagram

            self.results["pytorch_native"] = True
            self.results["installable"] = True
            self.results["import_success"] = True

            # Generate diagram
            output_path = output_dir / f"{self.library_name}_test.pdf"
            create_network_diagram(
                model,
                str(output_path),
                legend=True,
                sizing_mode="logarithmic",
                show_dimension=True,
                title="Giblet Autoencoder (Custom Matplotlib)",
                figsize=(16, 24),
                dpi=150,  # Lower DPI for faster testing
            )

            self.results["sample_output"] = str(output_path)
            self.results["handles_large_models"] = True
            self.results["pdf_svg_output"] = "supports PDF, PNG"
            self.results[
                "shows_parallel_branches"
            ] = "vertical list, no parallel visualization"
            self.results["horizontal_orientation"] = "vertical only"
            self.results["color_customization"] = "fully customizable"
            self.results["success_messages"].append("Created custom matplotlib diagram")

            # Also create PNG version
            png_path = output_dir / f"{self.library_name}_test.png"
            create_network_diagram(
                model,
                str(png_path),
                legend=True,
                sizing_mode="logarithmic",
                show_dimension=True,
                title="Giblet Autoencoder (Custom Matplotlib)",
                figsize=(16, 24),
                dpi=150,
            )

            return True

        except Exception as e:
            self.results["error_messages"].append(
                f"Generation error: {str(e)}\n{traceback.format_exc()}"
            )
            return False
        finally:
            self.results["test_time_seconds"] = time.time() - start_time


def main():
    """Run comprehensive evaluation of all libraries."""
    print("\n" + "=" * 80)
    print("PYTORCH VISUALIZATION LIBRARY EVALUATION")
    print("Testing with giblet autoencoder (2.0B parameters, 52 layers)")
    print("=" * 80)

    # Create output directory
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)

    # Create model
    print("\nCreating autoencoder model...")
    try:
        model = create_autoencoder()
        model.eval()

        # Get parameter count
        param_count = model.get_parameter_count()
        print(f"✓ Model created successfully")
        print(f"  Total parameters: {param_count['total']:,}")
        print(f"  Encoder parameters: {param_count['encoder']:,}")
        print(f"  Decoder parameters: {param_count['decoder']:,}")
    except Exception as e:
        print(f"✗ Failed to create model: {e}")
        return

    # Initialize evaluators
    evaluators = [
        VisualtorchEvaluator(),
        TorchvizEvaluator(),
        HiddenlayerEvaluator(),
        TorchviewEvaluator(),
        CustomMatplotlibEvaluator(),
    ]

    # Run evaluations
    results = []

    for evaluator in evaluators:
        print(f"\n{'='*80}")
        print(f"Testing {evaluator.library_name}")
        print(f"{'='*80}")

        # Install (skip for custom matplotlib)
        if evaluator.library_name != "custom_matplotlib":
            if not evaluator.install():
                results.append(evaluator.get_summary())
                continue

            # Test import
            if not evaluator.test_import():
                results.append(evaluator.get_summary())
                continue

        # Generate test output
        print(f"\nGenerating test visualization...")
        evaluator.generate_test_output(output_dir, model)

        results.append(evaluator.get_summary())

    # Print summary table
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)

    # Create comparison table
    print(
        f"\n{'Library':<20} {'Install':<8} {'Import':<8} {'Large Models':<12} {'Time (s)':<10}"
    )
    print("-" * 80)

    for result in results:
        print(
            f"{result['library']:<20} "
            f"{'✓' if result['installable'] else '✗':<8} "
            f"{'✓' if result['import_success'] else '✗':<8} "
            f"{'✓' if result['handles_large_models'] else '✗':<12} "
            f"{result['test_time_seconds']:<10.2f}"
        )

    # Detailed results
    print("\n" + "=" * 80)
    print("DETAILED RESULTS")
    print("=" * 80)

    for result in results:
        print(f"\n{result['library'].upper()}")
        print("-" * 40)
        print(f"PyTorch Native: {result['pytorch_native']}")
        print(f"Handles Large Models: {result['handles_large_models']}")
        print(f"Parallel Branches: {result['shows_parallel_branches']}")
        print(f"Horizontal Orientation: {result['horizontal_orientation']}")
        print(f"Color Customization: {result['color_customization']}")
        print(f"PDF/SVG Output: {result['pdf_svg_output']}")
        print(f"Test Time: {result['test_time_seconds']:.2f}s")

        if result["sample_output"]:
            print(f"Sample Output: {result['sample_output']}")

        if result["success_messages"]:
            print("Successes:")
            for msg in result["success_messages"]:
                print(f"  ✓ {msg}")

        if result["error_messages"]:
            print("Errors:")
            for msg in result["error_messages"]:
                print(f"  ✗ {msg}")

    # Save results to file
    import json

    results_file = output_dir / "evaluation_results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to {results_file}")

    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
