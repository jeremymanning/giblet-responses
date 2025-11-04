"""
Tests for network visualization utilities.

All tests use REAL MultimodalAutoencoder model (2B parameters).
NO MOCKS - tests verify actual functionality with real models.
"""

import pytest
import torch
from pathlib import Path

# Import project modules using absolute imports
from giblet.models.autoencoder import MultimodalAutoencoder
from giblet.utils.visualization import (
    create_network_diagram,
    create_model_summary,
    _get_layer_info,
    _calculate_layer_size,
    _get_layer_color
)


@pytest.fixture
def real_model():
    """Create a REAL MultimodalAutoencoder for testing."""
    model = MultimodalAutoencoder(
        video_frames_per_tr=1,  # Single frame for fast testing
        audio_frames_per_tr=1   # Single time step for fast testing
    )
    return model


class TestNetworkDiagram:
    """Test network diagram creation with REAL model."""

    def test_create_pdf_diagram(self, real_model, test_data_dir):
        """Test creating PDF network diagram from real model."""
        output_path = test_data_dir / "network.pdf"

        result = create_network_diagram(
            real_model,
            str(output_path),
            legend=True,
            sizing_mode='logarithmic',
            show_dimension=True
        )

        # Verify file was created
        assert output_path.exists(), "PDF file should be created"
        assert output_path.stat().st_size > 0, "PDF file should not be empty"
        assert result == output_path, "Should return output path"

    def test_create_png_diagram(self, real_model, test_data_dir):
        """Test creating PNG network diagram from real model."""
        output_path = test_data_dir / "network.png"

        result = create_network_diagram(
            real_model,
            str(output_path),
            legend=True,
            sizing_mode='logarithmic',
            show_dimension=True
        )

        # Verify file was created
        assert output_path.exists(), "PNG file should be created"
        assert output_path.stat().st_size > 0, "PNG file should not be empty"
        assert result == output_path, "Should return output path"

    def test_logarithmic_sizing(self, real_model, test_data_dir):
        """Test network diagram with logarithmic sizing."""
        output_path = test_data_dir / "network_log.pdf"

        result = create_network_diagram(
            real_model,
            str(output_path),
            sizing_mode='logarithmic'
        )

        assert output_path.exists(), "File should be created with logarithmic sizing"
        assert output_path.stat().st_size > 0, "File should not be empty"

    def test_linear_sizing(self, real_model, test_data_dir):
        """Test network diagram with linear sizing."""
        output_path = test_data_dir / "network_linear.pdf"

        result = create_network_diagram(
            real_model,
            str(output_path),
            sizing_mode='linear'
        )

        assert output_path.exists(), "File should be created with linear sizing"
        assert output_path.stat().st_size > 0, "File should not be empty"

    def test_with_legend(self, real_model, test_data_dir):
        """Test network diagram with legend enabled."""
        output_path = test_data_dir / "network_legend.pdf"

        result = create_network_diagram(
            real_model,
            str(output_path),
            legend=True
        )

        assert output_path.exists(), "File should be created with legend"
        assert output_path.stat().st_size > 0, "File should not be empty"

    def test_without_legend(self, real_model, test_data_dir):
        """Test network diagram without legend."""
        output_path = test_data_dir / "network_no_legend.pdf"

        result = create_network_diagram(
            real_model,
            str(output_path),
            legend=False
        )

        assert output_path.exists(), "File should be created without legend"
        assert output_path.stat().st_size > 0, "File should not be empty"

    def test_with_dimensions(self, real_model, test_data_dir):
        """Test network diagram with dimension labels."""
        output_path = test_data_dir / "network_dims.pdf"

        result = create_network_diagram(
            real_model,
            str(output_path),
            show_dimension=True
        )

        assert output_path.exists(), "File should be created with dimensions"
        assert output_path.stat().st_size > 0, "File should not be empty"

    def test_without_dimensions(self, real_model, test_data_dir):
        """Test network diagram without dimension labels."""
        output_path = test_data_dir / "network_no_dims.pdf"

        result = create_network_diagram(
            real_model,
            str(output_path),
            show_dimension=False
        )

        assert output_path.exists(), "File should be created without dimensions"
        assert output_path.stat().st_size > 0, "File should not be empty"

    def test_custom_title(self, real_model, test_data_dir):
        """Test network diagram with custom title."""
        output_path = test_data_dir / "network_title.pdf"
        custom_title = "Custom Network Architecture"

        result = create_network_diagram(
            real_model,
            str(output_path),
            title=custom_title
        )

        assert output_path.exists(), "File should be created with custom title"
        assert output_path.stat().st_size > 0, "File should not be empty"

    def test_output_directory_creation(self, real_model, test_data_dir):
        """Test that output directory is created if it doesn't exist."""
        output_path = test_data_dir / "subdir" / "network.pdf"

        result = create_network_diagram(
            real_model,
            str(output_path)
        )

        assert output_path.parent.exists(), "Parent directory should be created"
        assert output_path.exists(), "File should be created"
        assert output_path.stat().st_size > 0, "File should not be empty"

    def test_multiple_diagrams_same_model(self, real_model, test_data_dir):
        """Test creating multiple diagrams from the same model (reproducibility)."""
        output_path1 = test_data_dir / "network1.pdf"
        output_path2 = test_data_dir / "network2.pdf"

        # Create two diagrams with same settings
        create_network_diagram(real_model, str(output_path1))
        create_network_diagram(real_model, str(output_path2))

        # Both should exist
        assert output_path1.exists(), "First diagram should exist"
        assert output_path2.exists(), "Second diagram should exist"

        # Both should have content
        assert output_path1.stat().st_size > 0, "First diagram should not be empty"
        assert output_path2.stat().st_size > 0, "Second diagram should not be empty"

        # They should have similar sizes (allowing for small variations)
        size1 = output_path1.stat().st_size
        size2 = output_path2.stat().st_size
        assert abs(size1 - size2) / max(size1, size2) < 0.1, \
            "Diagrams should have similar sizes"


class TestModelSummary:
    """Test model summary generation with REAL model."""

    def test_create_summary_string(self, real_model):
        """Test creating model summary as string."""
        summary = create_model_summary(real_model)

        # Verify summary contains expected information
        assert isinstance(summary, str), "Summary should be a string"
        assert len(summary) > 0, "Summary should not be empty"
        assert "Total parameters" in summary, "Should show total parameters"
        assert "Layer Name" in summary, "Should have header"
        assert "encoder" in summary.lower(), "Should mention encoder"
        assert "decoder" in summary.lower(), "Should mention decoder"

    def test_create_summary_file(self, real_model, test_data_dir):
        """Test creating model summary and saving to file."""
        output_path = test_data_dir / "model_summary.txt"

        summary = create_model_summary(real_model, str(output_path))

        # Verify file was created
        assert output_path.exists(), "Summary file should be created"
        assert output_path.stat().st_size > 0, "Summary file should not be empty"

        # Verify file content matches returned string
        file_content = output_path.read_text()
        assert file_content == summary, "File content should match returned string"

    def test_summary_parameter_count(self, real_model):
        """Test that summary shows correct parameter count."""
        summary = create_model_summary(real_model)

        # Get actual parameter count
        param_info = real_model.get_parameter_count()
        total_params = param_info['total']

        # Check that parameter count appears in summary
        # (formatted with commas)
        formatted_params = f"{total_params:,}"
        assert formatted_params in summary, \
            f"Summary should contain parameter count {formatted_params}"


class TestLayerInfo:
    """Test layer information extraction with REAL model."""

    def test_get_layer_info(self, real_model):
        """Test extracting layer information from real model."""
        layers = _get_layer_info(real_model)

        # Should have many layers
        assert len(layers) > 0, "Should extract layers from model"

        # Each layer should have required fields
        for layer in layers:
            assert 'name' in layer, "Layer should have name"
            assert 'type' in layer, "Layer should have type"
            assert 'params' in layer, "Layer should have param count"
            assert 'module' in layer, "Layer should have module reference"
            assert layer['params'] > 0, "Layer should have parameters"

    def test_layer_types_present(self, real_model):
        """Test that expected layer types are present in real model."""
        layers = _get_layer_info(real_model)
        layer_types = [layer['type'] for layer in layers]

        # Should have different layer types
        assert 'Linear' in layer_types, "Should have Linear layers"
        assert 'Conv2d' in layer_types, "Should have Conv2d layers"
        assert 'Conv1d' in layer_types, "Should have Conv1d layers"
        assert 'BatchNorm1d' in layer_types or 'BatchNorm2d' in layer_types, \
            "Should have BatchNorm layers"

    def test_encoder_decoder_layers_present(self, real_model):
        """Test that both encoder and decoder layers are present."""
        layers = _get_layer_info(real_model)
        layer_names = [layer['name'] for layer in layers]

        # Should have encoder layers
        encoder_layers = [name for name in layer_names if 'encoder' in name]
        assert len(encoder_layers) > 0, "Should have encoder layers"

        # Should have decoder layers
        decoder_layers = [name for name in layer_names if 'decoder' in name]
        assert len(decoder_layers) > 0, "Should have decoder layers"


class TestLayerSizing:
    """Test layer sizing calculations."""

    def test_logarithmic_sizing(self):
        """Test logarithmic sizing calculation."""
        # Test various parameter counts
        size_small = _calculate_layer_size(100, 'logarithmic')
        size_medium = _calculate_layer_size(10000, 'logarithmic')
        size_large = _calculate_layer_size(1000000, 'logarithmic')

        # Sizes should be in valid range
        assert 20 <= size_small <= 200, "Small layer size should be in range"
        assert 20 <= size_medium <= 200, "Medium layer size should be in range"
        assert 20 <= size_large <= 200, "Large layer size should be in range"

        # Sizes should increase with parameter count
        assert size_small < size_medium < size_large, \
            "Sizes should increase with parameter count"

    def test_linear_sizing(self):
        """Test linear sizing calculation."""
        size_small = _calculate_layer_size(100, 'linear')
        size_medium = _calculate_layer_size(10000, 'linear')
        size_large = _calculate_layer_size(1000000, 'linear')

        # Sizes should be in valid range
        assert 20 <= size_small <= 200, "Small layer size should be in range"
        assert 20 <= size_medium <= 200, "Medium layer size should be in range"
        assert 20 <= size_large <= 200, "Large layer size should be in range"

    def test_zero_params(self):
        """Test sizing for zero parameters."""
        size = _calculate_layer_size(0, 'logarithmic')
        assert size == 20, "Zero params should return minimum size"


class TestLayerColors:
    """Test layer color assignment."""

    def test_video_encoder_color(self):
        """Test color for video encoder layers."""
        color = _get_layer_color('encoder.video_encoder.conv1', 'Conv2d')
        assert len(color) == 4, "Color should be RGBA tuple"
        assert all(0 <= c <= 1 for c in color), "Color values should be in [0, 1]"
        # Should be blue-ish for video
        assert color[2] > color[1], "Video should have blue component"

    def test_audio_encoder_color(self):
        """Test color for audio encoder layers."""
        color = _get_layer_color('encoder.audio_encoder.conv1', 'Conv1d')
        assert len(color) == 4, "Color should be RGBA tuple"
        # Should be orange-ish for audio
        assert color[0] > color[2], "Audio should have red component"

    def test_text_encoder_color(self):
        """Test color for text encoder layers."""
        color = _get_layer_color('encoder.text_encoder.fc1', 'Linear')
        assert len(color) == 4, "Color should be RGBA tuple"
        # Should be green-ish for text
        assert color[1] > color[0], "Text should have green component"

    def test_bottleneck_color(self):
        """Test color for bottleneck layers."""
        color = _get_layer_color('encoder.to_bottleneck.0', 'Linear')
        assert len(color) == 4, "Color should be RGBA tuple"
        # Should be purple-ish for bottleneck
        assert color[0] > 0.5 and color[2] > 0.5, "Bottleneck should be purple"

    def test_decoder_color(self):
        """Test color for decoder layers."""
        color = _get_layer_color('decoder.layer7.0', 'Linear')
        assert len(color) == 4, "Color should be RGBA tuple"
        # Should be orange-ish for decoder
        assert color[0] > color[2], "Decoder should have red component"


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '--tb=short'])
