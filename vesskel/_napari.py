"""Napari widget for vessel analysis."""

from __future__ import annotations

import time
from pathlib import Path
from typing import TYPE_CHECKING

from magicgui import magicgui
from magicgui.widgets import Container, PushButton
from napari.layers import Layer
from napari.utils.notifications import show_error, show_info
from qtpy.QtWidgets import QFileDialog

if TYPE_CHECKING:
    # These imports are only used for annotations and are therefore
    # guarded by TYPE_CHECKING to avoid runtime import-time coupling.
    from napari.layers import Image

from vesskel.config import (
    ExtractionConfig,
    OutputConfig,
    PipelineConfig,
    load_pipeline_config,
    save_pipeline_config,
)
from vesskel.pipeline import analyze_binary_image


class VesselAnalysisWidget(Container):
    """Analysis widget with configurable extraction."""

    _CONFIG_FILTER = "JSON Files (*.json);;All Files (*)"

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._setup_ui()

    def _setup_ui(self):
        def _params(
            image: "napari.layers.Image",
            extract_branches: bool = True,
            extract_branch_text: bool = True,
            extract_summary: bool = True,
            include_fractal: bool = False,
        ) -> None:
            return None

        params_gui = magicgui(
            _params,
            image={"label": "Input image"},
            extract_branches={"annotation": bool, "value": True},
            extract_branch_text={"annotation": bool, "value": True},
            extract_summary={"annotation": bool, "value": True},
            include_fractal={"annotation": bool, "value": False},
        )

        self._params_gui = params_gui
        self.image_widget = params_gui.image

        # === Layer Selection Section ===
        layer_group = Container()
        layer_group.label = "Extraction Layers"

        self.extract_branches_widget = params_gui.extract_branches
        self.extract_branches_widget.label = "Extract branches"

        self.extract_branch_text_widget = params_gui.extract_branch_text
        self.extract_branch_text_widget.label = "Add branch labels"

        self.extract_summary_widget = params_gui.extract_summary
        self.extract_summary_widget.label = "Extract summary statistics"

        layer_group.append(self.extract_branches_widget)
        layer_group.append(self.extract_branch_text_widget)
        layer_group.append(self.extract_summary_widget)

        # === Advanced Features Section ===
        advanced_group = Container()
        advanced_group.label = "Advanced Features"

        self.include_fractal_widget = params_gui.include_fractal
        self.include_fractal_widget.label = "Include fractal dimension (slow)"

        advanced_group.append(self.include_fractal_widget)

        # === Config Management Section ===
        config_group = Container()
        config_group.label = "Configuration"

        self.load_btn = PushButton(text="Load Config")
        self.load_btn.clicked.connect(self._on_load_config)

        self.save_btn = PushButton(text="Save Config")
        self.save_btn.clicked.connect(self._on_save_config)

        config_group.append(self.load_btn)
        config_group.append(self.save_btn)

        # === Analyze Button ===
        self.analyze_btn = PushButton(text="Analyze Vessels")
        self.analyze_btn.clicked.connect(self._on_analyze)

        # === Add all sections to widget ===
        # add the image widget (from the params gui)
        self.append(self.image_widget)
        self.append(layer_group)
        self.append(advanced_group)
        self.append(config_group)
        self.append(self.analyze_btn)

    def _get_current_config(self) -> ExtractionConfig:
        """Get current configuration from widget state."""
        return ExtractionConfig(
            branches=self.extract_branches_widget.value,
            branch_text=self.extract_branch_text_widget.value,
            summary=self.extract_summary_widget.value,
            fractal_dimension=self.include_fractal_widget.value,
        )

    def _set_config(self, config: ExtractionConfig) -> None:
        """Apply configuration to widget state."""
        self.extract_branches_widget.value = config.branches
        self.extract_branch_text_widget.value = config.branch_text
        self.extract_summary_widget.value = config.summary
        self.include_fractal_widget.value = config.fractal_dimension

    def _on_load_config(self) -> None:
        """Load configuration from file."""
        try:
            config_path, _ = QFileDialog.getOpenFileName(
                None,
                "Load Extraction Configuration",
                "",
                self._CONFIG_FILTER,
            )
            if not config_path:
                return

            pipeline_config = load_pipeline_config(Path(config_path))
            self._set_config(pipeline_config.extraction)
            show_info("Configuration loaded")
        except (ValueError, OSError) as e:
            show_error(f"Failed to load config: {e}")

    def _on_save_config(self) -> None:
        """Save current configuration to file."""
        try:
            config = self._get_current_config()
            pipeline_config = PipelineConfig(extraction=config, output=OutputConfig())
            config_path, _ = QFileDialog.getSaveFileName(
                None,
                "Save Extraction Configuration",
                "",
                self._CONFIG_FILTER,
            )
            if not config_path:
                return

            save_pipeline_config(pipeline_config, Path(config_path))
            show_info(f"Configuration saved to {config_path}")
        except (ValueError, OSError) as e:
            show_error(f"Failed to save config: {e}")

    def _on_analyze(self) -> None:
        """Execute analysis with current settings."""
        img = self.image_widget.value
        if img is None:
            show_info("Please select an image layer")
            return

        try:
            t0 = time.perf_counter()
            config = self._get_current_config()
            pipeline_config = PipelineConfig(extraction=config, output=OutputConfig())
            result = analyze_binary_image(
                image=img.data, base_name=img.name, config=pipeline_config
            )
            elapsed = time.perf_counter() - t0

            n_fg = int((img.data > 0).sum())
            n_skel = int(result.skeleton.sum())
            show_info(
                f"Analysis: {n_fg} → {n_skel} skeleton pixels " f"in {elapsed:.3f}s"
            )

            # Always add skeleton layer first.
            self.viewer.add_layer(
                Layer.create(result.skeleton, {"name": f"{img.name}_skeleton"}, "labels")
            )

            for data, meta, layer_type in result.layers:
                try:
                    layer = Layer.create(data, meta, layer_type)
                    self.viewer.add_layer(layer)
                except Exception as e:
                    show_info(
                        f"Failed to add layer {meta.get('name', '<unnamed>')}: {e}"
                    )
        except (ValueError, RuntimeError, OSError) as e:
            show_error(f"Analysis failed: {e}")
