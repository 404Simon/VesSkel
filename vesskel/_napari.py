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
    from napari.layers import Image  # noqa: F401

from vesskel.config import (
    ExtractionConfig,
    OutputConfig,
    PipelineConfig,
    load_pipeline_config,
    save_pipeline_config,
)
from vesskel.pipeline import analyze_binary_image


class VesselAnalysisWidget(Container):
    """Analysis configuration widget."""

    _CONFIG_FILTER = "JSON Files (*.json);;All Files (*)"

    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._setup_ui()

    def _setup_ui(self):
        # ---------- extraction parameters (magicgui) ----------
        def _extraction_params(
            image: "napari.layers.Image",  # noqa: F821
            extract_branches: bool = False,
            extract_branch_text: bool = False,
            extract_nodes: bool = False,
            extract_summary: bool = False,
            include_fractal: bool = False,
            include_vessel_radius: bool = False,
            junction_cleanup: bool = False,
            cleanup_threshold_factor: float = 2.5,
            fill_holes: bool = False,
            closing_iterations: int = 0,
            max_hole_size: int = 0,
            show_preprocessed: bool = False,
        ) -> None:
            return None

        extraction_gui = magicgui(
            _extraction_params,
            image={"label": "Input image"},
            extract_branches={"annotation": bool, "value": False},
            extract_branch_text={"annotation": bool, "value": False},
            extract_summary={"annotation": bool, "value": False},
            include_fractal={"annotation": bool, "value": False},
            junction_cleanup={"annotation": bool, "value": False},
            cleanup_threshold_factor={
                "annotation": float,
                "value": 2.5,
                "widget_type": "FloatSpinBox",
                "min": 1.0,
                "max": 10.0,
                "step": 0.1,
            },
            fill_holes={"annotation": bool, "value": False},
            closing_iterations={
                "annotation": int,
                "value": 0,
                "widget_type": "SpinBox",
                "min": 0,
                "max": 10,
                "step": 1,
            },
            max_hole_size={
                "annotation": int,
                "value": 0,
                "widget_type": "SpinBox",
                "min": 0,
                "max": 100000,
                "step": 100,
            },
            show_preprocessed={"annotation": bool, "value": False},
        )

        # ---------- output parameters (magicgui) ----------
        def _output_params(
            write_skeleton_npy: bool = True,
            write_skeleton_png: bool = False,
            write_summary_csv: bool = True,
            write_branch_csv: bool = False,
            write_node_csv: bool = False,
            write_radius: bool = False,
        ) -> None:
            return None

        output_gui = magicgui(_output_params)

        self._extraction_gui = extraction_gui
        self._output_gui = output_gui
        self.image_widget = extraction_gui.image

        # ============================================================
        # Extraction Layers
        # ============================================================
        extraction_group = Container()
        extraction_group.label = "Extraction Layers"

        self.extract_branches_widget = extraction_gui.extract_branches
        self.extract_branches_widget.label = "Extract branches"

        self.extract_branch_text_widget = extraction_gui.extract_branch_text
        self.extract_branch_text_widget.label = "Add branch labels"

        self.extract_summary_widget = extraction_gui.extract_summary
        self.extract_summary_widget.label = "Extract summary statistics"

        self.extract_nodes_widget = extraction_gui.extract_nodes
        self.extract_nodes_widget.label = "Extract node features"

        extraction_group.append(self.extract_branches_widget)
        extraction_group.append(self.extract_branch_text_widget)
        extraction_group.append(self.extract_summary_widget)
        extraction_group.append(self.extract_nodes_widget)

        # ============================================================
        # Advanced Features
        # ============================================================
        advanced_group = Container()
        advanced_group.label = "Advanced Features"

        self.include_fractal_widget = extraction_gui.include_fractal
        self.include_fractal_widget.label = "Include fractal dimension (slow)"

        advanced_group.append(self.include_fractal_widget)

        self.include_vessel_radius_widget = extraction_gui.include_vessel_radius
        self.include_vessel_radius_widget.label = "Compute vessel radius and diameter"

        advanced_group.append(self.include_vessel_radius_widget)

        self.junction_cleanup_widget = extraction_gui.junction_cleanup
        self.junction_cleanup_widget.label = "Collapse triangle junction artifacts"

        self.cleanup_threshold_widget = extraction_gui.cleanup_threshold_factor
        self.cleanup_threshold_widget.label = "Cleanup threshold factor"

        advanced_group.append(self.junction_cleanup_widget)
        advanced_group.append(self.cleanup_threshold_widget)

        self.fill_holes_widget = extraction_gui.fill_holes
        self.fill_holes_widget.label = "Fill holes in segmentation"

        self.closing_iterations_widget = extraction_gui.closing_iterations
        self.closing_iterations_widget.label = "Closing iterations"

        self.max_hole_size_widget = extraction_gui.max_hole_size
        self.max_hole_size_widget.label = "Max hole size (pixels)"

        self.show_preprocessed_widget = extraction_gui.show_preprocessed
        self.show_preprocessed_widget.label = "Show preprocessed binary layer"

        advanced_group.append(self.fill_holes_widget)
        advanced_group.append(self.closing_iterations_widget)
        advanced_group.append(self.max_hole_size_widget)
        advanced_group.append(self.show_preprocessed_widget)

        # ============================================================
        # Output Settings (CLI file export options)
        # ============================================================
        output_group = Container()
        output_group.label = "Output Settings"

        self.write_skeleton_npy_widget = output_gui.write_skeleton_npy
        self.write_skeleton_npy_widget.label = "Write skeleton (.npy)"

        self.write_skeleton_png_widget = output_gui.write_skeleton_png
        self.write_skeleton_png_widget.label = "Write skeleton (.png)"

        self.write_summary_csv_widget = output_gui.write_summary_csv
        self.write_summary_csv_widget.label = "Write summary CSV"

        self.write_branch_csv_widget = output_gui.write_branch_csv
        self.write_branch_csv_widget.label = "Write branch CSV"

        self.write_node_csv_widget = output_gui.write_node_csv
        self.write_node_csv_widget.label = "Write node CSV"

        self.write_radius_widget = output_gui.write_radius
        self.write_radius_widget.label = "Write radius matrix (.npy)"

        output_group.append(self.write_skeleton_npy_widget)
        output_group.append(self.write_skeleton_png_widget)
        output_group.append(self.write_summary_csv_widget)
        output_group.append(self.write_branch_csv_widget)
        output_group.append(self.write_node_csv_widget)
        output_group.append(self.write_radius_widget)

        # ============================================================
        # Configuration Management
        # ============================================================
        config_group = Container()
        config_group.label = "Configuration"

        self.load_btn = PushButton(text="Load Config")
        self.load_btn.clicked.connect(self._on_load_config)

        self.save_btn = PushButton(text="Save Config")
        self.save_btn.clicked.connect(self._on_save_config)

        config_group.append(self.load_btn)
        config_group.append(self.save_btn)

        # ============================================================
        # Analyze Button
        # ============================================================
        self.analyze_btn = PushButton(text="Analyze Vessels")
        self.analyze_btn.clicked.connect(self._on_analyze)

        # ============================================================
        # Assemble widget
        # ============================================================
        self.append(self.image_widget)
        self.append(extraction_group)
        self.append(advanced_group)
        self.append(output_group)
        self.append(config_group)
        self.append(self.analyze_btn)

    # ------------------------------------------------------------------
    # Config get / set (full PipelineConfig)
    # ------------------------------------------------------------------

    def _get_current_pipeline_config(self) -> PipelineConfig:
        junction_cleanup = self.junction_cleanup_widget.value
        return PipelineConfig(
            extraction=ExtractionConfig(
                branches=self.extract_branches_widget.value,
                branch_text=self.extract_branch_text_widget.value,
                nodes=self.extract_nodes_widget.value,
                summary=self.extract_summary_widget.value,
                fractal_dimension=self.include_fractal_widget.value,
                vessel_radius=self.include_vessel_radius_widget.value,
                junction_cleanup=junction_cleanup,
                cleanup_threshold_factor=self.cleanup_threshold_widget.value,
                fill_holes=self.fill_holes_widget.value,
                closing_iterations=self.closing_iterations_widget.value,
                max_hole_size=self.max_hole_size_widget.value,
                show_preprocessed=self.show_preprocessed_widget.value,
            ),
            output=OutputConfig(
                write_skeleton_npy=self.write_skeleton_npy_widget.value,
                write_skeleton_png=self.write_skeleton_png_widget.value,
                write_summary_csv=self.write_summary_csv_widget.value,
                write_branch_csv=self.write_branch_csv_widget.value,
                write_node_csv=self.write_node_csv_widget.value,
                write_radius=self.write_radius_widget.value,
            ),
        )

    def _set_pipeline_config(self, config: PipelineConfig) -> None:
        e = config.extraction
        self.extract_branches_widget.value = e.branches
        self.extract_branch_text_widget.value = e.branch_text
        self.extract_nodes_widget.value = e.nodes
        self.extract_summary_widget.value = e.summary
        self.include_fractal_widget.value = e.fractal_dimension
        self.include_vessel_radius_widget.value = e.vessel_radius
        self.junction_cleanup_widget.value = e.junction_cleanup
        self.cleanup_threshold_widget.value = e.cleanup_threshold_factor
        self.fill_holes_widget.value = e.fill_holes
        self.closing_iterations_widget.value = e.closing_iterations
        self.max_hole_size_widget.value = e.max_hole_size
        self.show_preprocessed_widget.value = e.show_preprocessed

        o = config.output
        self.write_skeleton_npy_widget.value = o.write_skeleton_npy
        self.write_skeleton_png_widget.value = o.write_skeleton_png
        self.write_summary_csv_widget.value = o.write_summary_csv
        self.write_branch_csv_widget.value = o.write_branch_csv
        self.write_node_csv_widget.value = o.write_node_csv
        self.write_radius_widget.value = o.write_radius

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

    def _on_load_config(self) -> None:
        """Load configuration from file."""
        try:
            config_path, _ = QFileDialog.getOpenFileName(
                None,
                "Load Pipeline Configuration",
                "",
                self._CONFIG_FILTER,
            )
            if not config_path:
                return

            pipeline_config = load_pipeline_config(Path(config_path))
            self._set_pipeline_config(pipeline_config)
            show_info("Configuration loaded")
        except (ValueError, OSError) as e:
            show_error(f"Failed to load config: {e}")

    def _on_save_config(self) -> None:
        """Save current configuration to file."""
        try:
            pipeline_config = self._get_current_pipeline_config()
            config_path, _ = QFileDialog.getSaveFileName(
                None,
                "Save Pipeline Configuration",
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
            pipeline_config = self._get_current_pipeline_config()
            result = analyze_binary_image(
                image=img.data, base_name=img.name, config=pipeline_config
            )
            elapsed = time.perf_counter() - t0

            n_fg = int((img.data > 0).sum())
            n_skel = int(result.skeleton.sum())
            show_info(f"Analysis: {n_fg} → {n_skel} skeleton pixels in {elapsed:.3f}s")

            # Always add skeleton layer first.
            self.viewer.add_layer(
                Layer.create(
                    result.skeleton,
                    {"name": f"{img.name}_skeleton"},
                    "labels",
                )
            )

            # -- optional: show preprocessed binary layer ----------------
            if (
                pipeline_config.extraction.show_preprocessed
                and result.preprocessed_binary is not None
            ):
                self.viewer.add_layer(
                    Layer.create(
                        result.preprocessed_binary,
                        {"name": f"{img.name}_preprocessed"},
                        "labels",
                    )
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
